# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Training and evaluation for score-based generative models. """

import gc
import io
import os
import time
from typing import Any
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import flax
import tqdm
import flax.jax_utils as flax_utils
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan
import logging
import functools
from flax.metrics import tensorboard
from flax.training import checkpoints
# Keep the import below for registering all model definitions
import cs
from models import ddpm, ncsnv2, ncsnpp
import losses
import sampling
import utils
from models import utils as mutils
import datasets
import sde_lib
from absl import flags
from PIL import Image, ImageDraw
import piq
import torch
import torchvision.transforms as transforms
import shutil

FLAGS = flags.FLAGS


def train(config, workdir):
  """Runs the training pipeline.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """

  # Create directories for experimental logs
  sample_dir = os.path.join(workdir, "samples")
  tf.io.gfile.makedirs(sample_dir)

  rng = jax.random.PRNGKey(config.seed)
  tb_dir = os.path.join(workdir, "tensorboard")
  tf.io.gfile.makedirs(tb_dir)
  if jax.host_id() == 0:
    writer = tensorboard.SummaryWriter(tb_dir)

  # Initialize model.
  rng, step_rng = jax.random.split(rng)
  score_model, init_model_state, initial_params = mutils.init_model(step_rng, config)
  optimizer = losses.get_optimizer(config).create(initial_params)
  state = mutils.State(step=0, optimizer=optimizer, lr=config.optim.lr,
                       model_state=init_model_state,
                       ema_rate=config.model.ema_rate,
                       params_ema=initial_params,
                       rng=rng)  # pytype: disable=wrong-keyword-args

  # Create checkpoints directory
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  # Intermediate checkpoints to resume training after pre-emption in cloud environments
  checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta")
  tf.io.gfile.makedirs(checkpoint_dir)
  tf.io.gfile.makedirs(checkpoint_meta_dir)
  # Resume training when intermediate checkpoints are detected
  state = checkpoints.restore_checkpoint(checkpoint_meta_dir, state)
  # `state.step` is JAX integer on the GPU/TPU devices
  initial_step = int(state.step)
  rng = state.rng

  # Build data iterators
  train_ds, eval_ds, _ = datasets.get_dataset(config,
                                              additional_dim=config.training.n_jitted_steps,
                                              uniform_dequantization=config.data.uniform_dequantization)
  train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types
  eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Build one-step training and evaluation functions
  optimize_fn = losses.optimization_manager(config)
  continuous = config.training.continuous
  reduce_mean = config.training.reduce_mean
  likelihood_weighting = config.training.likelihood_weighting
  train_step_fn = losses.get_step_fn(sde, score_model, train=True, optimize_fn=optimize_fn,
                                     reduce_mean=reduce_mean, continuous=continuous,
                                     likelihood_weighting=likelihood_weighting)
  # Pmap (and jit-compile) multiple training steps together for faster running
  p_train_step = jax.pmap(functools.partial(jax.lax.scan, train_step_fn), axis_name='batch', donate_argnums=1)
  eval_step_fn = losses.get_step_fn(sde, score_model, train=False, optimize_fn=optimize_fn,
                                    reduce_mean=reduce_mean, continuous=continuous,
                                    likelihood_weighting=likelihood_weighting)
  # Pmap (and jit-compile) multiple evaluation steps together for faster running
  p_eval_step = jax.pmap(functools.partial(jax.lax.scan, eval_step_fn), axis_name='batch', donate_argnums=1)

  # Building sampling functions
  if config.training.snapshot_sampling:
    sampling_shape = (config.training.batch_size // jax.local_device_count(), config.data.image_size,
                      config.data.image_size, config.data.num_channels)
    sampling_fn = sampling.get_sampling_fn(config, sde, score_model, sampling_shape, inverse_scaler, sampling_eps)

  # Replicate the training state to run on multiple devices
  pstate = flax_utils.replicate(state)
  num_train_steps = config.training.n_iters

  # In case there are multiple hosts (e.g., TPU pods), only log to host 0
  if jax.host_id() == 0:
    logging.info("Starting training loop at step %d." % (initial_step,))
  rng = jax.random.fold_in(rng, jax.host_id())

  # JIT multiple training steps together for faster training
  n_jitted_steps = config.training.n_jitted_steps
  # Must be divisible by the number of steps jitted together
  assert config.training.log_freq % n_jitted_steps == 0 and \
         config.training.snapshot_freq_for_preemption % n_jitted_steps == 0 and \
         config.training.eval_freq % n_jitted_steps == 0 and \
         config.training.snapshot_freq % n_jitted_steps == 0, "Missing logs or checkpoints!"

  for step in range(initial_step, num_train_steps + 1, config.training.n_jitted_steps):
    # Convert data to JAX arrays and normalize them. Use ._numpy() to avoid copy.
    batch = jax.tree_map(lambda x: scaler(x._numpy()), next(train_iter))  # pylint: disable=protected-access
    rng, *next_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
    next_rng = jnp.asarray(next_rng)
    # Execute one training step
    (_, pstate), ploss = p_train_step((next_rng, pstate), batch)
    loss = flax.jax_utils.unreplicate(ploss).mean()
    # Log to console, file and tensorboard on host 0
    if jax.host_id() == 0 and step % config.training.log_freq == 0:
      logging.info("step: %d, training_loss: %.5e" % (step, loss))
      writer.scalar("training_loss", loss, step)

    # Save a temporary checkpoint to resume training after pre-emption periodically
    if step != 0 and step % config.training.snapshot_freq_for_preemption == 0 and jax.host_id() == 0:
      saved_state = flax_utils.unreplicate(pstate)
      saved_state = saved_state.replace(rng=rng)
      checkpoints.save_checkpoint(checkpoint_meta_dir, saved_state,
                                  step=step // config.training.snapshot_freq_for_preemption,
                                  keep=1)

    # Report the loss on an evaluation dataset periodically
    if step % config.training.eval_freq == 0:
      eval_batch = jax.tree_map(lambda x: scaler(x._numpy()), next(eval_iter))  # pylint: disable=protected-access
      rng, *next_rng = jax.random.split(rng, num=jax.local_device_count() + 1)
      next_rng = jnp.asarray(next_rng)
      (_, _), peval_loss = p_eval_step((next_rng, pstate), eval_batch)
      eval_loss = flax.jax_utils.unreplicate(peval_loss).mean()
      if jax.host_id() == 0:
        logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss))
        writer.scalar("eval_loss", eval_loss, step)

    # Save a checkpoint periodically and generate samples if needed
    if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
      # Save the checkpoint.
      if jax.host_id() == 0:
        saved_state = flax_utils.unreplicate(pstate)
        saved_state = saved_state.replace(rng=rng)
        checkpoints.save_checkpoint(checkpoint_dir, saved_state,
                                    step=step // config.training.snapshot_freq,
                                    keep=np.inf)

      # Generate and save samples
      if config.training.snapshot_sampling:
        rng, *sample_rng = jax.random.split(rng, jax.local_device_count() + 1)
        sample_rng = jnp.asarray(sample_rng)
        sample, n = sampling_fn(sample_rng, pstate)
        this_sample_dir = os.path.join(
          sample_dir, "iter_{}_host_{}".format(step, jax.host_id()))
        tf.io.gfile.makedirs(this_sample_dir)
        image_grid = sample.reshape((-1, *sample.shape[2:]))
        nrow = int(np.sqrt(image_grid.shape[0]))
        sample = np.clip(sample * 255, 0, 255).astype(np.uint8)
        with tf.io.gfile.GFile(
            os.path.join(this_sample_dir, "sample.np"), "wb") as fout:
          np.save(fout, sample)

        with tf.io.gfile.GFile(
            os.path.join(this_sample_dir, "sample.png"), "wb") as fout:
          utils.save_image(image_grid, fout, nrow=nrow, padding=2)


def evaluate(config,
             workdir,
             eval_folder="eval"):
  """Evaluate trained models.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints.
    eval_folder: The subfolder for storing evaluation results. Default to
      "eval".
  """
  # Create directory to eval_folder
  eval_dir = os.path.join(workdir, eval_folder, f'host_{jax.process_index()}')
  tf.io.gfile.makedirs(eval_dir)

  rng = jax.random.PRNGKey(config.seed + 1)
  rng = jax.random.fold_in(rng, jax.process_index())

  # Build data pipeline
  test_data_dir = {
    'ct2d_320': 'LIDC_320.npz',
    'ldct_512': 'LDCT.npz',
    'brats': 'BraTS.npz'
  }[config.data.dataset]
  test_data_dir = os.path.join('test_data', test_data_dir)
  test_imgs = np.load(test_data_dir)['all_imgs']
  test_imgs = test_imgs.reshape((jax.process_count(), -1, *test_imgs.shape[1:]))[jax.process_index()]

  if 'mar' in config.sampling.task:
    mar = True
    mar_data_dir = {
      'ct2d_320': 'LIDC_320_MAR.npz',
      'ldct_512': 'LDCT_MAR.npz'
    }[config.data.dataset]
    mar_data_dir = os.path.join('test_data', mar_data_dir)
    mar_data = np.load(mar_data_dir)
    ma_imgs = mar_data['ma_imgs']
    metal_imgs = mar_data['metal_masks']
    metal_imgs = metal_imgs.reshape((jax.process_count(), -1, *metal_imgs.shape[1:]))[jax.process_index()]
    ma_imgs = ma_imgs.reshape((jax.process_count(), -1, *ma_imgs.shape[1:]))[jax.process_index()]
    gt_imgs = mar_data['gt_imgs']
    gt_imgs = gt_imgs.reshape((jax.process_count(), -1, *gt_imgs.shape[1:]))[jax.process_index()]
  else:
    mar = False

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Initialize model
  rng, model_rng = jax.random.split(rng)
  score_model, init_model_state, initial_params = mutils.init_model(model_rng, config)
  optimizer = losses.get_optimizer(config).create(initial_params)
  state = mutils.State(step=0, optimizer=optimizer, lr=config.optim.lr,
                       model_state=init_model_state,
                       ema_rate=config.model.ema_rate,
                       params_ema=initial_params,
                       rng=rng)  # pytype: disable=wrong-keyword-args

  checkpoint_dir = os.path.join(workdir, "checkpoints")

  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  sampling_shape = (config.eval.batch_size // jax.device_count(),
                    config.data.image_size, config.data.image_size,
                    config.data.num_channels)

  cs_solver = cs.get_cs_solver(config, sde, score_model, sampling_shape, inverse_scaler, eps=sampling_eps)
  state = checkpoints.restore_checkpoint(checkpoint_dir, state, step=config.eval.ckpt_id)
  pstate = flax.jax_utils.replicate(state)

  hyper_params = {
    'projection': [config.sampling.coeff, config.sampling.snr],
    'langevin_projection': [config.sampling.coeff, config.sampling.snr],
    'langevin': [config.sampling.projection_sigma_rate, config.sampling.snr],
    'baseline': [config.sampling.projection_sigma_rate, config.sampling.snr]
  }[config.sampling.cs_solver]

  per_host_batch_size = config.eval.batch_size // jax.host_count()
  num_batches = int(np.ceil(len(test_imgs) / per_host_batch_size))

  # Create a circular mask
  img_size = config.data.image_size
  mask = Image.new('L', (img_size, img_size), 0)
  draw = ImageDraw.Draw(mask)
  draw.pieslice([0, 0, img_size, img_size], 0, 360, fill=255)
  toTensor = transforms.ToTensor()
  mask = toTensor(mask)[0]

  def get_metric(predictions, targets, mask_roi=False, hist_norm=False):
    with torch.no_grad():
      if hist_norm:
        pred_hist = torch.histc(predictions, bins=255)
        targ_hist = torch.histc(targets, bins=255)

        peak_pred1 = torch.argmax(pred_hist[:75]) / 255.
        peak_pred2 = (torch.argmax(pred_hist[75:]) + 75) / 255.
        peak_targ1 = torch.argmax(targ_hist[:75]) / 255.
        peak_targ2 = (torch.argmax(targ_hist[75:]) + 75) / 255.

        predictions = torch.clamp((predictions - peak_pred1) / (peak_pred2 - peak_pred1), min=0)
        targets = torch.clamp((targets - peak_targ1) / (peak_targ2 - peak_targ1), min=0)

        predictions = torch.clamp(predictions, max=torch.max(targets).item(), min=0)
        predictions /= torch.max(targets)
        targets /= torch.max(targets)

      # Mask Region of Interest
      if mask_roi:
        predictions = predictions * mask
        targets = targets * mask

      return (piq.psnr(predictions[None, None, ...], targets[None, None, ...], data_range=1.).item(),
              piq.ssim(predictions[None, None, ...], targets[None, None, ...], data_range=1.).item())

  def compute_mar_metrics(gt, pred, metal_mask):
    gt[metal_mask] = 0.
    pred[metal_mask] = 0.
    gt = np.clip(gt, 0., 1.)
    pred = np.clip(pred, 0., 1.)
    ssim = structural_similarity(gt, pred)
    psnr = peak_signal_noise_ratio(gt, pred)
    rmse = np.sqrt(np.mean((gt - pred) ** 2))
    return ssim, psnr, rmse

  all_samples = []
  all_ssims = []
  all_psnrs = []
  all_ssims_mask = []
  all_psnrs_mask = []
  all_ssims_mask_hist = []
  all_psnrs_mask_hist = []
  all_mar_ssims = []
  all_mar_psnrs = []
  all_mar_rmses = []

  for batch in tqdm.tqdm(range(num_batches)):
    if not mar:
      current_batch = jnp.asarray(test_imgs[batch * per_host_batch_size:
                                            min((batch + 1) * per_host_batch_size,
                                                len(test_imgs))], dtype=jnp.float32) / 255.
    else:
      current_batch = jnp.asarray(ma_imgs[batch * per_host_batch_size:
                                          min((batch + 1) * per_host_batch_size,
                                              len(ma_imgs))], dtype=jnp.float32)
      test_batch = jnp.asarray(gt_imgs[batch * per_host_batch_size:
                                         min((batch + 1) * per_host_batch_size,
                                             len(gt_imgs))], dtype=jnp.float32)
      metal_batch = jnp.asarray(metal_imgs[batch * per_host_batch_size:
                                           min((batch + 1) * per_host_batch_size,
                                               len(metal_imgs))], dtype=jnp.bool_)
    n_effective_samples = len(current_batch)
    if n_effective_samples < per_host_batch_size:
      pad_len = per_host_batch_size - len(current_batch)
      current_batch = jnp.pad(current_batch, ((0, pad_len), (0, 0), (0, 0)),
                              mode='constant', constant_values=0.)

    current_batch = current_batch.reshape((-1, *sampling_shape))
    img = scaler(current_batch)

    rng, *step_rng = jax.random.split(rng, jax.local_device_count() + 1)
    step_rng = jnp.asarray(step_rng)
    samples = cs_solver(step_rng, pstate, img, *hyper_params)

    samples = np.clip(np.asarray(samples), 0., 1.)
    samples = samples.reshape((-1, config.data.image_size, config.data.image_size, 1))[:n_effective_samples]
    all_samples.extend(samples)

    if not mar:
      ground_truth = np.asarray(inverse_scaler(img)).reshape((-1, config.data.image_size,
                                                              config.data.image_size, 1))
      ground_truth = np.clip(ground_truth, 0., 1.)
      ground_truth = torch.from_numpy(ground_truth).permute(0, 3, 1, 2)
      samples = torch.from_numpy(samples).permute(0, 3, 1, 2)

      for i in range(n_effective_samples):
        p, s = get_metric(samples[i].squeeze(), ground_truth[i].squeeze())
        all_psnrs.append(p)
        all_ssims.append(s)

        p, s = get_metric(samples[i].squeeze(), ground_truth[i].squeeze(), mask_roi=True)
        all_psnrs_mask.append(p)
        all_ssims_mask.append(s)

        p, s = get_metric(samples[i].squeeze(), ground_truth[i].squeeze(), mask_roi=True, hist_norm=True)
        all_psnrs_mask_hist.append(p)
        all_ssims_mask_hist.append(s)

      print(f'PSNR: {np.asarray(all_psnrs).mean():.4f}, SSIM: {np.asarray(all_ssims).mean():.4f}')
      print(f'with mask: PSNR: {np.asarray(all_psnrs_mask).mean():.4f}, SSIM: {np.asarray(all_ssims_mask).mean():.4f}')
      print(
        f'with mask & hist: PSNR: {np.asarray(all_psnrs_mask_hist).mean():.4f}, SSIM: {np.asarray(all_ssims_mask_hist).mean():.4f}')
    else:
      ground_truth = np.array(test_batch)
      samples = np.array(samples)[..., 0]
      masks = np.array(metal_batch) > 0.

      for i in range(n_effective_samples):
        ssim, psnr, rmse = compute_mar_metrics(ground_truth[i], samples[i], masks[i])
        all_mar_ssims.append(ssim)
        all_mar_psnrs.append(psnr)
        all_mar_rmses.append(rmse)

      print(f'SSIM: {np.asarray(all_mar_ssims).mean():.4f}, PSNR: {np.asarray(all_mar_psnrs).mean():.4f}, '
            f'RMSE: {np.asarray(all_mar_rmses).mean():.4f}')

  all_samples = (np.stack(all_samples, axis=0) * 255.).astype(np.uint8)
  np.savez_compressed(os.path.join(eval_dir, "reconstructions.npz"), recon=all_samples)
  if not mar:
    all_psnrs = np.asarray(all_psnrs)
    all_ssims = np.asarray(all_ssims)
    all_psnrs_mask = np.asarray(all_psnrs_mask)
    all_ssims_mask = np.asarray(all_ssims_mask)
    all_psnrs_mask_hist = np.asarray(all_psnrs_mask_hist)
    all_ssims_mask_hist = np.asarray(all_ssims_mask_hist)

    np.savez_compressed(os.path.join(eval_dir, "metrics.npz"),
                        psnrs=all_psnrs,
                        ssims=all_ssims,
                        psnrs_mask=all_psnrs_mask,
                        ssims_mask=all_ssims_mask,
                        psnrs_mask_hist=all_psnrs_mask_hist,
                        ssims_mask_hist=all_ssims_mask_hist)

  else:
    all_psnrs = np.asarray(all_mar_psnrs)
    all_ssims = np.asarray(all_mar_ssims)
    all_rmses = np.asarray(all_mar_rmses)

    np.savez_compressed(os.path.join(eval_dir, "metrics.npz"),
                        psnrs=all_psnrs,
                        ssims=all_ssims,
                        rmses=all_rmses)


def hyperparam_search(config, workdir, eval_folder="eval"):
  """Evaluate trained models.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints.
  """
  # Create directory to eval_folder
  from ax.service.ax_client import AxClient

  eval_dir = os.path.join(workdir, eval_folder)
  tf.io.gfile.makedirs(eval_dir)

  rng = jax.random.PRNGKey(config.seed + 2)

  # Build data pipeline
  train_ds, eval_ds, _ = datasets.get_dataset(config,
                                              additional_dim=None,
                                              uniform_dequantization=False,
                                              evaluation=True)

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Initialize model
  rng, model_rng = jax.random.split(rng)
  score_model, init_model_state, initial_params = mutils.init_model(model_rng, config)
  optimizer = losses.get_optimizer(config).create(initial_params)
  state = mutils.State(step=0, optimizer=optimizer, lr=config.optim.lr,
                       model_state=init_model_state,
                       ema_rate=config.model.ema_rate,
                       params_ema=initial_params,
                       rng=rng)  # pytype: disable=wrong-keyword-args

  checkpoint_dir = os.path.join(workdir, "checkpoints")

  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  sampling_shape = (config.eval.batch_size // jax.local_device_count(),
                    config.data.image_size, config.data.image_size,
                    config.data.num_channels)

  cs_solver = cs.get_cs_solver(config, sde, score_model, sampling_shape, inverse_scaler, eps=sampling_eps)
  save_path = os.path.join(eval_dir, 'ax_client_state.json')
  backup_path = os.path.join(eval_dir, 'ax_client_state_backup.json')

  if tf.io.gfile.exists(save_path):
    # Recover from preemption
    ax_client = AxClient.load_from_json_file(filepath=save_path)
  else:
    ax_client = AxClient()
    if config.sampling.cs_solver.lower() in ('langevin', 'baseline'):
      if config.sampling.task == 'mri':
        snr_bounds = [0.1, 0.8]
      else:
        snr_bounds = [0.00, 0.40]
      ax_client.create_experiment(
        name="langevin",
        parameters=[
          {
            "name": "snr",
            "type": "range",
            "bounds": snr_bounds,
            "value_type": "float",  # Optional, defaults to inference from type of "bounds".
            "log_scale": False,  # Optional, defaults to False.
          },
          {
            "name": "projection_sigma_rate",
            "type": "range",
            "bounds": [0.1, 20.],
          },
        ],
        objective_name="ssim",
        minimize=False,  # Optional, defaults to False.
      )

    elif config.sampling.cs_solver.lower() in ('projection', 'langevin_projection'):
      if config.sampling.task == 'mri':
        snr_bounds = [0.2, 0.8]
      else:
        snr_bounds = [0.00, 0.40]
      ax_client.create_experiment(
        name="projection",
        parameters=[
          {
            "name": "snr",
            "type": "range",
            "bounds": snr_bounds,
            "value_type": "float",  # Optional, defaults to inference from type of "bounds".
            "log_scale": False,  # Optional, defaults to False.
          },
          {
            "name": "coeff",
            "type": "range",
            "bounds": [0., 1.],
            "value_type": "float",
            "log_scale": False
          },
        ],
        objective_name="ssim",
        minimize=False,  # Optional, defaults to False.
      )

    elif config.sampling.cs_solver.lower() in ['central_slice_inpainting', 'override_inpainting']:
      ax_client.create_experiment(
        name=config.sampling.cs_solver.lower(),
        parameters=[
          {
            "name": "snr",
            "type": "range",
            "bounds": [0.00, 0.4] if config.sampling.task == 'ct' else [0.2, 0.8],
            "value_type": "float",  # Optional, defaults to inference from type of "bounds".
            "log_scale": False,  # Optional, defaults to False.
          },
        ],
        objective_name="ssim",
        minimize=False,  # Optional, defaults to False.
      )
    else:
      raise ValueError(f'Solver name {config.sampling.cs_solver} not recognized.')

  if 'mar' in config.sampling.task:
    mar_data_dir = {
      'ct2d_320': 'LIDC_320_MAR_val.npz',
      'ldct_512': 'LDCT_MAR_val.npz'
    }[config.data.dataset]
    mar_data_dir = os.path.join('test_data', mar_data_dir)
    mar_data = np.load(mar_data_dir)
    ma_imgs = mar_data['ma_imgs']
    gt_imgs = mar_data['gt_imgs']
    metal_imgs = mar_data['metal_masks']

  def compute_mar_metrics(gt, pred, metal_mask):
    gt[metal_mask] = 0.
    pred[metal_mask] = 0.
    gt = np.clip(gt, 0., 1.)
    pred = np.clip(pred, 0., 1.)
    ssim = structural_similarity(gt, pred)
    psnr = peak_signal_noise_ratio(gt, pred)
    rmse = np.sqrt(np.mean((gt - pred) ** 2))
    return ssim, psnr, rmse

  mar = 'mar' in config.sampling.task.lower()
  def get_metric_fn(state, eval_iter, rng):
    data_iter = eval_iter
    rng = rng
    pstate = flax.jax_utils.replicate(state)

    def metric_fn(hyper_params):
      nonlocal rng
      if not mar:
        batch = jnp.asarray(next(data_iter)['image'].numpy())
        img = scaler(batch)
      else:
        len_img = len(ma_imgs)
        random_idx = np.random.choice(len_img, config.eval.batch_size)
        img = scaler(ma_imgs[random_idx].reshape((-1, *sampling_shape)))

      rng, *step_rng = jax.random.split(rng, jax.local_device_count() + 1)
      step_rng = jnp.asarray(step_rng)

      if config.sampling.cs_solver.lower() in ('langevin', 'baseline'):
        samples = cs_solver(step_rng, pstate, img, hyper_params['projection_sigma_rate'], hyper_params['snr'])
      elif config.sampling.cs_solver.lower() in ('projection', 'langevin_projection'):
        samples = cs_solver(step_rng, pstate, img, hyper_params['coeff'], hyper_params['snr'])
      elif config.sampling.cs_solver.lower() in ['central_slice_inpainting', 'override_inpainting']:
        samples = cs_solver(step_rng, pstate, img, hyper_params['snr'])

      if not mar:
        ground_truth = np.asarray(inverse_scaler(img)).reshape((-1, config.data.image_size,
                                                                config.data.image_size, 1))
        ground_truth = np.clip(ground_truth, 0., 1.)
        samples = np.clip(np.asarray(samples).reshape(ground_truth.shape), 0., 1.)
        ssims = []
        for i in range(len(ground_truth)):
          ssims.append(structural_similarity(ground_truth[i].squeeze(), samples[i].squeeze(), data_range=1.))
      else:
        ground_truth = gt_imgs[random_idx]
        metal_mask = metal_imgs[random_idx] > 0
        samples = np.array(samples).reshape(ground_truth.shape)
        ssims = []
        for i in range(len(random_idx)):
          ssim, _, _ = compute_mar_metrics(ground_truth[i], samples[i], metal_mask[i])
          ssims.append(ssim)

      ssims = np.asarray(ssims)
      value = ssims.mean()
      if np.isnan(value):
        return 0
      else:
        return value

    return metric_fn

  state = checkpoints.restore_checkpoint(checkpoint_dir, state, step=config.eval.ckpt_id)

  metric_fn = get_metric_fn(state, iter(train_ds), rng)
  for i in range(100):
    parameters, trial_idx = ax_client.get_next_trial()
    ssim = metric_fn(parameters)
    print(f'trial_idx: {trial_idx}, ssim: {ssim}, params: {parameters}')
    ax_client.complete_trial(trial_index=trial_idx, raw_data=ssim)
    try:
      shutil.copyfile(save_path, backup_path)
    except:
      pass
    ax_client.save_to_json_file(save_path)
