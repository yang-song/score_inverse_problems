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

# Lint as: python3
"""Training NCSNv3 on CelebAHQ with continuous sigmas."""

from configs.default_cs_configs import get_default_configs


def get_config():
  config = get_default_configs()
  # training
  training = config.training
  training.batch_size = 16
  training.n_iters = 2400001
  training.snapshot_freq = 20000
  training.snapshot_freq_for_preemption = 5000
  training.snapshot_sampling = True
  training.sde = 'vesde'
  training.continuous = True

  # eval
  evaluate = config.eval
  evaluate.batch_size = 36
  evaluate.num_samples = 50000
  evaluate.ckpt_id = 63

  # sampling
  sampling = config.sampling
  sampling.iradon_K = 2.2
  sampling.snr = 0.4
  # sampling.coeff = 0.802
  sampling.coeff = 0.72
  sampling.expansion = 4

  # data
  data = config.data
  data.dataset = 'ldct_512'
  data.image_size = 512
  data.num_channels = 1
  data.centered = False
  data.random_flip = False
  data.uniform_dequantization = False

  # model
  model = config.model
  model.name = 'ncsnpp'
  model.scale_by_sigma = True
  model.sigma_max = 220.
  model.num_scales = 1000
  model.ema_rate = 0.9999
  model.sigma_min = 0.01
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 16
  model.ch_mult = (1, 2, 4, 8, 16, 32, 32)
  model.num_res_blocks = 1
  model.attn_resolutions = (16,)
  model.dropout = 0.
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = True
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'output_skip'
  model.progressive_input = 'input_skip'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.init_scale = 0.
  model.fourier_scale = 16
  model.conv_size = 3

  # optim
  optim = config.optim
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.amsgrad = False
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.

  config.seed = 42

  return config
