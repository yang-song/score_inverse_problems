"""brats dataset."""

import tensorflow_datasets as tfds
import os
import SimpleITK as sitk
import numpy as np

_DESCRIPTION = """
BraTS 2021
"""

_CITATION = """
[1] U.Baid, et al., The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification, arXiv:2107.02314, 2021.

[2] B. H. Menze, A. Jakab, S. Bauer, J. Kalpathy-Cramer, K. Farahani, J. Kirby, et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)", IEEE Transactions on Medical Imaging 34(10), 1993-2024 (2015) DOI: 10.1109/TMI.2014.2377694

[3] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J.S. Kirby, et al., "Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features", Nature Scientific Data, 4:170117 (2017) DOI: 10.1038/sdata.2017.117

[4] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J. Kirby, et al., "Segmentation Labels and Radiomic Features for the Pre-operative Scans of the TCGA-GBM collection", The Cancer Imaging Archive, 2017. DOI: 10.7937/K9/TCIA.2017.KLXWJJ1Q

[5] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J. Kirby, et al., "Segmentation Labels and Radiomic Features for the Pre-operative Scans of the TCGA-LGG collection", The Cancer Imaging Archive, 2017. DOI: 10.7937/K9/TCIA.2017.GJQ7R0EF
"""


class Brats(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for brats dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
    '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
      builder=self,
      description=_DESCRIPTION,
      features=tfds.features.FeaturesDict({
        # These are the features of your dataset like images, labels ...
        'image': tfds.features.Image(shape=(240, 240, 1)),
        'label': tfds.features.ClassLabel(names=['t1', 't1ce', 't2', 'flair']),
      }),
      # If there's a common (input, target) tuple from the
      # features, specify them here. They'll be used if
      # `as_supervised=True` in `builder.as_dataset`.
      supervised_keys=('image', 'label'),  # Set to `None` to disable
      citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    train_path = '/raid/song/BraTS/training'
    return {
      'train': self._generate_examples(train_path),
    }

  def read_img(self, path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))

  def get_bound(self, data, return_coord=False):
    """
    get the boundary of image z y x
    data is padded with 0
    """
    data_0 = data - data.min()
    # display
    #     display_arr_stats(data_0)

    z, y, x = np.where(data_0)
    z_start, z_end = np.min(z), np.max(z)
    y_start, y_end = np.min(y), np.max(y)
    x_start, x_end = np.min(x), np.max(x)

    indicator = np.ones_like(data, dtype=bool)
    indicator[z_start:z_end, y_start:y_end, x_start:x_end] = False
    if return_coord:
      return z_start, z_end, y_start, y_end, x_start, x_end, indicator
    return indicator

  def mri_data_norm(self, data, scale=6.0, return_v=False):
    # important to transfer datatype to keep division works
    data = data.astype(float)

    # get a box mask to remove background
    min_z, max_z, min_y, max_y, min_x, max_x, indicator = self.get_bound(data, return_coord=True)
    crop_data = np.array(data[min_z:max_z, min_y:max_y, min_x:max_x] * 1.0)
    mean, std = np.mean(crop_data), np.std(crop_data)
    # clip outliers
    crop_data = np.clip(crop_data, max(mean - scale * std, crop_data.min()), min(mean + scale * std, crop_data.max()))

    # normalize scale [0,1]
    min_v = crop_data.min()
    crop_data = np.array(crop_data - min_v)
    max_v = crop_data.max() * 1.0
    crop_data = np.array(crop_data) / max_v

    data[min_z:max_z, min_y:max_y, min_x:max_x] = np.array(crop_data)
    data[indicator] = 0

    if return_v:
      return np.array(data), [min_v, max_v, np.float(min_y), np.float(max_y), np.float(min_x), np.float(max_x)]
    else:
      return np.array(data)

  def _generate_examples(self, path):
    """Yields examples."""
    img_list = os.listdir(path)
    domains = ['t1', 't1ce', 't2', 'flair']
    count = -1
    for img_folder in img_list:
      img_path = os.path.join(path, img_folder, os.path.split(img_folder)[-1] + '_t1.nii.gz')
      img_array = self.read_img(img_path)
      z, x, y = np.where(img_array)
      z_min, z_max = np.min(z), np.max(z)
      z_min = z_min + 40
      z_max = z_max - 25

      for domain in domains:
        img_path = os.path.join(path, img_folder, os.path.split(img_folder)[-1] + f'_{domain}.nii.gz')
        img_array = self.read_img(img_path)
        img_array = self.mri_data_norm(img_array, scale=6.0)
        for z_idx in range(z_min, z_max + 1):
          img = img_array[z_idx, ...]
          # Sanity check intensity values
          assert np.min(img) >= 0.0 and np.max(img) <= 1.0 and np.max(img) >= 0.1
          count += 1
          yield count, {
            'image': np.clip(img[..., None] * 255., 0.0, 255.).astype(np.uint8),
            'label': domain
          }