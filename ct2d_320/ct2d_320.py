"""ct2d_320 dataset."""

import tensorflow_datasets as tfds
import numpy as np
import os

_DESCRIPTION = """
2D CT dataset with resolution 320px
"""


class Ct2d320(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for ct2d 320 dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(ct2d_320): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=(320, 320, 1)),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=None,  # Set to `None` to disable
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(ct2d_320): Downloads the data and defines the splits
    path = os.path.join('/raid/song', 'data', 'image_bin_320')

    return {
        'train': self._generate_examples(path),
    }

  def slice_3d_img(self, img, view_plane='ax'):
    # normalization
    img = img - np.min(img)
    assert(np.min(img) == 0.)
    factor = 1. / np.max(img)
    img = img * factor
    imgs = []
    if view_plane == 'cor':
        for idx in range(img.shape[0]):
            imgs.append(img[idx,:,:])
    elif view_plane == 'sag':
        for idx in range(img.shape[1]):
            imgs.append(img[:,idx,:])
    elif view_plane == 'ap':
        for idx in range(img.shape[2]):
            imgs.append(img[:,:,idx])
    return imgs

  def _generate_examples(self, path):
    """Yields examples."""
    all_files = os.listdir(path)
    count = -1
    for file in all_files:
      file_path = os.path.join(path, file)
      img = np.fromfile(file_path, dtype='float32')
      img = np.reshape(img, (160, 320, 320))
      img_list = self.slice_3d_img(img, 'cor')
      for img in img_list:
        count += 1
        yield count, {
          'image': (np.array(img)[..., None] * 255).astype(np.uint8)
        }
