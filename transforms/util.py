import jax.numpy as jnp
import numpy as np

__all__ = ['prod', 'resize']


def _normalize_axes(axes, ndim):
  if axes is None:
    return tuple(range(ndim))
  else:
    return tuple(a % ndim for a in sorted(axes))


def _normalize_shape(shape):
  if isinstance(shape, int):
    return (shape,)
  else:
    return tuple(shape)


def _expand_shapes(*shapes):
  shapes = [list(shape) for shape in shapes]
  max_ndim = max(len(shape) for shape in shapes)
  shapes_exp = [[1] * (max_ndim - len(shape)) + shape
                for shape in shapes]

  return tuple(shapes_exp)


def prod(shape):
  """Computes product of shape.
  Args:
      shape (tuple or list): shape.
  Returns:
      Product.
  """
  return np.prod(shape, dtype=np.int32)


def resize(input, oshape, ishift=None, oshift=None):
  """Resize with zero-padding or cropping.
  Args:
      input (array): Input array.
      oshape (tuple of ints): Output shape.
      ishift (None or tuple of ints): Input shift.
      oshift (None or tuple of ints): Output shift.
  Returns:
      array: Zero-padded or cropped result.
  """

  ishape1, oshape1 = _expand_shapes(input.shape, oshape)

  if ishape1 == oshape1:
    return input.reshape(oshape)

  if ishift is None:
    ishift = [max(i // 2 - o // 2, 0) for i, o in zip(ishape1, oshape1)]

  if oshift is None:
    oshift = [max(o // 2 - i // 2, 0) for i, o in zip(ishape1, oshape1)]

  copy_shape = [min(i - si, o - so)
                for i, si, o, so in zip(ishape1, ishift, oshape1, oshift)]
  islice = tuple([slice(si, si + c) for si, c in zip(ishift, copy_shape)])
  oslice = tuple([slice(so, so + c) for so, c in zip(oshift, copy_shape)])

  output = jnp.zeros(oshape1, dtype=input.dtype)
  input = input.reshape(ishape1)
  output = output.at[oslice].set(input[islice])

  return output.reshape(oshape)
