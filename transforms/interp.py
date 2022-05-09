# -*- coding: utf-8 -*-
"""Interpolation functions.
"""
import numpy as np
import jax.numpy as jnp
import jax
import functools

from transforms import util

__all__ = ['interpolate']

KERNELS = ['spline', 'kaiser_bessel']


def interpolate(input, coord, kernel='spline', width=2, param=1):
  r"""Interpolation from array to points specified by coordinates.

  Let :math:`x` be the input, :math:`y` be the output,
  :math:`c` be the coordinates, :math:`W` be the kernel width,
  and :math:`K` be the interpolation kernel, then the function computes,

  .. math ::
      y[j] = \sum_{i : \| i - c[j] \|_\infty \leq W / 2}
             K\left(\frac{i - c[j]}{W / 2}\right) x[i]

  There are two types of kernels: 'spline' and 'kaiser_bessel'.

  'spline' uses the cardinal B-spline functions as kernels.
  The order of the spline can be specified using param.
  For example, param=1 performs linear interpolation.
  Concretely, for param=0, :math:`K(x) = 1`,
  for param=1, :math:`K(x) = 1 - |x|`, and
  for param=2, :math:`K(x) = \frac{9}{8} (1 - |x|)^2`
  for :math:`|x| > \frac{1}{3}`
  and :math:`K(x) = \frac{3}{4} (1 - 3 x^2)` for :math:`|x| < \frac{1}{3}`.

  These function expressions are derived from the reference wikipedia
  page by shifting and scaling the range to -1 to 1.
  When the coordinates specifies a uniformly spaced grid,
  it is recommended to use the original scaling with width=param + 1
  so that the interpolation weights add up to one.

  'kaiser_bessel' uses the Kaiser-Bessel function as kernel.
  Concretely, :math:`K(x) = I_0(\beta \sqrt{1 - x^2})`,
  where :math:`I_0` is the modified Bessel function of the first kind.
  The beta parameter can be specified with param.
  The modified Bessel function of the first kind is approximated
  using the power series, following the reference.

  Args:
      input (array): Input array of shape.
      coord (array): Coordinate array of shape [..., ndim]
      width (float or tuple of floats): Interpolation kernel full-width.
      kernel (str): Interpolation kernel, {'spline', 'kaiser_bessel'}.
      param (float or tuple of floats): Kernel parameter.

  Returns:
      output (array): Output array.

  References:
      https://en.wikipedia.org/wiki/Spline_wavelet#Cardinal_B-splines_of_small_orders
      http://people.math.sfu.ca/~cbm/aands/page_378.htm
  """
  ndim = coord.shape[-1]

  batch_shape = input.shape[:-ndim]
  batch_size = util.prod(batch_shape)

  pts_shape = coord.shape[:-1]
  npts = util.prod(pts_shape)

  input = input.reshape([batch_size] + list(input.shape[-ndim:]))
  coord = coord.reshape([npts, ndim])

  if np.isscalar(param):
    param = jnp.array([param] * ndim, coord.dtype)
  else:
    param = jnp.array(param, coord.dtype)

  if np.isscalar(width):
    width = np.array([width] * ndim, coord.dtype)
  else:
    width = np.array(width, coord.dtype)

  output = _interpolate[kernel][ndim - 1](input, coord, width, param)
  return output.reshape(batch_shape + pts_shape)


def _spline_kernel(x, order):
  if order == 0:
    return jnp.where(jnp.abs(x) > 1., 0., 1.)
  elif order == 1:
    return jnp.where(jnp.abs(x) > 1., 0., 1. - jnp.abs(x))
  elif order == 2:
    return jnp.where(jnp.abs(x) > 1., 0.,
                     jnp.where(jnp.abs(x) > 1 / 3,
                               9 / 8 * (1 - jnp.abs(x)) ** 2,
                               3 / 4 * (1 - 3 * x ** 2)))


def _kaiser_bessel_kernel(x, beta):
  xx = beta * (1. - x ** 2) ** 0.5
  t = xx / 3.75
  return jnp.where(jnp.abs(x) > 1., 0.,
                   jnp.where(xx < 3.75,
                             1 + 3.5156229 * t ** 2 + 3.0899424 * t ** 4 +
                             1.2067492 * t ** 6 + 0.2659732 * t ** 8 +
                             0.0360768 * t ** 10 + 0.0045813 * t ** 12,
                             xx ** -0.5 * jnp.exp(xx) * (
                                 0.39894228 + 0.01328592 * t ** -1 +
                                 0.00225319 * t ** -2 - 0.00157565 * t ** -3 +
                                 0.00916281 * t ** -4 - 0.02057706 * t ** -5 +
                                 0.02635537 * t ** -6 - 0.01647633 * t ** -7 +
                                 0.00392377 * t ** -8)
                             ))


def _get_interpolate(kernel):
  if kernel == 'spline':
    kernel = _spline_kernel
  elif kernel == 'kaiser_bessel':
    kernel = _kaiser_bessel_kernel

  def _interpolate1(input, coord, width, param):
    kx = coord[:, -1]
    x0 = jnp.ceil(kx - width[-1] / 2).astype(jnp.int32)
    x_range = x0[:, None] + jnp.arange(0, width[-1], dtype=jnp.int32)[None, :]
    w = kernel((x_range - kx[:, None]) / (width[-1] / 2), param[-1])
    input = jnp.take(input, x_range, axis=1, mode='wrap')
    output = jnp.sum(w * input, axis=2)

    return output

  def _interpolate2(input, coord, width, param):

    batch_size, ny, nx = input.shape

    kx = coord[:, -1]
    ky = coord[:, -2]
    x0 = jnp.ceil(kx - width[-1] / 2).astype(jnp.int32)
    y0 = jnp.ceil(ky - width[-2] / 2).astype(jnp.int32)
    arange_x = jnp.arange(0, width[-1], dtype=jnp.int32)
    arange_y = jnp.arange(0, width[-2], dtype=jnp.int32)
    x_range = x0[:, None] + arange_x[None, :]
    y_range = y0[:, None] + arange_y[None, :]

    wy = kernel((y_range - ky[:, None]) / (width[-2] / 2), param[-2])
    wx = kernel((x_range - kx[:, None]) / (width[-1] / 2), param[-1])
    w = wy[:, :, None] * wx[:, None, :]

    x_mesh, y_mesh = jnp.meshgrid(arange_x, arange_y, indexing='xy')
    x_mesh_range = (x0[:, None, None] + x_mesh) % nx
    y_mesh_range = (y0[:, None, None] + y_mesh) % ny

    input = input[:, y_mesh_range, x_mesh_range]

    return jnp.sum(w * input, axis=(2, 3))

  def _interpolate3(input, coord, width, param):
    batch_size, nz, ny, nx = input.shape

    kx = coord[:, -1]
    ky = coord[:, -2]
    kz = coord[:, -3]

    x0 = jnp.ceil(kx - width[-1] / 2).astype(jnp.int32)
    y0 = jnp.ceil(ky - width[-2] / 2).astype(jnp.int32)
    z0 = jnp.ceil(kz - width[-3] / 2).astype(jnp.int32)

    arange_x = jnp.arange(0, width[-1], dtype=jnp.int32)
    arange_y = jnp.arange(0, width[-2], dtype=jnp.int32)
    arange_z = jnp.arange(0, width[-3], dtype=jnp.int32)

    x_range = x0[:, None] + arange_x[None, :]
    y_range = y0[:, None] + arange_y[None, :]
    z_range = z0[:, None] + arange_z[None, :]

    wz = kernel((z_range - kz[:, None]) / (width[-3] / 2), param[-3])
    wy = kernel((y_range - ky[:, None]) / (width[-2] / 2), param[-2])
    wx = kernel((x_range - kx[:, None]) / (width[-1] / 2), param[-1])
    w = wz[:, :, None, None] * wy[:, None, :, None] * wx[:, None, None, :]

    z_mesh, y_mesh, x_mesh = jnp.meshgrid(arange_z, arange_y, arange_x, indexing='ij')
    x_mesh_range = (x0[:, None, None, None] + x_mesh) % nx
    y_mesh_range = (y0[:, None, None, None] + y_mesh) % ny
    z_mesh_range = (z0[:, None, None, None] + z_mesh) % nz

    input = input[:, z_mesh_range, y_mesh_range, x_mesh_range]

    return jnp.sum(w * input, axis=(2, 3, 4))

  return _interpolate1, _interpolate2, _interpolate3


def gridding(input, coord, shape, kernel="spline", width=2, param=1):
  r"""Gridding of points specified by coordinates to array.

  Let :math:`y` be the input, :math:`x` be the output,
  :math:`c` be the coordinates, :math:`W` be the kernel width,
  and :math:`K` be the interpolation kernel, then the function computes,

  .. math ::
      x[i] = \sum_{j : \| i - c[j] \|_\infty \leq W / 2}
             K\left(\frac{i - c[j]}{W / 2}\right) y[j]

  There are two types of kernels: 'spline' and 'kaiser_bessel'.

  'spline' uses the cardinal B-spline functions as kernels.
  The order of the spline can be specified using param.
  For example, param=1 performs linear interpolation.
  Concretely, for param=0, :math:`K(x) = 1`,
  for param=1, :math:`K(x) = 1 - |x|`, and
  for param=2, :math:`K(x) = \frac{9}{8} (1 - |x|)^2`
  for :math:`|x| > \frac{1}{3}`
  and :math:`K(x) = \frac{3}{4} (1 - 3 x^2)` for :math:`|x| < \frac{1}{3}`.

  These function expressions are derived from the reference wikipedia
  page by shifting and scaling the range to -1 to 1.
  When the coordinates specifies a uniformly spaced grid,
  it is recommended to use the original scaling with width=param + 1
  so that the interpolation weights add up to one.

  'kaiser_bessel' uses the Kaiser-Bessel function as kernel.
  Concretely, :math:`K(x) = I_0(\beta \sqrt{1 - x^2})`,
  where :math:`I_0` is the modified Bessel function of the first kind.
  The beta parameter can be specified with param.
  The modified Bessel function of the first kind is approximated
  using the power series, following the reference.

  Args:
      input (array): Input array.
      coord (array): Coordinate array of shape [..., ndim]
      width (float or tuple of floats): Interpolation kernel full-width.
      kernel (str): Interpolation kernel, {"spline", "kaiser_bessel"}.
      param (float or tuple of floats): Kernel parameter.

  Returns:
      output (array): Output array.

  References:
      https://en.wikipedia.org/wiki/Spline_wavelet#Cardinal_B-splines_of_small_orders
      http://people.math.sfu.ca/~cbm/aands/page_378.htm
  """
  ndim = coord.shape[-1]

  batch_shape = shape[:-ndim]
  batch_size = util.prod(batch_shape)

  pts_shape = coord.shape[:-1]
  npts = util.prod(pts_shape)

  input = input.reshape([batch_size, npts])
  coord = coord.reshape([npts, ndim])
  output = jnp.zeros([batch_size] + list(shape[-ndim:]), dtype=input.dtype)

  if np.isscalar(param):
    param = np.array([param] * ndim, coord.dtype)
  else:
    param = np.array(param, coord.dtype)

  if np.isscalar(width):
    width = np.array([width] * ndim, coord.dtype)
  else:
    width = np.array(width, coord.dtype)

  output = _gridding[kernel][ndim - 1](output, input, coord, width, param)

  return output.reshape(shape)


def _get_gridding(kernel):
  if kernel == 'spline':
    kernel = _spline_kernel
  elif kernel == 'kaiser_bessel':
    kernel = _kaiser_bessel_kernel

  interpolate1, interpolate2, interpolate3 = _get_interpolate(kernel)

  # implementing transposed convolutions with gradients
  def _gridding1(output, input, coord, width, param):
    def helper(output, input):
      value = interpolate1(output, coord, width, param)
      value = jnp.sum(value * input)
      return value

    grad_fn = jax.grad(helper, argnums=0)
    return grad_fn(output.real, input.real) + 1j * grad_fn(output.imag, input.imag)

  def _gridding2(output, input, coord, width, param):
    def helper(output, input):
      value = interpolate2(output, coord, width, param)
      value = jnp.sum(value * input)
      return value

    grad_fn = jax.grad(helper, argnums=0)
    return grad_fn(output.real, input.real) + 1j * grad_fn(output.imag, input.imag)

  def _gridding3(output, input, coord, width, param):
    def helper(output, input):
      value = interpolate3(output, coord, width, param)
      value = jnp.sum(value * input)
      return value

    grad_fn = jax.grad(helper, argnums=0)
    return grad_fn(output.real, input.real) + 1j * grad_fn(output.imag, input.imag)

  return _gridding1, _gridding2, _gridding3


_interpolate = {}
_gridding = {}
for kernel in KERNELS:
  _interpolate[kernel] = _get_interpolate(kernel)
  _gridding[kernel] = _get_gridding(kernel)
