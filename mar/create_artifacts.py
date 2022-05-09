"""
Code adapted from Lequan Yu's repo.
"""

from mar.utils import pkev2kvp, get_mar_params, interpolate_projection
import scipy.io as sio
import numpy as np
import math
from PIL import Image
import odl
import scipy.ndimage

MARpara = get_mar_params('assets/metal_masks')
metal_masks = sio.loadmat('assets/metal_masks/SampleMasks.mat')['CT_samples_bwMetal']


def convert_png_to_HU(image):
  return image * (3200 + 1024) - 1024


def convert_HU_to_png(image):
  return (image + 1024) / (3200 + 1024)


class Params:
  def __init__(self, image_size):
    self.param = {}
    self.reso = 512 / image_size * 0.03

    # image
    self.param['nx_h'] = image_size
    self.param['ny_h'] = image_size
    self.param['sx'] = self.param['nx_h'] * self.reso
    self.param['sy'] = self.param['ny_h'] * self.reso

    ## view
    self.param['startangle'] = np.pi / 2.
    self.param['endangle'] = np.pi * 3. / 2.

    self.param['nProj'] = image_size

    ## detector
    self.param['su'] = np.sqrt(self.param['sx'] ** 2 + self.param['sy'] ** 2)
    self.param['nu_h'] = math.ceil(np.sqrt(2.) * image_size)

    self.param['u_water'] = 0.192  # 0.0205


def build_geometry(param):
  reco_space_h = odl.uniform_discr(
    min_pt=[-param.param['sx'] / 2., -param.param['sy'] / 2.],
    max_pt=[param.param['sx'] / 2., param.param['sy'] / 2.], shape=[param.param['nx_h'], param.param['ny_h']],
    dtype='float32')

  angle_partition = odl.uniform_partition(param.param['startangle'], param.param['endangle'],
                                          param.param['nProj'])

  detector_partition_h = odl.uniform_partition(-(param.param['su']) / 2., (param.param['su']) / 2.,
                                               param.param['nu_h'])

  geometry_h = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition_h)

  ray_trafo_hh = odl.tomo.RayTransform(reco_space_h, geometry_h, impl='astra_cuda')
  FBPOper_hh = odl.tomo.fbp_op(ray_trafo_hh, filter_type='Ram-Lak', frequency_scaling=1.)

  return reco_space_h, ray_trafo_hh, FBPOper_hh


def create_mar_artifacts(image, mask_id):
  image_size = image.shape[0]
  img = convert_png_to_HU(image)
  params = Params(image_size)
  reco_space, ray_trafo, FBPOper = build_geometry(params)
  MiuWater = MARpara['MiuWater']
  threshWater = MARpara['threshWater']
  threshBone = MARpara['threshBone']

  img = img / 1000 * MiuWater + MiuWater
  imgWater = np.zeros_like(img)
  imgBone = np.zeros_like(img)
  bwWater = img <= threshWater
  bwBone = img >= threshBone
  bwBoth = (1 - bwWater - bwBone) > 0.5
  imgWater[bwWater] = img[bwWater]
  imgBone[bwBone] = img[bwBone]
  imgBone[bwBoth] = (img[bwBoth] - threshWater) / (threshBone - threshWater) * img[bwBoth]
  imgWater[bwBoth] = img[bwBoth] - imgBone[bwBoth]

  Pwater_kev = ray_trafo(imgWater)
  Pbone_kev = ray_trafo(imgBone)

  NumofRou, NumofTheta = Pwater_kev.shape
  projkevAll = np.zeros((NumofRou, NumofTheta, 3))
  projkevAll[:, :, 0] = Pwater_kev
  projkevAll[:, :, 1] = Pbone_kev
  projkvp = pkev2kvp(projkevAll, MARpara['spectrum'], MARpara['energies'], MARpara['kev'], MARpara['MiuAll'])

  # Poisson noise
  scatterPhoton = 20
  temp = np.round(np.exp(-projkvp) * MARpara['photonNum'])
  temp = temp + scatterPhoton
  ProjPhoton = np.random.poisson(temp)
  ProjPhoton[ProjPhoton == 0] = 1
  projkvpNoise = -np.log(ProjPhoton / MARpara['photonNum'])

  # correction
  p1 = np.reshape(projkvpNoise, (NumofRou * NumofTheta, 1))
  p1BHC = np.matmul(np.concatenate([p1, p1 ** 2, p1 ** 3], axis=1), np.asarray(MARpara['paraBHC']))
  poly_sinogram = np.reshape(p1BHC, (NumofRou, NumofTheta))
  poly_CT = np.asarray(FBPOper(poly_sinogram))

  imgMetal = metal_masks[:, :, mask_id].copy()
  imgMetal = np.rot90(imgMetal)
  imgMetal = np.array(Image.fromarray(imgMetal).resize((image_size, image_size)))

  Pmetal_kev = np.asarray(ray_trafo(imgMetal))
  metal_trace = Pmetal_kev > 0
  Pmetal_kev = MARpara['metalAtten'] * Pmetal_kev

  # partial volume effect
  Pmetal_kev_bw = scipy.ndimage.binary_erosion(Pmetal_kev > 0, structure=np.ones((1, 3)))
  Pmetal_edge = np.logical_xor((Pmetal_kev > 0), Pmetal_kev_bw)
  Pmetal_kev[Pmetal_edge] = Pmetal_kev[Pmetal_edge] / 4

  # sinogram with metal
  projkevAllLocal = projkevAll.copy()
  projkevAllLocal[:, :, 2] = Pmetal_kev
  projkvpMetal = pkev2kvp(projkevAllLocal, MARpara['spectrum'], MARpara['energies'], MARpara['kev'],
                          MARpara['MiuAll'])

  # Add Poisson noi44
  temp = np.round(np.exp(-projkvpMetal) * MARpara['photonNum'])
  temp = temp + scatterPhoton
  ProjPhoton = np.random.poisson(temp)
  ProjPhoton[ProjPhoton == 0] = 1
  projkvpMetalNoise = -np.log(ProjPhoton / MARpara['photonNum'])

  # correction
  p1 = np.reshape(projkvpMetalNoise, (NumofRou * NumofTheta, 1))
  p1BHC = np.matmul(np.concatenate([p1, p1 ** 2, p1 ** 3], axis=1), np.asarray(MARpara['paraBHC']))
  ma_sinogram = np.reshape(p1BHC, (NumofRou, NumofTheta))
  LI_sinogram = interpolate_projection(ma_sinogram, metal_trace)
  ma_CT = np.asarray(FBPOper(ma_sinogram))
  LI_CT = np.asarray(FBPOper(LI_sinogram))

  ma_CT = convert_HU_to_png((ma_CT - MiuWater) * 1000 / MiuWater)
  LI_CT = convert_HU_to_png((LI_CT - MiuWater) * 1000 / MiuWater)

  return ma_CT, ma_sinogram, imgMetal, metal_trace, LI_CT, poly_CT, image
