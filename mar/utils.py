"""
Code from Lequan Yu
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy import io as sio


def pkev2kvp(projkevAll, spectrum, energies, kev, MiuAll):
  # projkevAll: all single materials' projection at given kev, 3d: (bin, view, material)
  # spectrum
  # energies: energies girds, e.g. [20:120]
  # kev: kev under which projkev is obtained
  # Miu: mass attenuation of the all given material over energies, 3d: (energy, mode, materials)

  AttenuMode = 7
  if projkevAll.ndim < 3:
    projkevAll = projkevAll[:, :, None]
    MiuAll = MiuAll[:, :, None]

  matNum = projkevAll.shape[2]  # number of materials
  projAll = np.zeros_like(projkevAll)
  ProjEnergy = np.zeros((projkevAll.shape[0], projkevAll.shape[1]))
  projkvp = np.zeros((projkevAll.shape[0], projkevAll.shape[1]))

  for ien in energies:
    for imat in range(matNum):
      # projection at current energy for each material component
      projAll[:, :, imat] = MiuAll[ien - 1, AttenuMode - 1, imat] / MiuAll[kev - 1, AttenuMode - 1, imat] * projkevAll[
                                                                                                            :, :, imat]
    proj = np.sum(projAll, axis=2)
    Ptmp = spectrum[ien - 1] * np.exp(-proj)  # according to the spectrum ratio
    ProjEnergy = ProjEnergy + Ptmp
  ProjEnergyBlankRatio = np.sum(spectrum[energies - 1]) * np.ones_like(projkvp)
  projkvp = -np.log(ProjEnergy / ProjEnergyBlankRatio)

  return projkvp


def interpolate_projection(proj, metalTrace):
  # projection linear interpolation
  # Input:
  # proj:         uncorrected projection
  # metalTrace:   metal trace in projection domain (binary image)
  # Output:
  # Pinterp:      linear interpolation corrected projection

  Pinterp = proj.copy()
  for i in range(Pinterp.shape[0]):
    mslice = metalTrace[i]
    pslice = Pinterp[i]

    metalpos = np.nonzero(mslice == 1)[0]
    nonmetalpos = np.nonzero(mslice == 0)[0]
    pnonmetal = pslice[nonmetalpos]
    pslice[metalpos] = interp1d(nonmetalpos, pnonmetal)(metalpos)
    # aa = np.interp(metalpos, nonmetalpos, pnonmetal)
    Pinterp[i] = pslice

  return Pinterp


def marBHC(proj, metalBW, ray_trafo, FBPOper, param):
  # his code is to reduce metal artifacts using a first order beam hardening correction method

  views, bins = proj.shape

  # projection of metal
  projMetal = np.asarray(ray_trafo(metalBW) / param.reso)

  # LI projection
  Pinterp = interpolate_projection(proj, projMetal > 0)

  projDiff = proj - Pinterp
  projDiff1 = np.reshape(projDiff, (bins * views))
  projMetal1 = np.reshape(projMetal, (bins * views))
  projMetalbw1 = np.reshape(projMetal > 0, (bins * views))

  # first order beam hardening correction
  A = np.zeros((bins * views, 3))
  A[:, 0] = projMetalbw1 * projMetal1  # only consider projections affected by metal
  A[:, 1] = projMetalbw1 * projMetal1 ** 2
  A[:, 2] = projMetalbw1 * projMetal1 ** 3
  X0 = np.linalg.lstsq(A, projMetalbw1 * projDiff1)[0]  # coefficients of the polynomial
  X0 = np.squeeze(X0)
  projFit1 = np.zeros((bins * views))
  for icoe in range(A.shape[1]):
    projFit1 = projFit1 + X0[icoe] * A[:, icoe]

  # difference between projections of piece-wise constant image and its hardened one
  projDelta1 = projMetalbw1 * (X0[0] * projMetal1 - projFit1)
  projBHC = proj + np.reshape(projDelta1, (views, bins))

  # reconstruction
  imBHC = np.asarray(FBPOper(projBHC))

  return imBHC


def get_mar_params(param_root):
  # materials
  a = sio.loadmat(param_root + '/MiuofH2O.mat')
  MiuofH2O = a['MiuofH2O']

  a = sio.loadmat(param_root + '/MiuofTi.mat')
  MiuofTi = a['MiuofTi']

  a = sio.loadmat(param_root + '/MiuofFe.mat')
  MiuofFe = a['MiuofFe']

  a = sio.loadmat(param_root + '/MiuofCu.mat')
  MiuofCu = a['MiuofCu']

  a = sio.loadmat(param_root + '/MiuofAu.mat')
  MiuofAu = a['MiuofAu']

  a = sio.loadmat(param_root + '/MiuofBONE_Cortical_ICRU44.mat')
  MiuofBONE_Cortical_ICRU44 = a['MiuofBONE_Cortical_ICRU44']

  # spectrum data
  a = sio.loadmat(param_root + '/GE14Spectrum120KVP.mat')
  GE14Spectrum120KVP = a['GE14Spectrum120KVP']

  kVp = 120
  energies = np.arange(20, kVp + 1)
  kev = 70
  photonNum = 2 * 10 ** 7
  materialID = 0

  threshWaterHU = 100
  threshBoneHU = 1500
  MiuWater = 0.192
  threshWater = threshWaterHU / 1000 * MiuWater + MiuWater
  threshBone = threshBoneHU / 1000 * MiuWater + MiuWater

  MiuofMetal = np.stack((MiuofTi[:kVp, :], MiuofFe[:kVp, :], MiuofCu[:kVp, :], MiuofAu[:kVp, :]), axis=2)

  densityMetal = [4.5, 7.8, 8.9, 2]
  metalAtten = densityMetal[materialID] * MiuofMetal[kev - 1, 6, materialID]

  # materials
  MiuAll = np.stack((MiuofH2O[:kVp, :], MiuofBONE_Cortical_ICRU44[:kVp, :], MiuofMetal[:kVp, :, materialID]), axis=2)

  spectrum = GE14Spectrum120KVP[:kVp, 1]

  # water BHC
  thickness = np.arange(0, 50.01, 0.05).reshape(-1, 1)  # thickness of water, cm
  pwaterkev = MiuofH2O[kev - 1, 6] * thickness
  pwaterkvp = pkev2kvp(pwaterkev, spectrum, energies, kev, MiuofH2O[:kVp, :])
  A = np.concatenate((pwaterkvp, pwaterkvp ** 2, pwaterkvp ** 3), axis=1)
  paraBHC = np.matmul(np.linalg.pinv(A), pwaterkev)

  # return
  MARpara = {}
  MARpara['kev'] = kev
  MARpara['spectrum'] = spectrum
  MARpara['energies'] = energies
  MARpara['photonNum'] = photonNum
  MARpara['MiuWater'] = MiuWater
  MARpara['MiuAll'] = MiuAll
  MARpara['threshWater'] = threshWater
  MARpara['threshBone'] = threshBone
  MARpara['paraBHC'] = paraBHC
  MARpara['metalAtten'] = metalAtten  # buyiyang

  return MARpara
