# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
test out the oneeuro filter 
"""
import sys, os
sys.path.append(os.getcwd())
import cv2
import numpy as np
import glob
from tqdm import tqdm
import pickle as pkl
import os.path as osp

from OneEuroFilter import OneEuroFilter


import numpy as np
from scipy.spatial.transform import Rotation as R

# ---------- One Euro filter (scalar) ----------
class OneEuro1D:
    def __init__(self, min_cutoff=0.5, beta=2.0, dcutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.dcutoff = float(dcutoff)
        self._x_hat = None
        self._dx_hat = 0.0

    @staticmethod
    def _alpha(fc, dt):
        tau = 1.0 / (2.0 * np.pi * fc)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self._x_hat, self._dx_hat = None, 0.0

    def __call__(self, x, dt):
        x = float(x)
        if self._x_hat is None:
            self._x_hat = x
            self._dx_hat = 0.0
            return x
        # derivative estimate (finite diff) then LPF it
        dx = (x - self._x_hat) / dt
        a_d = self._alpha(self.dcutoff, dt)
        self._dx_hat = a_d * dx + (1 - a_d) * self._dx_hat
        # dynamic cutoff
        fc = self.min_cutoff + self.beta * abs(self._dx_hat)
        a = self._alpha(fc, dt)
        self._x_hat = a * x + (1 - a) * self._x_hat
        return self._x_hat

class OneEuroVec3:
    def __init__(self, min_cutoff=0.5, beta=2.0, dcutoff=1.0):
        self.fx, self.fy, self.fz = (
            OneEuro1D(min_cutoff, beta, dcutoff),
            OneEuro1D(min_cutoff, beta, dcutoff),
            OneEuro1D(min_cutoff, beta, dcutoff),
        )
    def reset(self):
        self.fx.reset(); self.fy.reset(); self.fz.reset()
    def __call__(self, v3, dt):
        return np.array([self.fx(v3[0], dt), self.fy(v3[1], dt), self.fz(v3[2], dt)], dtype=np.float64)

# ---------- A) Tangent-space One Euro (log/exp) ----------
def smooth_rotations_logexp_oneeuro(R_seq, dt, min_cutoff=0.5, beta=2.0, dcutoff=1.0):
    """
    R_seq: (L,3,3) array of rotation matrices
    dt:    constant timestep (seconds)
    returns: (L,3,3) filtered rotation matrices
    """
    R_seq = np.asarray(R_seq, dtype=np.float64)
    L = R_seq.shape[0]
    rots_raw = R.from_matrix(R_seq)
    rots_out = [rots_raw[0]]
    filt = OneEuroVec3(min_cutoff, beta, dcutoff)

    for i in range(1, L):
        r_prev = rots_out[-1]
        r_cur  = rots_raw[i]
        # relative rotation in SO(3)
        r_rel = r_prev.inv() * r_cur
        # log map -> rotvec in R^3
        rotvec = r_rel.as_rotvec()              # axis * angle, |rotvec| in radians (principal)
        # One Euro on the tangent vector
        rotvec_f = filt(rotvec, dt)
        # exp map back and compose
        step = R.from_rotvec(rotvec_f)
        r_f = r_prev * step
        rots_out.append(r_f)

    return R.from_quat([r.as_quat() for r in rots_out]).as_matrix()

# ---------- B) Geodesic One Euro (adaptive SLERP step) ----------
def smooth_rotations_geodesic_oneeuro(R_seq, dt, min_cutoff=0.5, beta=2.0):
    """
    Same I/O as above. Uses an adaptive geodesic step (like SLERP with One-Euro α).
    """
    R_seq = np.asarray(R_seq, dtype=np.float64)
    L = R_seq.shape[0]
    rots_raw = R.from_matrix(R_seq)
    rots_out = [rots_raw[0]]

    def alpha_from_fc(fc):
        tau = 1.0 / (2.0 * np.pi * fc)
        return 1.0 / (1.0 + tau / dt)

    for i in range(1, L):
        r_prev = rots_out[-1]
        r_cur  = rots_raw[i]
        r_rel = r_prev.inv() * r_cur
        rotvec = r_rel.as_rotvec()
        theta = np.linalg.norm(rotvec)          # radians (principal)
        omega = theta / dt                       # angular speed
        fc = min_cutoff + beta * omega
        alpha = alpha_from_fc(fc)
        # take a fraction alpha along the geodesic
        step = R.from_rotvec(alpha * rotvec)
        r_f = r_prev * step
        rots_out.append(r_f)

    return R.from_quat([r.as_quat() for r in rots_out]).as_matrix()


def filter_3axis(data, freq=120):
    "input data: (L, 3), np array, filter each axis separately"
    config = {
                    'freq': freq,       # Hz
                    'mincutoff': 1.0,  # Hz
                    'beta': 0.1,       
                    'dcutoff': 1.0    
                    }
    oneeuro = OneEuroFilter(**config)
    filtered_data = []
    for axis in range(3):
        oneeuro.reset()
        data_axis = data[:, axis]
        filtered_data_axis = []
        for i in range(len(data_axis)):
            filtered_data_axis.append(oneeuro(data_axis[i], i/freq))
        filtered_data_axis = np.array(filtered_data_axis)
        filtered_data.append(filtered_data_axis)
    filtered_data = np.stack(filtered_data, axis=1)
    return filtered_data


