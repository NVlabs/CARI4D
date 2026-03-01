# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import chumpy
import numpy as np
import cv2

class Rodrigues(chumpy.Ch):
    dterms = 'rt'

    def compute_r(self):
        return cv2.Rodrigues(self.rt.r)[0]

    def compute_dr_wrt(self, wrt):
        if wrt is self.rt:
            return cv2.Rodrigues(self.rt.r)[1].T

def verts_core(pose, v, J, weights, kintree_table, bs_style, want_Jtr=False, xp=chumpy):
    if xp == chumpy:
        assert (hasattr(pose, 'dterms'))
        assert (hasattr(v, 'dterms'))
        assert (hasattr(J, 'dterms'))
        assert (hasattr(weights, 'dterms'))

    assert (bs_style == 'lbs')
    result = lbs_verts_core(pose, v, J, weights, kintree_table, want_Jtr, xp)

    return result

def global_rigid_transformation(pose, J, kintree_table, xp):
    results = {}
    pose = pose.reshape((-1, 3))
    id_to_col = {kintree_table[1, i]: i for i in range(kintree_table.shape[1])}
    parent = {i: id_to_col[kintree_table[0, i]] for i in range(1, kintree_table.shape[1])}

    if xp == chumpy:
        rodrigues = lambda x: Rodrigues(x)
    else:
        import cv2
        rodrigues = lambda x: cv2.Rodrigues(x)[0]

    with_zeros = lambda x: xp.vstack((x, xp.array([[0.0, 0.0, 0.0, 1.0]])))
    results[0] = with_zeros(xp.hstack((rodrigues(pose[0, :]), J[0, :].reshape((3, 1)))))

    for i in range(1, kintree_table.shape[1]):
        results[i] = results[parent[i]].dot(with_zeros(xp.hstack((
            rodrigues(pose[i, :]),
            ((J[i, :] - J[parent[i], :]).reshape((3, 1)))
        ))))

    pack = lambda x: xp.hstack([np.zeros((4, 3)), x.reshape((4, 1))])

    results = [results[i] for i in sorted(results.keys())]
    results_global = results

    if True:
        results2 = [results[i] - (pack(
            results[i].dot(xp.concatenate(((J[i, :]), 0))))
        ) for i in range(len(results))]
        results = results2
    result = xp.dstack(results)
    return result, results_global


def lbs_verts_core(pose, v, J, weights, kintree_table, want_Jtr=False, xp=chumpy):
    A, A_global = global_rigid_transformation(pose, J, kintree_table, xp)
    T = A.dot(weights.T)

    rest_shape_h = xp.vstack((v.T, np.ones((1, v.shape[0]))))

    v = (T[:, 0, :] * rest_shape_h[0, :].reshape((1, -1)) +
         T[:, 1, :] * rest_shape_h[1, :].reshape((1, -1)) +
         T[:, 2, :] * rest_shape_h[2, :].reshape((1, -1)) +
         T[:, 3, :] * rest_shape_h[3, :].reshape((1, -1))).T

    v = v[:, :3]

    if not want_Jtr:
        return v
    Jtr = xp.vstack([g[:3, 3] for g in A_global])
    return (v, Jtr)
