# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def compute_scale_and_shift_robust(pred, target, mask):
    "more robust version propose in the paper: https://arxiv.org/pdf/1907.01341, input are np arrays"
    t_pr = np.median(pred[mask])
    t_tar = np.median(target[mask])
    scale_pr = np.mean(np.abs(pred[mask] - t_pr))
    scale_tar = np.mean(np.abs(target[mask] - t_tar))
    scale = scale_tar / scale_pr
    shift = t_tar - scale * t_pr
    return scale, shift 

def bilateral_filter_depth_cpu_fast(depth: np.ndarray,
                                    radius: int = 2,
                                    zfar: float = 100.0,
                                    sigmaD: float = 2.0,
                                    sigmaR: float = 100000.0) -> np.ndarray:
    """
    Vectorized CPU implementation matching Utils.bilateral_filter_depth semantics:
    - valid pixels: 0.001 <= depth < zfar
    - local valid-mean gate: |neighbor - local_valid_mean| < 0.01
    - weight = exp(-((dx^2+dy^2)/(2*sigmaD^2) + (center-neighbor)^2/(2*sigmaR^2)))
    - if no valid weights or no valid neighbors => output 0
    """
    assert depth.ndim == 2
    depth = np.asarray(depth, dtype=np.float32, order='C')
    H, W = depth.shape
    k = 2 * radius + 1

    # Pad image and masks so sliding windows cover borders identically to in-bounds-only iteration
    depth_pad = np.pad(depth, radius, mode='constant', constant_values=0.0)
    valid = (depth >= 0.001) & (depth < zfar)
    valid_pad = np.pad(valid.astype(np.uint8), radius, mode='constant', constant_values=0)

    # Sliding windows (H,W,k,k)
    Dp = sliding_window_view(depth_pad, (k, k))
    Vp = sliding_window_view(valid_pad, (k, k)).astype(bool)

    # Local valid neighbor count and mean
    num_valid = Vp.sum(axis=(2, 3)).astype(np.float32)                # (H,W)
    sum_valid = (Dp * Vp).sum(axis=(2, 3), dtype=np.float32)          # (H,W)
    mean_valid = sum_valid / np.maximum(num_valid, 1.0)               # safe divide

    # Spatial Gaussian (k,k)
    ys = np.arange(-radius, radius + 1, dtype=np.float32)
    xs = np.arange(-radius, radius + 1, dtype=np.float32)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing='ij')
    spatial = np.exp(-(grid_x**2 + grid_y**2) / (2.0 * sigmaD * sigmaD)).astype(np.float32)

    # Gate by local valid mean
    gate = Vp & (np.abs(Dp - mean_valid[..., None, None]) < 0.01)

    # Range Gaussian around center depth
    center = depth[..., None, None]                                   # (H,W,1,1)
    range_w = np.exp(-((center - Dp) ** 2) / (2.0 * sigmaR * sigmaR)).astype(np.float32)

    # Weights and normalization
    weights = range_w * spatial[None, None, ...]
    weights = np.where(gate, weights, 0.0).astype(np.float32)
    sum_w = weights.sum(axis=(2, 3), dtype=np.float32)                # (H,W)

    # Weighted sum
    num = (weights * Dp).sum(axis=(2, 3), dtype=np.float32)           # (H,W)
    out = np.zeros((H, W), dtype=np.float32)
    valid_out = (sum_w > 0.0) & (num_valid > 0.0)
    out[valid_out] = (num[valid_out] / sum_w[valid_out]).astype(np.float32)
    return out


def erode_depth_cpu_fast(depth: np.ndarray,
                         radius: int = 2,
                         depth_diff_thres: float = 0.001,
                         ratio_thres: float = 0.8,
                         zfar: float = 100.0) -> np.ndarray:
    """
    Vectorized CPU implementation matching Utils.erode_depth semantics:
    - For each pixel, count neighbors in the window (in-bounds only).
    - A neighbor is 'bad' if invalid (<0.001 or >= zfar) or |neighbor-center| > depth_diff_thres.
    - If bad_ratio > ratio_thres => output 0, else keep center depth.
    - If center invalid => output 0.
    """
    assert depth.ndim == 2
    depth = np.asarray(depth, dtype=np.float32, order='C')
    H, W = depth.shape
    k = 2 * radius + 1

    # Pad for windows
    depth_pad = np.pad(depth, radius, mode='constant', constant_values=0.0)

    # Sliding windows (H,W,k,k)
    Dp = sliding_window_view(depth_pad, (k, k))                        # (H,W,k,k)

    # In-bounds neighbor count matches kernel's 'total'
    ones = np.ones((H, W), dtype=np.uint8)
    ones_pad = np.pad(ones, radius, mode='constant', constant_values=0)
    totals = sliding_window_view(ones_pad, (k, k)).sum(axis=(2, 3)).astype(np.float32)  # (H,W)

    center = depth[..., None, None]                                    # (H,W,1,1)
    bad = ((Dp < 0.001) | (Dp >= zfar) | (np.abs(Dp - center) > depth_diff_thres))
    bad_count = bad.sum(axis=(2, 3)).astype(np.float32)

    ratio = bad_count / np.maximum(totals, 1.0)
    center_invalid = (depth < 0.001) | (depth >= zfar)
    out = np.where(center_invalid | (ratio > ratio_thres), 0.0, depth).astype(np.float32)
    return out