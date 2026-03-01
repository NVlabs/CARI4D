# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
common functions for image operations
"""

import cv2
import numpy as np
from scipy.spatial.distance import cdist

def sample_mask(K, N, mask_hum):
    """
    sample points from mask
    K: number of farthest samples
    N: number of random samples
    """
    yid, xid = np.where(mask_hum > 127)
    points_mask = np.stack([xid, yid], -1) # correct way to get 2d coordinates
    rand_inds = np.random.choice(len(points_mask), N, replace=False)
    sampled_points = [points_mask[x] for x in rand_inds]
    for _ in range(K - 1):
        # Compute the distance from the last chosen point to all other points
        distances = cdist(sampled_points, points_mask, metric='euclidean')

        # Find the point with the maximum minimum distance to the sampled points
        farthest_point = points_mask[np.argmax(np.min(distances, axis=0))]

        # Add the farthest point to the list of sampled points
        sampled_points.append(farthest_point)
    print("FPS done!")
    return sampled_points

def crop(img, center, crop_size):
    """
    crop image around the given center, pad zeros for borders
    :param img:
    :param center: np array
    :param crop_size: np array or a float size of the resulting crop
    :return: a square crop around the center
    """
    assert isinstance(img, np.ndarray)
    h, w = img.shape[:2]
    topleft = np.round(center - crop_size / 2).astype(int)
    bottom_right = np.round(center + crop_size / 2).astype(int)

    x1 = max(0, topleft[0])
    y1 = max(0, topleft[1])
    x2 = min(w - 1, bottom_right[0])
    y2 = min(h - 1, bottom_right[1])
    cropped = img[y1:y2, x1:x2]

    p1 = max(0, -topleft[0])  # padding in x, top
    p2 = max(0, -topleft[1])  # padding in y, top
    p3 = max(0, bottom_right[0] - w + 1)  # padding in x, bottom
    p4 = max(0, bottom_right[1] - h + 1)  # padding in y, bottom

    dim = len(img.shape)
    if dim == 3:
        padded = np.pad(cropped, [[p2, p4], [p1, p3], [0, 0]])
    elif dim == 2:
        padded = np.pad(cropped, [[p2, p4], [p1, p3]])
    else:
        raise NotImplemented
    return padded


def resize(img, img_size, mode=cv2.INTER_LINEAR):
    """
    resize image to the input
    :param img:
    :param img_size: (width, height) of the target image size
    :param mode:
    :return:
    """
    h, w = img.shape[:2]
    load_ratio = 1.0 * w / h
    netin_ratio = 1.0 * img_size[0] / img_size[1]
    assert load_ratio == netin_ratio, "image aspect ration not matching, given image: {}, net input: {}".format(
        img.shape, img_size)
    resized = cv2.resize(img, img_size, interpolation=mode)
    return resized


def masks2bbox(masks, threshold=127):
    """

    :param masks:
    :param threshold:
    :return: bounding box corner coordinate
    """
    mask_comb = np.zeros_like(masks[0], dtype=bool)
    for m in masks:
        mask_comb = mask_comb | (m > threshold)

    yid, xid = np.where(mask_comb)
    bmin = np.array([xid.min(), yid.min()])
    bmax = np.array([xid.max(), yid.max()])
    return bmin, bmax


def compute_translation(crop_center, crop_size, is_behave=True, std_coverage=3.5):
    """
    solve for an optimal translation that project gaussian in origin to the crop
    Parameters
    ----------
    crop_center: (x, y) of the crop center
    crop_size: float, the size of the square crop
    std_coverage: which edge point should be projected back to the edge of the 2d crop

    Returns
    -------
    the estimated translation

    """
    x0, y0 = crop_center
    x1, y1 = x0 + crop_size/2, y0
    x2, y2 = x0 - crop_size/2, y0
    x3, y3 = x0, y0 + crop_size/2.
    # predefined kinect intrinsics
    if is_behave:
        fx = 979.7844
        fy = 979.840
        cx = 1018.952
        cy = 779.486
    else:
        # intercap camera
        fx, fy = 918.457763671875, 918.4373779296875
        cx, cy = 956.9661865234375, 555.944580078125

    # Construct the matrix
    # First two equations: origin (0, 0, 0) is projected to the crop center
    # Last two equations: edge point (std_coverage, 0, z) is projected to the edge of crop
    A = np.array([
        [fx, 0, cx-x0, cx-x0],
        [0, fy, cy-y0, cy-y0],
        [fx, 0, fx-x1,   0],
        [0, fy, cy-y1,   0]
    ])
    b = np.array([0, 0, -std_coverage * fx, 0]).reshape((-1, 1))  # 3.5->half of 7.0
    x = np.matmul(np.linalg.inv(A), b)

    # A is always a full-rank matrix

    return x.flatten()[:3]
