# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch


def quaternion_to_rotation_matrix(quaternion):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quaternion: size = [batch_size, 4], format: (w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [batch_size, 3, 3]
    """
    normalized = quaternion / quaternion.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = normalized[:, 0], normalized[:, 1], normalized[:, 2], normalized[:, 3]

    batch_size = quaternion.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotation_matrix = torch.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
        2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
        2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2
    ], dim=1).view(batch_size, 3, 3)
    return rotation_matrix


def batch_axis_angle_to_rotation_matrix(axis_angle):
    """Convert batch of axis-angle representations to flattened rotation matrices.
    Args:
        axis_angle: (N, 3) axis-angle vectors
    Returns:
        (N, 9) flattened rotation matrices
    """
    angle_norm = torch.norm(axis_angle + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(angle_norm, -1)
    axis_normalized = torch.div(axis_angle, angle)
    half_angle = angle * 0.5
    cos_half = torch.cos(half_angle)
    sin_half = torch.sin(half_angle)
    quaternion = torch.cat([cos_half, sin_half * axis_normalized], dim=1)
    rotation_matrix = quaternion_to_rotation_matrix(quaternion)
    return rotation_matrix.view(rotation_matrix.shape[0], 9)


def decompose_axis_angle(vector):
    """Decompose axis-angle vector into unit axis and angle.
    Args:
        vector: (N, 3) axis-angle vectors
    Returns:
        axes: (N, 3) unit rotation axes
        angles: (N,) rotation angles
    """
    angles = torch.norm(vector, 2, 1)
    axes = vector / angles.unsqueeze(1)
    return axes, angles


def axis_angle_to_rotation_matrices(pose_vectors):
    """Convert axis-angle pose parameters to concatenated flattened rotation matrices.
    Args:
        pose_vectors: (batch_size, num_joints*3) pose parameters in axis-angle representation
    Returns:
        (batch_size, num_joints*9) concatenated flattened rotation matrices
    """
    num_joints = int(pose_vectors.shape[1] / 3)
    rotation_matrices = []
    for joint_idx in range(num_joints):
        joint_axis_angle = pose_vectors[:, joint_idx * 3:(joint_idx + 1) * 3]
        rotation_matrix = batch_axis_angle_to_rotation_matrix(joint_axis_angle)
        rotation_matrices.append(rotation_matrix)

    return torch.cat(rotation_matrices, 1)


def pad_homogeneous_row(transform_3x4):
    """Append a [0, 0, 0, 1] row to a batch of 3x4 transformation matrices to make them 4x4.
    Args:
        transform_3x4: (batch_size, 3, 4) transformation matrices
    Returns:
        (batch_size, 4, 4) homogeneous transformation matrices
    """
    batch_size = transform_3x4.shape[0]
    bottom_row = transform_3x4.new([0.0, 0.0, 0.0, 1.0])
    bottom_row.requires_grad = False
    return torch.cat([transform_3x4, bottom_row.view(1, 1, 4).repeat(batch_size, 1, 1)], 1)


def pad_translation_columns(translation_4x1):
    """Prepend three zero columns to a batch of 4x1 translation vectors to form 4x4 matrices.
    Args:
        translation_4x1: (batch_size, 4, 1) translation column vectors
    Returns:
        (batch_size, 4, 4) matrices with zero rotation and given translation
    """
    batch_size = translation_4x1.shape[0]
    zero_columns = translation_4x1.new_zeros((batch_size, 4, 3))
    zero_columns.requires_grad = False
    return torch.cat([zero_columns, translation_4x1], 2)


def subtract_flat_identity(rotation_matrices, hands=False):
    """Subtract flattened identity matrices from concatenated flattened rotation matrices.
    This computes the pose blend shape coefficients (R - I) for each joint.
    Args:
        rotation_matrices: (batch_size, num_joints*9) flattened rotation matrices
        hands: if True, use 51 joints (SMPL-H), otherwise 23 joints (SMPL)
    Returns:
        (batch_size, num_joints*9) rotation matrices minus identity
    """
    num_joints = 51 if hands else 23
    identity_flat = torch.eye(
        3, dtype=rotation_matrices.dtype, device=rotation_matrices.device
    ).view(1, 9).repeat(rotation_matrices.shape[0], num_joints)
    return rotation_matrices - identity_flat


def to_list(tensor):
    # type: (List[int]) -> List[int]
    return tensor
