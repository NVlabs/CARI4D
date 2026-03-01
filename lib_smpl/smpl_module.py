# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
reimplementation of SMPL and SMPL-H layer
"""

import os

import numpy as np
import torch
from torch.nn import Module
from .io import load_smpl_model_data
from .smpl_utils import (
    axis_angle_to_rotation_matrices,
    pad_homogeneous_row,
    pad_translation_columns,
    to_list,
    subtract_flat_identity,
)


class SMPLLayer(Module):
    __constants__ = ['kintree_parents', 'gender', 'center_idx', 'num_joints']

    def __init__(self,
                 center_idx=None,
                 gender='neutral',
                 model_root='smpl/native/models',
                 num_betas=300,
                 hands=False):
        """
        Args:
            center_idx: index of center joint in our computations,
            model_root: path to pkl files for the model
            gender: 'neutral' (default) or 'female' or 'male', for smplh, only supports male or female
        """
        super().__init__()

        self.center_idx = center_idx
        self.gender = gender

        self.model_root = model_root
        self.hands = hands
        if self.hands:
            assert self.gender in ['male', 'female'], \
                'SMPL-H model only supports male or female, not {}'.format(self.gender)
            self.model_path = os.path.join(model_root, f"SMPLH_{self.gender}.pkl")
        else:
            self.model_path = os.path.join(model_root, f"SMPL_{self.gender}.pkl")

        model_data = load_smpl_model_data(self.model_path)
        self.smpl_data = model_data
        self.faces = model_data['f'].astype(np.int32)

        self.register_buffer('th_betas',
                             torch.Tensor(model_data['betas'].r.copy()).unsqueeze(0))
        self.register_buffer('th_shapedirs',
                             torch.Tensor(model_data['shapedirs'][:, :, :num_betas].r.copy()))
        self.register_buffer('th_posedirs',
                             torch.Tensor(model_data['posedirs'].r.copy()))
        self.register_buffer(
            'th_v_template',
            torch.Tensor(model_data['v_template'].r.copy()).unsqueeze(0))
        self.register_buffer(
            'th_J_regressor',
            torch.Tensor(np.array(model_data['J_regressor'].toarray())))
        self.register_buffer('th_weights',
                             torch.Tensor(model_data['weights'].r.copy()))
        self.register_buffer('th_faces',
                             torch.Tensor(model_data['f'].astype(np.int32)).long())

        # Kinematic chain params
        self.kintree_table = model_data['kintree_table']
        parent_indices = list(self.kintree_table[0].tolist())
        self.kintree_parents = parent_indices
        self.num_joints = len(parent_indices)  # 24

    def forward(self,
                pose_axis,
                betas=torch.zeros(1),
                transl=torch.zeros(1, 3),
                scale=1.,
                ret_glb_rot=False):
        """
        Args:
        pose_axis (Tensor (batch_size x 72)): pose parameters in axis-angle representation
        betas (Tensor (batch_size x 10)): if provided, uses given shape parameters
        transl (Tensor (batch_size x 3)): if provided, applies trans to joints and vertices
        """

        batch_size = pose_axis.shape[0]

        # Convert axis-angle representation to flattened rotation matrices
        pose_rotation_matrices = axis_angle_to_rotation_matrices(pose_axis)

        # Separate root rotation (global) from per-joint rotations
        root_rotation = pose_rotation_matrices[:, :9].view(batch_size, 3, 3)
        per_joint_rotations = pose_rotation_matrices[:, 9:]
        pose_blend_coeffs = subtract_flat_identity(per_joint_rotations, self.hands)

        # Compute shape-dependent vertex offsets: v_shaped = v_template + shapedirs * betas
        if betas is None:
            vertices_shaped = self.th_v_template + torch.matmul(
                self.th_shapedirs, self.th_betas.transpose(1, 0)).permute(2, 0, 1)
            joints_rest = torch.matmul(self.th_J_regressor, vertices_shaped).repeat(
                batch_size, 1, 1)
        else:
            vertices_shaped = self.th_v_template + torch.matmul(
                self.th_shapedirs, betas.transpose(1, 0)).permute(2, 0, 1)
            joints_rest = torch.matmul(self.th_J_regressor, vertices_shaped)

        # Compute pose-dependent vertex offsets: v_posed = v_shaped + posedirs * pose_blend_coeffs
        vertices_posed = vertices_shaped + torch.matmul(
            self.th_posedirs, pose_blend_coeffs.transpose(0, 1)).permute(2, 0, 1)
        vertices_tpose = vertices_posed

        # Build kinematic chain: compute global transformation for each joint
        joint_transforms = []
        root_joint = joints_rest[:, 0, :].contiguous().view(batch_size, 3, 1)
        joint_transforms.append(
            pad_homogeneous_row(torch.cat([root_rotation, root_joint], 2))
        )

        for i in range(self.num_joints - 1):
            joint_idx = int(i + 1)
            joint_rotation = per_joint_rotations[:, (joint_idx - 1) * 9:joint_idx * 9].contiguous().view(batch_size, 3, 3)
            joint_position = joints_rest[:, joint_idx, :].contiguous().view(batch_size, 3, 1)
            parent_idx = to_list(self.kintree_parents)[joint_idx]
            parent_position = joints_rest[:, parent_idx, :].contiguous().view(batch_size, 3, 1)
            relative_transform = pad_homogeneous_row(
                torch.cat([joint_rotation, joint_position - parent_position], 2)
            )
            joint_transforms.append(torch.matmul(joint_transforms[parent_idx], relative_transform))
        global_transforms = joint_transforms

        # Compute skinning transforms by removing rest-pose joint positions
        skinning_transforms = root_joint.new_zeros((batch_size, 4, 4, self.num_joints))
        for i in range(self.num_joints):
            zero_pad = joints_rest.new_zeros(1)
            joint_homogeneous = torch.cat([
                joints_rest[:, i],
                zero_pad.view(1, 1).repeat(batch_size, 1)
            ], 1)
            rest_offset = torch.bmm(joint_transforms[i], joint_homogeneous.unsqueeze(2))
            skinning_transforms[:, :, :, i] = joint_transforms[i] - pad_translation_columns(rest_offset)

        # Blend skinning weights with per-joint transforms to get per-vertex transforms
        vertex_transforms = torch.matmul(skinning_transforms, self.th_weights.transpose(0, 1))

        # Apply vertex transforms to rest-pose vertices (in homogeneous coordinates)
        rest_shape_homogeneous = torch.cat([
            vertices_posed.transpose(2, 1),
            vertex_transforms.new_ones((batch_size, 1, vertices_posed.shape[1])),
        ], 1)

        vertices = (vertex_transforms * rest_shape_homogeneous.unsqueeze(1)).sum(2).transpose(2, 1)
        vertices = vertices[:, :, :3]
        joints_transformed = torch.stack(global_transforms, dim=1)[:, :, :3, 3]

        # Apply scale
        vertices = vertices * scale
        joints_transformed = joints_transformed * scale

        # Apply translation
        joints_transformed = joints_transformed + transl.unsqueeze(1)
        vertices = vertices + transl.unsqueeze(1)

        if ret_glb_rot:
            global_rotations = torch.stack(global_transforms, dim=1)[:, :, :3, :3]
            return vertices, joints_transformed, vertices_posed, vertices_tpose, global_rotations
        # Vertices and joints in meters
        return vertices, joints_transformed, vertices_posed, vertices_tpose

    def get_root_joint(self, th_pose_axisang, th_betas, th_trans):
        """
        compute the position of root joint
        Args:
            th_pose_axisang: (B, 72)
            th_betas: (B, num_betas)
            th_trans: (B, 3), global translation

        Returns: (B, 1, 3), the location of root joint

        """
        batch_size = th_pose_axisang.shape[0]

        # Convert axis-angle representation to flattened rotation matrices
        pose_rotation_matrices = axis_angle_to_rotation_matrices(th_pose_axisang)

        # Separate root rotation from per-joint rotations
        root_rotation = pose_rotation_matrices[:, :9].view(batch_size, 3, 3)
        per_joint_rotations = pose_rotation_matrices[:, 9:]
        pose_blend_coeffs = subtract_flat_identity(per_joint_rotations, self.hands)

        # Compute shape-dependent vertices and regress rest-pose joints
        vertices_shaped = self.th_v_template + torch.matmul(
            self.th_shapedirs, th_betas.transpose(1, 0)).permute(2, 0, 1)
        joints_rest = torch.matmul(self.th_J_regressor, vertices_shaped)

        # Build root joint transform
        root_joint = joints_rest[:, 0, :].contiguous().view(batch_size, 3, 1)
        root_transform = pad_homogeneous_row(torch.cat([root_rotation, root_joint], 2))

        # Extract root joint position and apply translation
        joints_transformed = torch.stack([root_transform], dim=1)[:, :, :3, 3]
        joints_transformed = joints_transformed + th_trans.unsqueeze(1)
        return joints_transformed

    def get_joints(self, th_pose_axisang, th_betas, th_trans, axis2rot=True, ret_rots=False):
        "regress joints only"
        batch_size = th_pose_axisang.shape[0]

        # Convert axis-angle representation to flattened rotation matrices
        if axis2rot:
            pose_rotation_matrices = axis_angle_to_rotation_matrices(th_pose_axisang)
        else:
            pose_rotation_matrices = th_pose_axisang  # already (B, 9*(num_joints + 1))

        # Separate root rotation from per-joint rotations
        root_rotation = pose_rotation_matrices[:, :9].view(batch_size, 3, 3)
        per_joint_rotations = pose_rotation_matrices[:, 9:]
        pose_blend_coeffs = subtract_flat_identity(per_joint_rotations, self.hands)

        # Compute shape-dependent vertices and regress rest-pose joints
        if th_betas is None:
            vertices_shaped = self.th_v_template + torch.matmul(
                self.th_shapedirs, self.th_betas.transpose(1, 0)).permute(2, 0, 1)
            joints_rest = torch.matmul(self.th_J_regressor, vertices_shaped).repeat(
                batch_size, 1, 1)
        else:
            vertices_shaped = self.th_v_template + torch.matmul(
                self.th_shapedirs, th_betas.transpose(1, 0)).permute(2, 0, 1)
            joints_rest = torch.matmul(self.th_J_regressor, vertices_shaped)

        # Build kinematic chain: compute global transformation for each joint
        joint_transforms = []
        root_joint = joints_rest[:, 0, :].contiguous().view(batch_size, 3, 1)
        joint_transforms.append(
            pad_homogeneous_row(torch.cat([root_rotation, root_joint], 2))
        )

        for i in range(self.num_joints - 1):
            joint_idx = int(i + 1)
            joint_rotation = per_joint_rotations[:, (joint_idx - 1) * 9:joint_idx * 9].contiguous().view(batch_size, 3, 3)
            joint_position = joints_rest[:, joint_idx, :].contiguous().view(batch_size, 3, 1)
            parent_idx = to_list(self.kintree_parents)[joint_idx]
            parent_position = joints_rest[:, parent_idx, :].contiguous().view(batch_size, 3, 1)
            relative_transform = pad_homogeneous_row(
                torch.cat([joint_rotation, joint_position - parent_position], 2)
            )
            joint_transforms.append(torch.matmul(joint_transforms[parent_idx], relative_transform))
        global_transforms = joint_transforms

        # Extract joint positions from global transforms
        all_transforms = torch.stack(global_transforms, dim=1)
        joints_transformed = all_transforms[:, :, :3, 3] + th_trans.unsqueeze(1)
        if ret_rots:
            return joints_transformed, all_transforms[:, :, :3, :3]
        return joints_transformed
