# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops

from tools.geometry_utils import aa_to_rotmat, rot6d_to_rotmat
from ..modules.pose_transformer import TransformerDecoder


class SMPLTransformerDecoderHead(nn.Module):
    """ Cross-attention based SMPL Transformer decoder
    default config: hmr2/configs_hydra/experiment/hmr_vit_transformer.yaml
    """

    def __init__(self, num_body_joints=23, joint_rep='6d', transformer_input='zero',
                 transformer_decoder_cfg=None, mean_params_path='assets/data/smpl_mean_params.npz',
                 IEF_ITERS=1):
        super().__init__()
        self.IEF_ITERS = IEF_ITERS
        self.num_body_joints = num_body_joints
        self.joint_rep_type = joint_rep
        self.joint_rep_dim = {'6d': 6, 'aa': 3}[self.joint_rep_type]
        npose = self.joint_rep_dim * (num_body_joints + 1)
        self.npose = npose
        self.input_is_mean_shape = transformer_input == 'mean_shape' # TODO: double check this default setting
        transformer_args = dict(
            num_tokens=1,
            token_dim=(npose + 10 + 3) if self.input_is_mean_shape else 1,
            dim=1024,
        )
        transformer_args = (transformer_args | dict(transformer_decoder_cfg))
        print('Transformer decoder args:', transformer_args)
        self.transformer = TransformerDecoder(
            **transformer_args
        )
        dim=transformer_args['dim']
        self.decpose = nn.Linear(dim, npose)
        self.decshape = nn.Linear(dim, 10)
        self.deccam = nn.Linear(dim, 3)

        mean_params = np.load(mean_params_path)
        init_body_pose = torch.from_numpy(mean_params['pose'].astype(np.float32)).unsqueeze(0)
        init_betas = torch.from_numpy(mean_params['shape'].astype('float32')).unsqueeze(0)
        init_cam = torch.zeros(1, 3)
        self.register_buffer('init_body_pose', init_body_pose)
        self.register_buffer('init_betas', init_betas)
        self.register_buffer('init_cam', init_cam) # (1, D)

    def forward(self, x, **kwargs):
        # assume input x has shape (B, HW, C)
        batch_size = x.shape[0]
        # vit pretrained backbone is channel-first. Change to token-first

        init_body_pose = kwargs.get('init_pose', self.init_body_pose.expand(batch_size, -1))
        init_betas = kwargs.get('init_betas', self.init_betas.expand(batch_size, -1))
        init_cam = kwargs.get('init_trans', self.init_cam.expand(batch_size, -1))

        # TODO: Convert init_body_pose to aa rep if needed
        if self.joint_rep_type == 'aa':
            raise NotImplementedError

        pred_body_pose = init_body_pose
        pred_betas = init_betas
        pred_cam = init_cam
        pred_body_pose_list = []
        pred_betas_list = []
        pred_cam_list = []
        for i in range(self.IEF_ITERS): # default is 1
            # Input token to transformer is zero token
            if self.input_is_mean_shape:
                token = torch.cat([pred_body_pose, pred_betas, pred_cam], dim=1)[:,None,:]
            else:
                token = torch.zeros(batch_size, 1, 1).to(x.device) # default setting

            # Pass through transformer
            token_out = self.transformer(token, context=x) # there is an MLP inside the transformer to project it to correct dim
            token_out = token_out.squeeze(1) # (B, C)

            # Readout from token_out
            pred_body_pose = self.decpose(token_out) + pred_body_pose # simply MLP layers
            pred_betas = self.decshape(token_out) + pred_betas
            pred_cam = self.deccam(token_out) + pred_cam
            pred_body_pose_list.append(pred_body_pose)
            pred_betas_list.append(pred_betas)
            pred_cam_list.append(pred_cam)

        # Convert self.joint_rep_type -> rotmat
        joint_conversion_fn = {
            '6d': rot6d_to_rotmat,
            'aa': lambda x: aa_to_rotmat(x.view(-1, 3).contiguous())
        }[self.joint_rep_type]

        pred_body_pose = joint_conversion_fn(pred_body_pose).view(batch_size, self.num_body_joints+1, 3, 3)

        pred_smpl_params = { #'global_orient': pred_body_pose[:, [0]],
                            'pose': pred_body_pose,
                            'betas': pred_betas}
        return pred_smpl_params, pred_cam #, pred_smpl_params_list
