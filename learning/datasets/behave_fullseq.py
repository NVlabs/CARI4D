# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import json
import os
from pathlib import Path
import os.path as osp
import torch
from torch.utils.data import Dataset
from lib_smpl import get_smpl
import numpy as np
import joblib
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import tools.geometry_utils as geom_utils
from copy import deepcopy


from behave_data.const import get_test_view_id, _sub_gender


class BehaveFullSeqTestDataset(Dataset):
    def __init__(self, cfg, seqs, split='val', test_kids=1):
        super().__init__()
        self.cfg = cfg
        self.seqs = seqs
        self.split = split
        self.out_dim_contact = cfg.denoiser_cfg.out_dim_contact 

        # get the nlf, fp, and packed path 
        nlf_root = cfg.nlf_root
        fp_root = cfg.fp_root
        packed_root = cfg.packed_root
        contacts_root = cfg.contacts_root

        # load the nlf, fp, and packed data 
        nlf_files, fp_files, packed_files, seqs_valid = [], [], [], []
        contact_files = []
        for seq in seqs:
            # check if individual files exist
            if not osp.isfile(f'{nlf_root}/{seq}_params.pkl'):
                print(f'no nlf file for {seq}...skipping')
                continue
            if not osp.isfile(f'{fp_root}/{seq}_all.pkl'):
                print(f'no fp file for {seq}...skipping')
                continue
            if not osp.isfile(f'{packed_root}/{seq}_GT-packed.pkl'):
                print(f'no packed file for {seq}...skipping')
                continue
            if not osp.isfile(f'{contacts_root}/{seq}_contact-jts.npz'):
                print(f'no contact file for {seq}...skipping')
                continue
            nlf_files.append(f'{nlf_root}/{seq}_params.pkl')
            fp_files.append(f'{fp_root}/{seq}_all.pkl')
            packed_files.append(f'{packed_root}/{seq}_GT-packed.pkl')
            seqs_valid.append(seq)
            if contacts_root is not None:
                contact_files.append(f'{contacts_root}/{seq}_contact-jts.npz')
        # load camera transforms
        self.w2cs = joblib.load('/home/xianghuix/datasets/behave/calibs/w2cs_all.pkl')
        # decide which kinect to be used for each view 
        if isinstance(test_kids, int):
            self.kinect_inds = [1]*len(nlf_files)
        else:
            self.kinect_inds = test_kids

        # store the file paths
        self.nlf_files = nlf_files
        self.fp_files = fp_files
        self.packed_files = packed_files
        self.seqs_valid = seqs_valid
        self.contact_files = contact_files
        self.smpl_male = get_smpl('male', hands=True)
        self.smpl_female = get_smpl('female', hands=True)
        print(f'Loaded {len(self.nlf_files)} sequences')

    def __len__(self):
        return len(self.nlf_files)
    
    def __getitem__(self, idx):
        "load NLF, FP, and packed data from the file paths, and pack them into a batch dict"
        nlf_data = joblib.load(self.nlf_files[idx])
        fp_data = joblib.load(self.fp_files[idx])
        packed_data = joblib.load(self.packed_files[idx])
        seq_name = self.seqs_valid[idx]
        date = seq_name.split('_')[0]

        # pack them into a batch dict
        batch = {}
        kid = self.kinect_inds[idx]
        # use selected view id 
        if self.cfg.job == 'test' and self.cfg.use_sel_view:
            view_id = get_test_view_id(seq_name)
            kid = view_id if view_id is not None else 1
            print(f'Using test view id: {view_id} for {seq_name}')

        w2c_k = self.w2cs[date][kid]

        # load the data
        nlf_data = joblib.load(self.nlf_files[idx])
        fp_data = joblib.load(self.fp_files[idx])
        packed_data = joblib.load(self.packed_files[idx])
                
        # get the NLF poses 
        nlf_poses = nlf_data['poses'][:, kid] 
        nlf_betas = nlf_data['betas'][:, kid]
        nlf_transls = nlf_data['transls'][:, kid]
        betas_avg_nlf = np.mean(nlf_betas, axis=0)[None].repeat(len(nlf_poses), axis=0)
        trans_ref = nlf_transls[0].copy() 

        # get FP poses 
        fp_poses = fp_data['fp_poses'][:, kid]
        
        # compute rot6d for NLF and FP 
        nlf_rot6d = geom_utils.axis_to_rot6D(torch.from_numpy(nlf_poses).float().reshape(-1, 3)).reshape(len(nlf_poses), -1).numpy()
        fp_rot6d = geom_utils.rotmat_to_6d(torch.from_numpy(fp_poses[:, :3, :3]).float()).reshape(len(fp_poses), 6).numpy()
        nlf_transl_rel = nlf_transls.copy() - trans_ref
        fp_transl_rel = fp_poses[:, :3, 3] - trans_ref
        noisy_hum = np.concatenate([nlf_rot6d[:, :144], betas_avg_nlf, nlf_transl_rel], axis=-1)
        noisy_obj = np.concatenate([fp_rot6d, fp_transl_rel], axis=-1)
        L_pred = len(nlf_poses)

        # compute GT human and object as well 
        gender = _sub_gender[seq_name.split('_')[1]]
        smpl_pose_gt = packed_data['poses'][:L_pred] # use full pose
        smpl_transl_gt = packed_data['trans'][:L_pred]
        smpl_betas_gt = packed_data['betas'][:L_pred]    
        print(nlf_poses.shape, smpl_pose_gt.shape, smpl_transl_gt.shape, smpl_betas_gt.shape)
        
        # convert SMPL poses to local camera coordinate 
        global_rot = R.from_rotvec(smpl_pose_gt[:, :3]).as_matrix()
        new_rot = np.stack([np.matmul(w2c_k[:3, :3], r) for r in global_rot], 0)
        smpl_pose_gt[:, :3] = R.from_matrix(new_rot).as_rotvec()
        smpl_joints_root = packed_data['joints_smpl'][:L_pred, 0]
        joints_cent = smpl_joints_root - smpl_transl_gt
        smpl_transl_gt = np.matmul(smpl_transl_gt, w2c_k[:3, :3].T) + w2c_k[:3, 3] + np.matmul(joints_cent, w2c_k[:3, :3].T) - joints_cent

        smpl_post_gt_rot6d = geom_utils.axis_to_rot6D(torch.from_numpy(smpl_pose_gt).float().reshape(-1, 3)).reshape(len(smpl_pose_gt), -1).numpy()
        smpl_transl_gt_rel = smpl_transl_gt - trans_ref
        gt_hum = np.concatenate([smpl_post_gt_rot6d[:, :144], smpl_betas_gt, smpl_transl_gt_rel], axis=-1)
        # GT obj
        obj_rot = R.from_rotvec(packed_data['obj_angles'][:L_pred]).as_matrix()
        obj_pose_gt = np.eye(4)[None].repeat(len(obj_rot), 0)
        obj_pose_gt[:, :3, :3] = obj_rot
        obj_pose_gt[:, :3, 3] = packed_data['obj_trans'].copy()[:L_pred]  

        # compute local object pose 
        obj_pose_gt_local = [np.matmul(w2c_k, pose) for pose in obj_pose_gt]
        obj_pose_gt_local = np.stack(obj_pose_gt_local, 0)

        gt_rot6d = geom_utils.rotmat_to_6d(torch.from_numpy(obj_pose_gt_local[:, :3, :3]).float()).reshape(len(obj_pose_gt_local), 6).numpy()
        gt_transl = obj_pose_gt_local[:, :3, 3].copy() - trans_ref
        gt_obj = np.concatenate([gt_rot6d, gt_transl], axis=-1)

        frames = [osp.join(seq_name, x) for x in packed_data['frames'][:L_pred]]

        # convert all to float32
        pre_dict = {
            # network inputs
            'noisy_hum': noisy_hum.astype(np.float32),
            'noisy_obj': noisy_obj.astype(np.float32),
            # for computing loss 
            'gt_hum': gt_hum.astype(np.float32),
            'gt_obj': gt_obj.astype(np.float32),
            'smpl_transl_gt': smpl_transl_gt_rel.astype(np.float32),
            # for saveing results 
            'image_files': frames,
            'trans_ref': trans_ref.astype(np.float32),
            'is_male': np.array([gender == 'male'] * len(frames)),
            # for evaluation logging
            'pose_perturbed': fp_poses.astype(np.float32),
            'pose_gt': obj_pose_gt_local.astype(np.float32),

            # additional info 
            'nlf_poses': nlf_poses.astype(np.float32),
            'betas_nlf': betas_avg_nlf.astype(np.float32),
            'nlf_transl': nlf_transl_rel.astype(np.float32),
            'smpl_poses_gt': smpl_pose_gt.astype(np.float32),
            'betas_gt': smpl_betas_gt.astype(np.float32),
        }

        if self.out_dim_contact > 0:
            # add distance from human joints to object mesh  
            contact_data = np.load(self.contact_files[idx])
            contacts_in = contact_data['dists_noisy']
            contacts_gt = contact_data['dists_gt']
            pre_dict['contact_dist_in'] = contacts_in.astype(np.float32)[kid, :L_pred, :self.out_dim_contact]
            pre_dict['contact_dist_gt'] = contacts_gt.astype(np.float32)[:L_pred, :self.out_dim_contact]

        return pre_dict


        