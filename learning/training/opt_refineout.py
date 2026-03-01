# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import sys, os

import imageio
sys.path.append(os.getcwd())
from tools import img_utils
from learning.training.training_config import RefineOutOptimConfig
from omegaconf import OmegaConf
import torch, time 
import numpy as np
import trimesh
import wandb
import smplx
from tqdm import tqdm
from pytorch3d.transforms.rotation_conversions import axis_angle_to_matrix, matrix_to_axis_angle
from behave_data.utils import load_template
from behave_data.const import _sub_gender, BEHAVE_ROOT
import os.path as osp
import cv2
from pytorch3d.renderer import look_at_view_transform
import torch.nn.functional as F
from VolumetricSMPL import attach_volume
from lib_smpl.body_landmark import BodyLandmarks
from lib_smpl.th_hand_prior import mean_hand_pose, SMPL_ASSETS_ROOT
from lib_smpl.const import SMPL_MODEL_ROOT
from behave_data.behave_video import BaseBehaveVideoData, VideoController
from transformers import get_scheduler
from behave_data.const import get_test_view_id
from behave_data.utils import get_intrinsics_unified
import h5py
from pytorch3d.ops.knn import knn_points
import Utils
import nvdiffrast.torch as dr
from glob import glob
from argparse import Namespace 
from learning.training.training_utils import TrainState


HAND_JOINT_INDICES = [22, 23+15]

class RefineOutOptimizer(BaseBehaveVideoData):
    def __init__(self, cfg: RefineOutOptimConfig):
        seq_name = osp.basename(cfg.pth_file).split('.')[0]
        view_id = get_test_view_id(seq_name)
        view_id = view_id if view_id is not None else 1
        if cfg.wild_video:
            view_id = 0

        self.view_id = view_id
        self.seq_name = seq_name

        save_name_old = osp.basename(osp.dirname(cfg.pth_file))
        self.save_name_old = save_name_old 

        args_default = Namespace(
            video=osp.join(cfg.video_root, f'{seq_name}.{view_id}.color.mp4'),
            outpath=cfg.outpath,
            fps=30,
            tstart=3.0,
            tend=None,
            redo=False,
            kid=view_id,
            start=0,
            end=-1,
            nodepth=True,
            cameras=[view_id],
            wild_video=cfg.wild_video,
            data_source=cfg.data_source,
            viz_path=None,
            debug=0,
            index=None,
            chunk_start=None,
            chunk_end=None,
            masks_root=cfg.masks_root,
        )

        super().__init__(args_default)

        self.cfg = cfg
        self.device = 'cuda'
        sind = self.save_name_old.find('hy3d')
        sind = len(self.save_name_old) if sind == -1 else sind
        self.exp_dir = f'{cfg.outpath}/{self.save_name_old[:sind]}-hy3d3-{self.cfg.save_name}'
        print(f'Exp dir: {self.exp_dir}')
        os.makedirs(self.exp_dir, exist_ok=True)
        self.landmark = BodyLandmarks(SMPL_ASSETS_ROOT)
        mean_hand = mean_hand_pose(SMPL_ASSETS_ROOT)
        self.mean_lhand = torch.from_numpy(mean_hand[:45]).float().to(self.device)
        self.mean_rhand = torch.from_numpy(mean_hand[45:]).float().to(self.device) 

        # init rasterizer 
        self.glctx = dr.RasterizeCudaContext()
        self.scale_ratio = 4

        # setup wandb 
        if not self.cfg.no_wandb:
            wandb_dir = f'{self.exp_dir}/{seq_name}'
            run_folders = sorted(glob(osp.join(wandb_dir, 'wandb/run-*')))
            if len(run_folders) == 0:
                print(f'No wandb run found, starting from scratch!')
                wandb.init(project='e2etracker-opt', name=f'{self.save_name_old[:sind]}hy3d3-{cfg.save_name}+{seq_name}', job_type='test',
                        config=OmegaConf.to_container(self.cfg),
                        dir=wandb_dir)
            else:
                print(f'Found wandb run {run_folders[-1]}')
                print('resume wandb from', run_folders[-1])
                wandb.init(project='e2etracker-opt', name=f'{self.save_name_old[:sind]}hy3d3-{cfg.save_name}+{seq_name}', job_type='test',
                        config=OmegaConf.to_container(self.cfg),
                        id=osp.basename(run_folders[-1]).split('-')[-1],
                        dir=wandb_dir, resume='must')


    def optimize(self):
        "load the predictions from pth file, and optimize"
        pth_data = torch.load(self.cfg.pth_file, map_location='cpu') # this should have only one view 
        if self.cfg.use_input:
            pth_data['pr'] = pth_data['in']
            print("Optimizing input!")
        # apply oneeuro filter to results first
        if self.cfg.oneeuro_type == 'smplz':
            # smooth smplz
            from tools.filter_oneeuro import filter_3axis 
            smpl_trans = filter_3axis(pth_data['pr']['smpl_t'].cpu().numpy())
            pth_data['pr']['smpl_t'][:, 2] = torch.from_numpy(smpl_trans[:, 2]).to(pth_data['pr']['smpl_t'].device).float()
            print("Smoothed smplz")
        
        # TODO: load data from checkpoint 
        ckpt_files = sorted(glob(osp.join(self.exp_dir, f'{self.seq_name}*.pth')))
        ckpt_file = None 
        if len(ckpt_files) > 0:
            ckpt_file = ckpt_files[-1]
            ckpt_data = torch.load(ckpt_file, map_location='cpu', weights_only=False)
            pr = ckpt_data['pr']
            smpl_pose = pr['smpl_pose'].clone() # 72dim 
            smpl_trans = pr['smpl_t'].clone()
            betas = pr['betas'].clone() 
            frames_pr = [x.split('/')[-1] for x in pr['frames']]
            train_state = pr['train_state']
            print(f'Loaded checkpoint {ckpt_file}')
        else:
            train_state = TrainState()
            print(f'No checkpoint found, starting from scratch')
            pr = pth_data['pr'] 
            # copy pr to in 
            pth_data['in'] = pr 
            smpl_pose = pr['smpl_pose'].clone() # 72dim 
            smpl_trans = pr['smpl_t'].clone()
            betas = pr['betas'].clone() 
        frames_pr = [x.split('/')[-1] for x in pr['frames']]
        if train_state.step >= self.cfg.num_steps:
            print(f'Step {train_state.step} is greater than or equal to num_steps {self.cfg.num_steps}, skipping')
            return

        # separate smpl pose into global and local body poses 
        smpl_global_pose = smpl_pose[:, :3].detach().clone()
        smpl_body_pose = smpl_pose[:, 3:66].detach().clone()

        # get subject gender
        seq_name = osp.basename(self.cfg.pth_file).split('.')[0]
        gender = _sub_gender[seq_name.split('_')[1]]
        smplh_model = smplx.create(
            model_path=SMPL_MODEL_ROOT,
            model_type='smplh',
            gender=gender,
            use_pca=False,
            batch_size=self.cfg.batch_size,
            flat_hand_mean=True # the given hands are directly in the axis angle format
        ).to(self.device)
        attach_volume(smplh_model, device=self.device) # allows penetration computation 

        # load the object mesh template 
        obj_name = seq_name.split('_')[2]
        if not self.cfg.wild_video:
            crop_center = np.zeros(3) # assume the object is already centered
            
            # load hy3d mesh 
            use_hy3d = True
            if use_hy3d:
                from behave_data.const import get_hy3d_mesh_file
                hy3d_file = get_hy3d_mesh_file(seq_name, meshes_root=self.cfg.hy3d_meshes_root)
                obj_template = trimesh.load(hy3d_file, process=False)
            else:
                obj_template = load_template(obj_name, cent=False, dataset_path=BEHAVE_ROOT)
        else:
            from behave_data.const import get_hy3d_mesh_file
            hy3d_file = get_hy3d_mesh_file(seq_name, meshes_root=self.cfg.hy3d_meshes_root)
            obj_template = trimesh.load(hy3d_file, process=False)
            crop_center = np.mean(obj_template.vertices, 0)
        obj_template.vertices = obj_template.vertices - crop_center
        
        obj_pts = torch.from_numpy(obj_template.sample(1000)).float().to(self.device)
        obj_rot = pr['pose_abs'][:, :3, :3]
        obj_trans = pr['pose_abs'][:, :3, 3].clone()  
        obj_axis = matrix_to_axis_angle(obj_rot).to(self.device)
        obj_trans = obj_trans.to(self.device) 
        
        # 2d joint projection loss
        joints25_mask = np.ones((25,1))
        joints25_mask[[0, 15, 16, 17, 18, 19, 20, 21, 23, 24]] = 0 # mask out head, toes, these can lead to artifacts
        import joblib

        opt_dict = {
            'obj_axis': obj_axis if not self.cfg.opt_rot else obj_axis.requires_grad_(True), 
            'obj_trans': obj_trans if not self.cfg.opt_trans else obj_trans.requires_grad_(True),
            'obj_axis_orig': obj_axis.clone(),
            'obj_trans_orig': pth_data['pr']['pose_abs'][:, :3, 3].clone().to(self.device),
            'smpl_trans_orig': pth_data['pr']['smpl_t'].clone().to(self.device),
            'smpl_pose_orig': pth_data['pr']['smpl_pose'][:, 3:66].clone().to(self.device),

            # human params 
            'smpl_pose_global': smpl_global_pose.to(self.device), # do not optimize global orientation, assume it is already good  if not self.cfg.opt_smpl_pose else smpl_global_pose.requires_grad_(True).to(self.device),
            'smpl_pose_body': smpl_body_pose.to(self.device) if not self.cfg.opt_smpl_pose else smpl_body_pose.to(self.device).requires_grad_(True),
            'smpl_trans': smpl_trans.to(self.device) if not self.cfg.opt_smpl_trans else smpl_trans.to(self.device).requires_grad_(True), # requires_grad_ must be the last step 
            'betas': betas.to(self.device),

            # object points  
            'obj_pts': obj_pts,
            'obj_verts': torch.from_numpy(obj_template.vertices).float().to(self.device),
            'obj_template': obj_template,

            'joints25_mask': torch.from_numpy(joints25_mask).float().to(self.device),
        }

        # construct mesh tensors: uniform color for object and part based SMPL colors 
        vc_obj = torch.tensor([0., 128 / 255.0, 114 / 255.0], device=self.device).repeat(len(obj_template.vertices), 1)
        vc_smpl = torch.from_numpy(trimesh.load('data/assets/smpl-meshes/parts_surrel.obj').visual.vertex_colors).float().to(self.device)[:, :3]/255.
        # also part texture: assets/meshlab-corr-order 
        mesh_tensors = {
            'vertex_color': torch.cat([vc_smpl, vc_obj], 0),
            'faces': torch.from_numpy(np.concatenate([smplh_model.faces, obj_template.faces+ 6890], 0)).int().to(self.device),
            'pos': None,
        }
        opt_dict['mesh_tensors'] = mesh_tensors
        opt_dict['mesh_tensors_obj'] = {
            "faces": torch.from_numpy(obj_template.faces).int().to(self.device),
            "pos": torch.from_numpy(obj_template.vertices).float().to(self.device),
            # pure white color 
            'vertex_color': torch.tensor([1, 1, 1.], device=self.device).repeat(len(obj_template.vertices), 1),
        } # no need to have texture, it will be depth only rendering 

        # additional data loading 
        # get view id for this seq 
        # 2d joints and contacts (GT)
        view_id = self.view_id
        pack_file = f'{self.cfg.packed_root}/{seq_name}_GT-packed.pkl'
        pack_data = joblib.load(pack_file)
        frames_gt = pack_data['frames'] 
        # get indices at gt data
        frame_inds = np.array([frames_gt.index(x) for x in frames_pr])
        if isinstance(pack_data['joints2d'], list):
            pack_data['joints2d'] = np.stack(pack_data['joints2d'])
        joints_2d = pack_data['joints2d'][frame_inds, view_id] # (L, 25, 3)
        op_thres = self.cfg.op_thres

        # use predicted contacts 
        if 'contact_logits' in pth_data['pr']:
            contact_logits = pth_data['pr']['contact_logits']
            if self.cfg.contact_pred_type == 'binary':
                contact_mask = contact_logits.cpu().numpy() > 0. # simply binary mask 
            elif self.cfg.contact_pred_type == 'distance':
                contact_mask = contact_logits.cpu().numpy() < self.cfg.contact_mask_thres
            else:
                raise ValueError(f'Invalid contact prediction type: {self.cfg.contact_pred_type}')
            print(f"Using predicted contacts, prediction type: {self.cfg.contact_pred_type}")
        else:
            assert self.cfg.use_gt, 'no contact logits found and use_gt is False'
            contact_mask = pack_data['dists_h2o'][0, frame_inds][:, np.array(HAND_JOINT_INDICES)] < self.cfg.contact_mask_thres # hand contact mask
            print("Using GT contacts")
        assert len(contact_mask) == len(obj_axis) == len(joints_2d), f'lengths do not match: {len(contact_mask)}, {len(obj_axis)}, {len(joints_2d)}'
        opt_dict['contact_mask'] = torch.from_numpy(contact_mask).float().to(self.device)
        opt_dict['joints_2d'] = torch.from_numpy(joints_2d).float().to(self.device)

        # 2D masks and occlusion masks 
        if not self.cfg.wild_video:
            K_full = get_intrinsics_unified(self.cfg.data_source, seq_name, view_id, self.cfg.wild_video)
        else:
            # simply use the camera
            K_full = self.camera_K.copy() 
            # assert self.scale_ratio == 1.0, f'scale ratio should be 1 instead of {self.scale_ratio} for wild video'
            print(f'Using camera intrinsics: {K_full}')
        K_full_th = torch.from_numpy(K_full).float().to(self.device)
        rend_size = 256
        focal = np.array([K_full[0, 0], K_full[1, 1]])
        principal_point = np.array([K_full[0, 2], K_full[1, 2]])
        tar_mask = h5py.File(self.tar_path.replace('_masks_k0.h5', f'_masks_k{view_id}.h5'), 'r')
        keep_masks, image_refs, K_rois = [], [], [] # keep_mask is occlusion aware, image_ref is the object reference mask

        debug_len = self.cfg.batch_size *2 if self.cfg.debug else len(frames_pr)
        for frame in tqdm(frames_pr[:debug_len], desc=f'loading masks {seq_name}'):
            frame_time = frame.split('/')[-1]
            mname_h = f'{self.video_prefix}/{frame_time}-k{view_id}.person_mask.png'
            mname_o = f'{self.video_prefix}/{frame_time}-k{view_id}.obj_rend_mask.png'
            mask_h = tar_mask[mname_h][:].astype(np.uint8) * 255
            mask_o = tar_mask[mname_o][:].astype(np.uint8) * 255
            # now crop, compute Kroi, and compute image ref 
            masks = [mask_h, mask_o] if np.sum(mask_o>127) < 200 else [mask_o] 
            bmin, bmax = img_utils.masks2bbox(masks)
            # compute kroi 
            crop_center = (bmax + bmin) / 2
            radius = np.max(bmax - bmin) * 1.1/2
            top_left = crop_center - radius
            bottom_right = crop_center + radius
            K_roi = self.Kroi_from_corners(bottom_right, top_left, focal=focal, principal_point=principal_point, rend_size=rend_size)
            K_rois.append(K_roi)

            # crop images and resize to render size 
            mask_h = cv2.resize(img_utils.crop(mask_h, crop_center, radius*2), (rend_size, rend_size))
            mask_o = cv2.resize(img_utils.crop(mask_o, crop_center, radius*2), (rend_size, rend_size))
            person_mask = (mask_h > 127).astype(np.float32)
            obj_mask = (mask_o > 127).astype(np.float32)
            fore_mask = obj_mask > 0.5
            ps_mask = person_mask > 0.5
            mask_inv = - ps_mask.astype(np.float32)
            mask_inv[fore_mask] = 1.  # convention: 1--foreground, -1: occlusion ignore
            keep_masks.append(mask_inv>=0)
            image_refs.append(obj_mask>0)
        keep_masks = torch.from_numpy(np.stack(keep_masks, 0)).float().cpu()
        image_refs = torch.from_numpy(np.stack(image_refs, 0)).float().cpu() # do not keep them on gpu, too expensive 
        K_rois = np.stack(K_rois, 0)
        opt_dict['keep_masks'] = keep_masks
        opt_dict['image_refs'] = image_refs
        opt_dict['K_rois'] = K_rois

        symmetric_objects = ['boxlong', 'boxlarge', 'boxmedium', 'boxsmall', 'boxtiny', 'yogamat', 
                    # 'stool',  'trashbin', 'plasticcontainer', 'tablesquare', 'suitcase', 'backpack', 
                     'obj02', 'obj04', 'obj05', 'obj06']
        is_symmetric = seq_name.split('_')[2] in symmetric_objects # for these the object temporal smoothness is different 

        # optimization parameters
        opt_params = []
        if self.cfg.opt_trans:
            opt_params.append({'params': opt_dict['obj_trans'], 'lr': self.cfg.lr})
        if self.cfg.opt_rot:
            opt_params.append({'params': opt_dict['obj_axis'], 'lr': self.cfg.lr})
        if self.cfg.opt_smpl_pose:
            opt_params.append({'params': opt_dict['smpl_pose_body'], 'lr': self.cfg.lr})
        if self.cfg.opt_smpl_trans:
            opt_params.append({"params": opt_dict['smpl_trans'], "lr": self.cfg.lr })
        if self.cfg.opt_betas:
            opt_params.append({"params": opt_dict['betas'], "lr": self.cfg.lr })
        optimizer = torch.optim.Adam(opt_params, lr=self.cfg.lr)
        scheduler = get_scheduler(optimizer=optimizer, name='cosine',
                                              num_warmup_steps=self.cfg.num_steps//10,
                                              num_training_steps=int(self.cfg.num_steps * 1.5))
        if ckpt_file is not None:
            optimizer.load_state_dict(pr['optimizer'])
            scheduler.load_state_dict(pr['scheduler'])
            train_state = pr['train_state']
            print(f'Loaded optimizer and scheduler from checkpoint {ckpt_file}')
        seq_len, batch_size = len(obj_axis), self.cfg.batch_size
        seq_len = debug_len if self.cfg.debug else seq_len
        batch_size = seq_len if batch_size >= seq_len else batch_size
        self.cfg.batch_size = batch_size
        print(f'Using batch size: {batch_size}')

        # optimization loop 
        for step in tqdm(range(train_state.step, self.cfg.num_steps + 1)):
            # sample one batch of data 
            rid = np.random.randint(0, seq_len - batch_size+1)
            start_ind, end_ind = rid, min(seq_len, rid + batch_size)
            
            # compute the loss 
            loss = 0 

            # smpl model forward: 
            smplh_output = smplh_model(
                betas=opt_dict['betas'][start_ind:end_ind],
                body_pose=opt_dict['smpl_pose_body'][start_ind:end_ind],
                global_orient=opt_dict['smpl_pose_global'][start_ind:end_ind],
                transl=opt_dict['smpl_trans'][start_ind:end_ind],
                left_hand_pose=self.mean_lhand.repeat(batch_size, 1),
                right_hand_pose=self.mean_rhand.repeat(batch_size, 1), # use mean hand poses 
                return_verts=True,
                return_full_pose=True
            )

            # compute object 
            obj_rot = axis_angle_to_matrix(opt_dict['obj_axis'][start_ind:end_ind])
            obj_trans = opt_dict['obj_trans'][start_ind:end_ind]
            obj_pts_batch = obj_pts[None].to(self.device).repeat(batch_size, 1, 1)
            obj_pts_posed = torch.matmul(obj_pts_batch, obj_rot.permute(0, 2, 1)) + obj_trans[:, None]
            obj_verts_posed = torch.matmul(opt_dict['obj_verts'][None].repeat(batch_size, 1, 1), obj_rot.permute(0, 2, 1)) + obj_trans[:, None]

            # 2D projection loss on SMPL joints: get 25 body joints 
            t0 = time.time()
            loss_j2d = torch.tensor(0, device=self.device)
            if self.cfg.w_j2d > 0:
                joints_25_3d = self.landmark.get_body_kpts_batch_torch(smplh_output.vertices)
                joints_25_proj = joints_25_3d @ K_full_th.T
                joints_25_proj_xy = joints_25_proj[:, :, :2] / joints_25_proj[:, :, 2:3]  
                j2d_conf = opt_dict['joints_2d'][start_ind:end_ind, :, 2:3]
                j2d_conf[j2d_conf < op_thres] = 0.0
                j2d_mask = j2d_conf
                loss_j2d = F.mse_loss(joints_25_proj_xy, opt_dict['joints_2d'][start_ind:end_ind, :, :2], reduction='none') * j2d_mask.float() 
                loss_j2d = loss_j2d.mean() * self.cfg.w_j2d
            t1 = time.time()
            
            # contact losses: compute hands to object distances using knn 
            joints_hand = smplh_output.joints[:, HAND_JOINT_INDICES, :] # left and right hand joints 
            closest_dist_in_obj = knn_points(joints_hand, obj_pts_posed, K=1, return_nn=True)
            dists_actual = closest_dist_in_obj.dists.squeeze(-1) # (B, 2) 
            cont_mask = opt_dict['contact_mask'][start_ind:end_ind] 
            loss_contact = (dists_actual*cont_mask).mean() * self.cfg.w_contact # contact point smoothness loss? 
            t2 = time.time()
            
            # object silhouette loss: use nvdiff rasterization to obtain masks 
            ret = Utils.nvdiff_color_depth_render(opt_dict['K_rois'][start_ind:end_ind], self.glctx, opt_dict['mesh_tensors_obj'], (rend_size, rend_size), 
                        obj_verts_posed, depth_only=False) # TODO: replace this with pytorch3d renderer 
            color_obj, depth, xyz_map = ret
            # compute the mask
            mask_obj = color_obj.mean(-1) # pure depth mask will vanish the gradient, use color to obtain pseudo mask 
            image_rend = opt_dict['keep_masks'][start_ind:end_ind].to(self.device) * mask_obj.float() 
            loss_sil = F.mse_loss(image_rend, opt_dict['image_refs'][start_ind:end_ind].to(self.device), reduction='none').sum((1, 2)).mean() * self.cfg.w_sil
            t3 = time.time()

            # penetration loss: something is cached here, making the repeatition runtime error.
            loss_pen = 0
            if self.cfg.w_pen > 0 and step > self.cfg.pen_loss_start * self.cfg.num_steps:
                smplh_model.volume.detach_cache() # avoid repeated backprop
                # TODO: do min-chunk to allow larger batch size overall.
                loss_pen = smplh_model.volume.collision_loss(obj_pts_posed, smplh_output)[0].mean() * self.cfg.w_pen
            t4 = time.time()    

            # temporal smoothness loss: on SMPL vertices and object vertices  
            # the acceleration should be zero 
            verts_h, verts_o = smplh_output.vertices, obj_pts_posed
            acc_h = smplh_output.joints[:-2] -2 * smplh_output.joints[1:-1] + smplh_output.joints[2:] # focus on joints, especially hands should be stable 
            if is_symmetric:
                # consider only the translation smoothness
                acc_o = obj_trans[:-2] - 2 * obj_trans[1:-1] + obj_trans[2:]
                velo_o = obj_trans[1:] - obj_trans[:-1]
            else:
                acc_o = verts_o[:-2] - 2 * verts_o[1:-1] + verts_o[2:]
                velo_o = verts_o[1:] - verts_o[:-1]
            loss_temp = (acc_h**2).sum(-1).mean() * self.cfg.w_temp + (acc_o**2).sum(-1).mean()  * self.cfg.w_temp
            loss_velo = 0

            # apply static loss when there is no contact on the object, it should remain static 
            if self.cfg.w_velo > 0:
                kernel = torch.ones(1, 1, 5).to(self.device) # dilate to one neighbourhood
                cont_mask_dilate = F.conv1d(opt_dict['contact_mask'][start_ind:end_ind].mean(-1).view(1, 1, -1), kernel, padding=2)
                static_mask = 1 - (cont_mask_dilate > 0).float() 
                loss_velo = (velo_o**2 * static_mask.view(-1, 1, 1)[1:]).sum(-1).mean()* self.cfg.w_velo 
            t5 = time.time()
            # time taken: j2d: 0.419, contact: 0.002, sil: 0.016, pen: 0.325, temp: 0.001

            # initial object translation loss: the object should be close to the initial position 
            loss_init_ot = 0
            if self.cfg.w_init_ot > 0 and self.cfg.opt_trans:
                loss_init_ot = F.mse_loss(opt_dict['obj_trans_orig'][start_ind:end_ind], opt_dict['obj_trans'][start_ind:end_ind], reduction='none').sum(-1).mean() * self.cfg.w_init_ot
            loss_init_ht = 0
            if self.cfg.w_init_ht > 0 and self.cfg.opt_smpl_trans:
                loss_init_ht = F.mse_loss(opt_dict['smpl_trans_orig'][start_ind:end_ind], opt_dict['smpl_trans'][start_ind:end_ind], reduction='none').sum(-1).mean() * self.cfg.w_init_ht
            t6 = time.time()
            # initial pose loss: the pose should be close to the initial pose 
            loss_init_p = 0 
            if self.cfg.w_pinit > 0:
                loss_init_p = F.mse_loss(opt_dict['smpl_pose_orig'][start_ind:end_ind], opt_dict['smpl_pose_body'][start_ind:end_ind], reduction='none').sum(-1).mean() * self.cfg.w_pinit
            # compute the total loss 
            loss = loss_j2d + loss_contact + loss_sil + loss_pen + loss_temp + loss_velo + loss_init_ot + loss_init_ht + loss_init_p

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            train_state.step += 1

            # log losses to wandb
            if not self.cfg.no_wandb:
                wandb.log({
                    'loss_j2d': loss_j2d.item(),
                    'loss_contact': loss_contact.item(),
                    'loss_sil': loss_sil.item(),
                    'loss_pen': loss_pen,
                    'loss_temp': loss_temp.item(),
                    'loss_velo': loss_velo,
                    'loss_init_ot': loss_init_ot,
                    'loss_init_ht': loss_init_ht,
                    'loss_init_p': loss_init_p,
                    'loss_total': loss.item(),
                    'lr': scheduler.get_last_lr()[0],
                }, step=step)
            # save ckpt
            if step % self.cfg.save_every_n_steps == 0:
                # save ckpt, same format as the pr file, so we can directly reuse the viz_code 
                pose_abs = torch.eye(4).repeat(len(obj_axis), 1, 1)
                pose_abs[:, :3, :3] = axis_angle_to_matrix(opt_dict['obj_axis']).detach().cpu()
                pose_abs[:, :3, 3] = opt_dict['obj_trans'].detach().cpu()
                smpl_pose = torch.cat([opt_dict['smpl_pose_global'], opt_dict['smpl_pose_body'], torch.zeros_like(opt_dict['smpl_trans']), torch.zeros_like(opt_dict['smpl_trans'])], 1)
                ckpt = {
                    "frames": pth_data['pr']['frames'],
                    "pose_abs": pose_abs,
                    "smpl_pose": smpl_pose.detach().cpu(),
                    "smpl_t": opt_dict['smpl_trans'].detach().cpu(),
                    "betas": opt_dict['betas'].detach().cpu(),

                    # optimizer and scheduler if last step, do not save the optimizer state, and scheduler 
                    "optimizer": optimizer.state_dict() if step < self.cfg.num_steps else None,
                    "scheduler": scheduler.state_dict() if step < self.cfg.num_steps else None,
                    'pth_file': self.cfg.pth_file,
                    'train_state': train_state,
                }
                # 
                if 'contact_logits' in pth_data['pr']:
                    ckpt['contact_logits'] = pth_data['pr']['contact_logits']
                pth_data['pr'] = ckpt

                torch.save(pth_data, osp.join(self.exp_dir, f'{seq_name}.pth'))
                print(f'Saved ckpt to {osp.join(self.exp_dir, f"{seq_name}.pth")}')

            
            # visualization
            if step % self.cfg.viz_steps == 0:
                # load original RGB image and render the full seq 
                video_file = osp.join(self.cfg.video_root, seq_name + f'.{view_id}.color.mp4')
                video_ctrl = VideoController(video_file)
                self.side_view_z = None 
                video_file_out = f'{self.exp_dir}/{seq_name}+step{step:06d}.mp4'
                vw = imageio.get_writer(video_file_out,'FFMPEG', fps=30)

                for chunk_ind in tqdm(range(0, debug_len, batch_size)):
                    frames_pr_chunk = frames_pr[chunk_ind:chunk_ind+batch_size]
                    chunk_end = min(chunk_ind + batch_size, len(frames_pr))
                    print(f"Visualizing step {step}, chunk {chunk_ind}->{chunk_end}")
                    batch_size_actual = chunk_end - chunk_ind
                    # get smpl and obj verts, no grad for this 
                    with torch.no_grad():
                        smplh_out = smplh_model(
                            betas=opt_dict['betas'][chunk_ind:chunk_ind+batch_size],
                            body_pose=opt_dict['smpl_pose_body'][chunk_ind:chunk_ind+batch_size],
                            global_orient=opt_dict['smpl_pose_global'][chunk_ind:chunk_ind+batch_size],
                            transl=opt_dict['smpl_trans'][chunk_ind:chunk_ind+batch_size],
                            left_hand_pose=self.mean_lhand.repeat(batch_size_actual, 1),
                            right_hand_pose=self.mean_rhand.repeat(batch_size_actual, 1), # use mean hand poses 
                            return_verts=True,
                            return_full_pose=True)
                        # get obj verts 
                        obj_rot = axis_angle_to_matrix(opt_dict['obj_axis'][chunk_ind:chunk_ind+batch_size])
                        obj_trans = opt_dict['obj_trans'][chunk_ind:chunk_ind+batch_size]
                        obj_pts_batch = obj_pts[None].to(self.device).repeat(batch_size_actual, 1, 1)
                        obj_verts_posed = torch.matmul(opt_dict['obj_verts'][None].repeat(batch_size_actual, 1, 1), obj_rot.permute(0, 2, 1)) + obj_trans[:, None]
                    # get verts comb 
                    verts_comb = torch.cat([smplh_out.vertices, obj_verts_posed], 1) 

                    # get rgbs
                    rgbs = []
                    for frame_time in frames_pr_chunk:
                        frame_time = frame_time.split('/')[-1]
                        t = self.time_str_to_float(frame_time)
                        rgb = video_ctrl.get_closest_frame(t)
                        scale_ratio = self.scale_ratio
                        h,w = rgb.shape[:2] 
                        # resize rgb
                        rgb = cv2.resize(rgb, (w//scale_ratio, h//scale_ratio), interpolation=cv2.INTER_LINEAR)
                        rgbs.append(rgb)
                    
                    # render the chunk, front view
                    K_render = K_full.copy()
                    K_render[:2] /= scale_ratio # no need to render full image 
                    ret = Utils.nvdiff_color_depth_render(K_render[None].repeat(batch_size_actual, 0), self.glctx, opt_dict['mesh_tensors'], (h//scale_ratio, w//scale_ratio), verts_comb, depth_only=False)
                    color_front, depth, xyz_map = ret
                    # visualize the results
                    viz = color_front.cpu().numpy()
                    viz_front = (viz * 255).astype(np.uint8)

                    # render side view as well 
                    if self.side_view_z is None:
                        self.side_view_z = torch.mean(verts_comb[:, :, 2])
                    z_now = torch.mean(verts_comb[:, :, 2])
                    if abs(z_now - self.side_view_z) > 0.5:
                        self.side_view_z = z_now
                    z = self.side_view_z
                    at = torch.tensor([[0.0, 0.0, z]], device=self.device, dtype=torch.float)
                    dist = z*1.75 if self.cfg.wild_video else z*1.3
                    R, T = look_at_view_transform(dist=dist, elev=0, azim=75, at=at, up=((0, 1, 0),))
                    view_mat = torch.eye(4, device=self.device, dtype=torch.float)
                    view_mat[:3, :3] = R[0]
                    view_mat[:3, 3] = T[0]
                    verts_side = torch.matmul(verts_comb, view_mat[:3, :3]) + view_mat[:3, 3]
                    ret = Utils.nvdiff_color_depth_render(K_render[None].repeat(batch_size_actual, 0), self.glctx, opt_dict['mesh_tensors'], (h//scale_ratio, w//scale_ratio), verts_side, depth_only=False)
                    color_side, depth, xyz_map = ret
                    # visualize the results
                    viz_side = color_side.cpu().numpy()
                    viz_side = (viz_side * 255).astype(np.uint8)

                    # add to video
                    crop_y0_ratio=0.15  if "Date03" in seq_name and not self.cfg.wild_video else 0 # only cut for behave but not wild video  
                    cut_x0_ratio = 0.15 if not self.cfg.wild_video else 0
                    cut_x1_ratio = 0.85 if not self.cfg.wild_video else 1.0

                    # visualize the 2d kpts 
                    if True:
                        joints_25_3d = self.landmark.get_body_kpts_batch_torch(smplh_out.vertices)
                        joints_25_proj = joints_25_3d @ K_full_th.T
                        joints_25_proj_xy = joints_25_proj[:, :, :2] / joints_25_proj[:, :, 2:3] / scale_ratio

                        # visualize the masks for debug 
                        color_obj = Utils.nvdiff_color_depth_render(opt_dict['K_rois'][chunk_ind:chunk_ind+batch_size], self.glctx, opt_dict['mesh_tensors_obj'], (rend_size, rend_size), obj_verts_posed, depth_only=False)[0]
                        mask_obj = color_obj.mean(-1) 
                        image_rend = opt_dict['keep_masks'][chunk_ind:chunk_ind+batch_size].to(self.device) * mask_obj.float() 
                        image_refs_chunk = opt_dict['image_refs'][chunk_ind:chunk_ind+batch_size]
                        sil_viz = torch.cat([torch.stack([image_refs_chunk, image_rend.cpu(), torch.zeros_like(image_refs_chunk)], -1),
                                            mask_obj.float().unsqueeze(-1).repeat(1, 1, 1, 3).cpu()], 2)
                        cont_mask = opt_dict['contact_mask'][chunk_ind:chunk_ind+batch_size].float().cpu().numpy()

                        # get static mask 
                        kernel = torch.ones(1, 1, 5).to(self.device) # dilate to one neighbourhood
                        cont_mask_dilate = F.conv1d(opt_dict['contact_mask'][chunk_ind:chunk_ind+batch_size].mean(-1).view(1, 1, -1), kernel, padding=2)
                        static_mask = 1 - (cont_mask_dilate > 0).float().view(-1) 
                        
                    for j, (img, vf, vs) in enumerate(zip(rgbs, viz_front, viz_side)):
                        h, w = img.shape[:2]
                        x1, x2 = int(w*cut_x0_ratio), int(w*cut_x1_ratio)
                        y1, y2 = int(h*crop_y0_ratio), int(h*1)
                        
                        # draw 2d kpts as circiles in the img 
                        if True:
                            j2d_pr, j2d_gt = joints_25_proj_xy[j].cpu().numpy().astype(int), (opt_dict['joints_2d'][chunk_ind+j, :, :2]/scale_ratio).cpu().numpy().astype(int)
                            j2d_mask = opt_dict['joints_2d'][chunk_ind+j, :, 2:3].cpu().numpy() > op_thres
                            j2d_gt[~j2d_mask.repeat(2, -1)] = 0 # set to -1 to avoid drawing 

                            for j1, j2 in zip(j2d_pr, j2d_gt):
                                cv2.circle(img, (j1[0], j1[1]), 3, (0, 255, 255), -1)
                                cv2.circle(img, (j2[0], j2[1]), 3, (255, 0, 0), -1)
                                cv2.circle(vf, (j1[0], j1[1]), 3, (0, 255, 255), -1)
                        img = np.concatenate([img[y1:y2, x1:x2], vf[y1:y2, x1:x2], vs[y1:y2, x1:x2]], 1)
                        # add frame time info
                        frame_time = frames_pr_chunk[j]
                        cv2.putText(img, frame_time, (img.shape[1]*2//3, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        if True:
                            # resize sil_viz to the same height 
                            sil_viz_j = (sil_viz[j].cpu().numpy()*255).astype(np.uint8)
                            hi = img.shape[0]
                            hs, ws = sil_viz_j.shape[:2]
                            sil_viz_j = cv2.resize(sil_viz_j, (ws*hi//hs, hi))
                            img = np.concatenate([img, sil_viz_j], 1)

                            # add contact mask text to the img
                            cont_text = f'lh: {cont_mask[j, 0]:.1f}, rh: {cont_mask[j, 1]:.1f}, static: {static_mask[j]:.1f}'
                            cv2.putText(img, cont_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        vw.append_data(img)
                
                vw.close()
                print(f"Visualization saved to {video_file_out} done")

            
        if not self.cfg.no_wandb:
            wandb.finish()
            
    def Kroi_from_corners(self, bottom_right, top_left, focal=None, principal_point=None, rend_size=256):
        crop_size = np.mean(bottom_right - top_left)  # this is a square anyways
        scale = rend_size / crop_size
        focal_roi = focal * scale
        principal_roi = (principal_point - top_left) * scale
        K_roi = np.array([[focal_roi[0], 0, principal_roi[0]],
                          [0, focal_roi[1], principal_roi[1]],
                          [0, 0, 1.]])
        return K_roi


def get_config():
    """get the config for contact optimization"""
    base_conf = OmegaConf.structured(RefineOutOptimConfig)
    cfg_cli = OmegaConf.from_cli()
    cfg = OmegaConf.merge(base_conf, cfg_cli)
    return cfg


def main():
    import glob 
    cfg = get_config() 

    if osp.isfile(cfg.pth_file):
        files = [cfg.pth_file]
    else:
        files = sorted(glob.glob(cfg.pth_file))
    if cfg.index is not None:
        chunk_size = len(files) // 8
        chunk_size = chunk_size + 1 if len(files) % 8 != 0 else chunk_size
        files = files[cfg.index * chunk_size:(cfg.index + 1) * chunk_size]
    for file in files:
        cfg.pth_file = file
        cfg.data_source = 'intercap' if 'ICap' in file else 'behave'

        opt = RefineOutOptimizer(cfg)
        opt.optimize()
    print("all done")

if __name__ == '__main__':
    main()