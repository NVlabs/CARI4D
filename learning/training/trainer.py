# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
trainer
"""
import sys, os
import time

import cv2
import trimesh
from PIL import Image

sys.path.append(os.getcwd())
import wandb
import torch
from glob import glob
from tqdm import tqdm
from omegaconf import OmegaConf
import numpy as np
from accelerate import Accelerator
from learning.datasets import get_dataset
from learning.training.training_utils import TrainState, get_scheduler
from tools.geometry_utils import geodesic_distance
from learning.training.training_config import TrainTemporalRefinerConfig
from pytorch3d.transforms.so3 import so3_log_map, so3_exp_map
from torch.optim.lr_scheduler import LambdaLR
import logging
import os.path as osp
import Utils
from lib_smpl import get_smpl, pose72to156
import torch.nn.functional as F
from accelerate import DistributedDataParallelKwargs
from learning.models import get_model
import tools.geometry_utils as geom_utils


class Trainer(object):
    def __init__(self, cfg:TrainTemporalRefinerConfig):
        self.cfg = cfg
        self.exp_dir = osp.join(cfg.save_dir, cfg.exp_name)
        os.makedirs(self.exp_dir, exist_ok=True)

        # --- 1. Initialize Accelerator ---
        # `Accelerator` will automatically handle device placement, gradient scaling, etc.
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)  # fix ddp problem
        accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

        # --- 2. Create Model, Optimizer, and Loss Function ---
        model = get_model(cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        self.train_state = TrainState()
        if cfg.lr_scheduler.type == 'none':
            scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
        else:
            scheduler = get_scheduler(cfg, optimizer)
        if cfg.ckpt_file is not None:
            ckpt_files = [cfg.ckpt_file]
        else:
            # load ckpt
            ckpt_files = sorted(glob(osp.join(self.exp_dir, '*.pth')))
        if len(ckpt_files) == 0:
            assert cfg.job == 'train', 'No ckpt found, and job is not train'
            fp_ckpt_file = "experiments/weights/2023-10-28-18-33-37/model_best.pth"
            
            if cfg.use_fp_pretrained:
                ckpt = torch.load(fp_ckpt_file)
                if 'model' in ckpt:
                    ckpt = ckpt['model']
                # TODO: adapt the pose_embed.pe tensor
                ckpt_new = {}
                for k, v in ckpt.items():
                    if k in ['pos_embed.pe', 'time_pose.pe']:
                        if model.pos_embed.pe.shape != ckpt[k].shape:
                            print(f"Warning: {k} in ckpt shape {ckpt[k].shape} != {model.pos_embed.pe.shape}")
                        else:
                            ckpt_new[k] = v
                    else: ckpt_new[k] = v
                missing_keys, unexpected_keys = model.load_state_dict(ckpt_new, strict=False)
                if len(missing_keys):
                    print(f' - Missing_keys: {missing_keys}')
                if len(unexpected_keys):
                    print(f' - Unexpected_keys: {unexpected_keys}')
                print('loaded model from checkpoint', fp_ckpt_file)
            else:
                if cfg.use_fp_head:
                    states_model = model.state_dict()
                    for k, v in states_model.items():
                        if 'rot_head' in k or 'trans_head' in k:
                            states_model[k] = ckpt[k]
                            print(f'reusing the weight of {k} from FP')
                    model.load_state_dict(states_model)
                else:
                    print("Not loading any ckpt, train from scratch!")
        else:
            # load model and optimizer state
            ckpt = torch.load(ckpt_files[-1], weights_only=False)
            ckpt_new = {}
            for k, v in ckpt['model'].items():
                if k in ['pos_embed.pe', 'time_pose.pe']:
                    if model.pos_embed.pe.shape != ckpt['model'][k].shape:
                        print(f"Warning: {k} in ckpt shape {ckpt['model'][k].shape} != {model.pos_embed.pe.shape}")
                    else:
                        ckpt_new[k] = v
                else:
                    ckpt_new[k] = v
            missing_keys, unexpected_keys = model.load_state_dict(ckpt_new, strict=False)
            if 'optimizer' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer'])
            else:
                print("Warning: no optimizer states found in the ckpt!")
            self.train_state = TrainState(ckpt['epoch'], ckpt['step'], ckpt['best_val'])
            fp_ckpt_file = ckpt_files[-1]
            if 'scheduler' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler'])
            else:
                print('No scheduler states found in the ckpt!')

            print('loaded model from checkpoint', fp_ckpt_file)
        self.ckpt_file = fp_ckpt_file
        print('loss type:', cfg.loss_type)
        print("Total number of trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

        # --- 3. Create the DataLoader ---
        dataloader_train, dataloader_val, dataset_test, dataset_train = get_dataset(cfg)

        # --- 4. The magic `accelerator.prepare()` call ---
        # This wraps all our components, making them ready for distributed training.
        self.model, self.optimizer, self.train_dataloader, self.val_dataloader, self.scheduler = accelerator.prepare(
            model, optimizer, dataloader_train, dataloader_val, scheduler,
        )
        self.accelerator = accelerator
        self.dataset_test = dataset_test

        # init logging
        if not cfg.no_wandb and accelerator.is_main_process:
            # find out the previous wandb run
            run_folders = sorted(glob(osp.join(self.exp_dir, 'wandb/run-*')))
            if len(run_folders) == 0:
                print("NO wandb run found, starting from scratch!")
                wandb.init(project=cfg.wandb_project, name=cfg.exp_name, job_type=cfg.job,
                           config=OmegaConf.to_container(cfg),
                           dir=self.exp_dir)
            else:
                print('found runs:', [osp.basename(x) for x in run_folders])
                print('resume wandb from', run_folders[-1])
                wandb.init(project=cfg.wandb_project, name=cfg.exp_name, job_type=cfg.job,
                           config=OmegaConf.to_container(cfg),
                           id=osp.basename(run_folders[-1]).split('-')[-1],
                           dir=self.exp_dir, resume='must')


        # Init human model
        if cfg.nlf_root is not None:
            self.smpl_male = get_smpl('male', True).cuda()
            self.smpl_female = get_smpl('female', True).cuda()


    def train(self):
        cfg = self.cfg
        accelerator = self.accelerator
        model, optimizer, train_dataloader, val_dataloader = self.model, self.optimizer, self.train_dataloader, self.val_dataloader
        scheduler = self.scheduler
        # --- 5. The Training Loop ---
        train_state = self.train_state
        if cfg.val_at_start:
            print('Evaluation at the start of training.')
            self.eval_model(cfg, model, train_state, val_dataloader)
        accelerator.print(f"Starting training...")
        for epoch in range(cfg.num_epochs):
            model.train()
            total_loss = 0.0

            for step, batch in enumerate(train_dataloader):
                # No need for .to(device), accelerate handles it!
                # Forward pass
                loss, loss_r, loss_t, loss_acc = self.forward_step(batch, cfg, model, vis=step%cfg.vis_every_n_steps==0)
                if not cfg.no_wandb and accelerator.is_main_process:
                    log_dict = {"loss_train": loss.item(), 'loss_train_r': loss_r.item(),
                                'loss_train_t': loss_t.item(), 'loss_train_acc': loss_acc.item(),
                                'lr': optimizer.param_groups[0]['lr']}
                    wandb.log(log_dict, step=train_state.step)

                # Backward pass - accelerator handles the backward pass
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                total_loss += loss.item()
                train_state.step += 1

                # Print progress from the main process only
                if accelerator.is_main_process and step % 20 == 0:
                    accelerator.print(f"Epoch [{epoch + 1}/{cfg.num_epochs}], Step [{step}], Loss: {loss.item():.4f}")

                if train_state.step % cfg.val_step_interval == 0:
                    self.eval_model(cfg, model, train_state, val_dataloader)
                if train_state.step % cfg.ckpt_interval == 0 and accelerator.is_main_process:
                    self.save_checkpoint(accelerator, cfg, model, optimizer, scheduler, train_state)

                lr = optimizer.param_groups[0]['lr']
                if lr < 1e-7:
                    print("Learning rate too small, stopping training.")
                    self.eval_model(cfg, model, train_state, val_dataloader)
                    if accelerator.is_main_process:
                        self.save_checkpoint(accelerator, cfg, model, optimizer, scheduler, train_state)
                    return

            # Log average loss for the epoch from the main process
            if accelerator.is_main_process:
                avg_loss = total_loss / len(train_dataloader)
                accelerator.print(f"--- End of Epoch [{epoch + 1}/{cfg.num_epochs}], Average Loss: {avg_loss:.4f} ---")
            train_state.epoch += 1
        accelerator.print("Training complete!")

    def save_checkpoint(self, accelerator, cfg, model, optimizer, scheduler, train_state):
        ckpt_file = osp.join(self.exp_dir, f'step{train_state.step:06d}.pth')
        print(f"Training state: epoch={train_state.epoch}, step={train_state.step}")
        checkpoint_dict = {
            'model': accelerator.unwrap_model(model).state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': train_state.epoch,
            'step': train_state.step,
            'best_val': train_state.best_val,
            'cfg': cfg
        }
        accelerator.save(checkpoint_dict, ckpt_file)
        print(f"ckpt saved to {ckpt_file}")

    def eval_model(self, cfg, model, train_state, val_dataloader):
        model.eval()
        loss_val = []
        loss_val_r, loss_val_t = [], []
        loss_val_acc = []
        for step_val, batch in enumerate(tqdm(val_dataloader)):
            if step_val >= cfg.max_step_val:
                break
            with torch.no_grad():
                loss, loss_r, loss_t, loss_acc = self.forward_step(batch, cfg, model, vis=step_val==0)
            loss_val.append(loss.item())
            loss_val_r.append(loss_r.item())
            loss_val_t.append(loss_t.item())
            loss_val_acc.append(loss_acc.item())
        loss_val = np.mean(loss_val)
        if train_state.best_val is None or loss_val < train_state.best_val:
            train_state.best_val = loss_val
        if not cfg.no_wandb and self.accelerator.is_main_process:
            # logging only using the main process
            wandb.log({'loss_val': loss_val, 'loss_val_r': np.mean(loss_val_r),
                       'loss_val_t': np.mean(loss_val_t), 'loss_val_acc': np.mean(loss_val_acc)},
                      step=train_state.step)
        print(f'--- Eval at step {train_state.step}, loss: {loss_val:.4f} lr: {self.optimizer.param_groups[0]["lr"]:.5f} ---')
        model.train()

    def forward_step(self, batch, cfg, model, vis=False):
        "one model forward and return loss"
        # pre-trained model: A is the rendered, B is the input
        rot_delta_pred, rot_delta_gt, trans_delta_gt, trans_delta_pred, out_dict = self.forward_batch(batch, cfg, model, vis, ret_dict=True)
        loss_acc = torch.tensor(0, device=rot_delta_gt.device)
        if cfg['loss_type'] == 'l1':
            loss_t = torch.abs(trans_delta_pred - trans_delta_gt).mean()
            loss_r = torch.abs(rot_delta_pred - rot_delta_gt).mean() * cfg['w_rot']
            loss =  loss_r + loss_t
        elif cfg['loss_type'] == 'l1+self-acc':
            loss_t = torch.abs(trans_delta_pred - trans_delta_gt).mean()
            loss_r = torch.abs(rot_delta_pred - rot_delta_gt).mean() * cfg['w_rot']
            loss = loss_r + loss_t

            poseA = batch['pose_perturbed']
            B_in_cams, B_in_cams_gt = self.compute_abspose(poseA.shape[0], batch, cfg, poseA, rot_delta_pred, rot_delta_gt,
                                                           trans_delta_gt, trans_delta_pred)
            d1 = geodesic_distance(B_in_cams[:, 1:-1, :3, :3].reshape(-1, 3, 3),
                                          B_in_cams[:, :-2, :3, :3].reshape(-1, 3, 3))
            d2 = geodesic_distance(B_in_cams[:, 1:-1, :3, :3].reshape(-1, 3, 3),
                                   B_in_cams[:, 2:, :3, :3].reshape(-1, 3, 3)) # (B, t-2, )
            loss_acc = torch.abs(d1 - d2).mean() * self.cfg.lw_acc
            loss = loss + loss_acc
        elif cfg['loss_type'] == 'l1-abs':
            # predicting absolute pose
            pose_gt = batch['pose_gt'] # (B, T, 4, 4)
            rot_gt_axis = so3_log_map(pose_gt[:, :, :3, :3].reshape(-1, 3, 3).permute(0, 2, 1))
            loss_r = torch.abs(rot_delta_pred - rot_gt_axis).mean() * cfg['w_rot']
            trans_gt = pose_gt[:, :, :3, 3].reshape(-1, 3)
            loss_t = torch.abs(trans_delta_pred - trans_gt).mean()
            loss = loss_r + loss_t
            # it is not able to predict depth?
        elif cfg['loss_type'] == 'l1-abs-delta':
            # model predicts both delta and abs pose
            loss_t = torch.abs(trans_delta_pred - trans_delta_gt).mean()
            loss_r = torch.abs(rot_delta_pred - rot_delta_gt).mean() * cfg['w_rot']

            # abs pose error
            pose_gt = batch['pose_gt']  # (B, T, 4, 4)
            if self.cfg.rot_rep == 'axis_angle':
                rot_gt_axis = so3_log_map(pose_gt[:, :, :3, :3].reshape(-1, 3, 3).permute(0, 2, 1)) # BT, 3
                loss_r_abs = torch.abs(out_dict['rot_abs'] - rot_gt_axis).mean() * cfg.w_abs_rot
            elif self.cfg.rot_rep == '6d':
                rot_gt_axis = pose_gt[:, :, :3, 0:3].reshape(-1, 3, 3).view(-1, 6)
                loss_r_abs = torch.abs(out_dict['rot_abs'] - rot_gt_axis).mean() * cfg.w_abs_rot
            else:
                raise NotImplementedError
            trans_gt = pose_gt[:, :, :3, 3].reshape(-1, 3)
            if cfg.loss_abs_trans_rela:
                trans_gt_rela = pose_gt[:, :, :3, 3].clone() - pose_gt[:, 0:1, :3, 3] # relative to 1st frame
                loss_t_abs = torch.abs(out_dict['trans_abs_rela'] - trans_gt_rela).mean() * cfg.w_abs_trans
            else:
                print('loss directly to final GT translation')
                loss_t_abs = torch.abs(out_dict['trans_abs'] - trans_gt).mean() * cfg.w_abs_trans
            loss = loss_r + loss_t + loss_r_abs + loss_t_abs

            if not self.cfg.no_wandb and self.accelerator.is_main_process:
                key = 'train' if self.model.training else 'val'
                wandb.log({f'loss_t_abs_{key}': loss_t_abs, f'loss_r_abs_{key}': loss_r_abs,}, step=self.train_state.step)
        elif cfg['loss_type'] == 'l2-abs-delta':
            # model predicts both delta and abs pose
            loss_t = ((trans_delta_pred - trans_delta_gt)**2).mean()
            loss_r = ((rot_delta_pred - rot_delta_gt)**2).mean() * cfg['w_rot']

            # abs pose error
            pose_gt = batch['pose_gt']  # (B, T, 4, 4)
            rot_gt_axis = so3_log_map(pose_gt[:, :, :3, :3].reshape(-1, 3, 3).permute(0, 2, 1)) # BT, 3
            loss_r_abs = ((out_dict['rot_abs'] - rot_gt_axis)**2).mean() * cfg.w_abs_rot
            trans_gt = pose_gt[:, :, :3, 3].reshape(-1, 3)
            if cfg.loss_abs_trans_rela:
                trans_gt_rela = pose_gt[:, :, :3, 3].clone() - pose_gt[:, 0:1, :3, 3] # relative to 1st frame
                loss_t_abs = ((out_dict['trans_abs_rela'] - trans_gt_rela)**2).mean() * cfg.w_abs_trans
            else:
                loss_t_abs = ((out_dict['trans_abs'] - trans_gt)**2).mean() * cfg.w_abs_trans
            loss = loss_r + loss_t + loss_r_abs + loss_t_abs

            if not self.cfg.no_wandb and self.accelerator.is_main_process:
                key = 'train' if self.model.training else 'val'
                wandb.log({f'loss_t_abs_{key}': loss_t_abs, f'loss_r_abs_{key}': loss_r_abs,}, step=self.train_state.step)
        elif cfg['loss_type'] in ['l1-absrot-delta', 'l1-absrot-delta-hum', 'l2-absrot-delta-humabs', 'l1-absrot-delta-humabs', 'l2-absrot-delta-hum']:
            assert cfg.obj_pose_dim_input in [3, 6], 'must encode object rotation only!'
            loss_func = F.l1_loss if 'l1' in cfg['loss_type'] else F.mse_loss
            key = 'train' if self.model.training else 'val'
            loss_dict = {}

            # model predicts both delta and abs pose, abs pose contains rotation only
            frame_mask = batch['frame_mask'].unsqueeze(-1) # (B, T, 1)
            bs, t = frame_mask.shape[:2]
            FIX_LOSS = 0
            if self.cfg.pred_uncertainty:
                # see https://github.com/martius-lab/beta-nll/blob/master/depth_estimation/models/unet_adaptive_bins.py#L189
                uncert_t = F.softplus(out_dict['trans_uncertainty']) + self.cfg.var_epsilon # (BT, 1) eps to protect
                uncert_r = F.softplus(out_dict['rot_uncertainty']) + self.cfg.var_epsilon
                FIX_LOSS = 20 
                loss_t = 0.5 *(loss_func(trans_delta_pred, trans_delta_gt, reduction='none')/uncert_t + uncert_t.log()+ FIX_LOSS) # * cfg['w_transl']
                loss_t = ((loss_t * (uncert_t.detach() ** cfg.beta_nll)).reshape(bs, t, -1)*frame_mask).mean()* cfg['w_transl']
                # do the same for r
                loss_r = 0.5 * (loss_func(rot_delta_pred, rot_delta_gt, reduction='none')/uncert_r + uncert_r.log()+ FIX_LOSS)
                loss_r = ((loss_r * (uncert_r.detach()**cfg.beta_nll)).reshape(bs, t, -1)*frame_mask).mean() * cfg['w_rot']

                # keep track of the classic loss
                with torch.no_grad():
                    loss_t_raw = (loss_func(trans_delta_pred, trans_delta_gt, reduction='none').reshape(bs, t,
                                                                                                    -1) * frame_mask).mean() * cfg['w_transl']
                    loss_r_raw = (loss_func(rot_delta_pred, rot_delta_gt, reduction='none').reshape(bs, t,
                                                                                                -1) * frame_mask).mean() * cfg['w_rot']
                    loss_dict[f'{key}/loss_t_raw'] = loss_t_raw
                    loss_dict[f'{key}/loss_r_raw'] = loss_r_raw

            else:
                if self.cfg.symm_loss:
                    # consider symmetries. 
                    B_in_cams_interm = self.abspose_from_relative(batch, cfg, batch['pose_perturbed'], out_dict['rot'], out_dict['trans']) # (B, T, 4, 4) 
                    pose_gt_symm = batch['pose_gt_symm'] # (B, T, N, 4, 4) 
                    # simply the smallest rotation and translation error from all symmetries
                    loss_r = loss_func(B_in_cams_interm[:, :, None, :3, :3], pose_gt_symm[:, :, :, :3, :3], reduction='none').sum(dim=(-1, -2)).min(-1)[0] # (B, T, N)
                    # mask with frame_mask and weight 
                    loss_r = (loss_r[:, :, None] * frame_mask).mean() * cfg['w_rot']
                    # same for translation
                    loss_t = loss_func(B_in_cams_interm[:, :, None, :3, 3], pose_gt_symm[:, :, :, :3, 3], reduction='none').sum(dim=(-1)).min(-1)[0] # (B, T, N)
                    loss_t = (loss_t[:, :, None] * frame_mask).mean() * cfg['w_transl']
                else:
                    loss_t = (loss_func(trans_delta_pred, trans_delta_gt, reduction='none').reshape(bs, t, -1)*frame_mask).mean()* cfg['w_transl']
                    loss_r = (loss_func(rot_delta_pred, rot_delta_gt, reduction='none').reshape(bs, t, -1)*frame_mask).mean() * cfg['w_rot']


            # abs pose error, TODO: also add symmetry loss here 
            pose_gt = batch['pose_gt']  # (B, T, 4, 4)
            if self.cfg.rot_rep == 'axis_angle':
                rot_gt_axis = so3_log_map(pose_gt[:, :, :3, :3].reshape(-1, 3, 3).permute(0, 2, 1))  # BT, 3
                loss_r_abs = (loss_func(out_dict['rot_abs'], rot_gt_axis, reduction='none').reshape(bs, t, -1)*frame_mask).mean() * cfg.w_abs_rot
            elif self.cfg.rot_rep == '6d':
                if self.cfg.symm_loss:
                    pose_gt_symm = batch['pose_gt_symm'] # (B, T, N, 4, 4) 
                    N = pose_gt_symm.shape[2]
                    pose_gt_symm6d = pose_gt_symm[..., :3, 0:2].reshape(-1, N, 6) # BT, N, 6
                    # compute min of all symmetries 
                    loss_r_abs = loss_func(out_dict['rot_abs'][:, None], pose_gt_symm6d, reduction='none').sum(-1).min(-1)[0] # (B, T, N) -> (B, T)
                    loss_r_abs = (loss_r_abs.reshape(bs, t, 1)*frame_mask).mean() * cfg.w_abs_rot
                else:
                    rot_gt_axis = pose_gt[:, :, :3, 0:2].reshape(-1, 3, 2).reshape(-1, 6)
                    loss_r_abs = (loss_func(out_dict['rot_abs'], rot_gt_axis, reduction='none').reshape(bs, t, -1)*frame_mask).mean() * cfg.w_abs_rot
            else:
                raise NotImplementedError

            loss_t_abs = torch.tensor(0, device=rot_delta_gt.device) # abs do not correct translation
            loss = loss_r + loss_t + loss_r_abs + loss_t_abs
            loss_dict.update(**{f'loss_t_abs_{key}': loss_t_abs, f'loss_r_abs_{key}': loss_r_abs})

            # velocity of the abs object pose 

            # compute additional human pose loss
            if self.cfg.nlf_root is not None:
                if cfg.loss_type in ['l1-absrot-delta-hum', 'l2-absrot-delta-hum']:
                    smpl_delta_r = out_dict['hum_pose']
                    smpl_delta_t = out_dict['hum_trans']

                    if self.cfg.rot_rep_hum == '6d':
                        gt_delta_r = batch['delta_smpl_rot'][:, :, :, :, :2].reshape(-1, 24*6)
                        gt_delta_t = batch['delta_smpl_trans'].reshape(-1, 3)
                        if not self.cfg.pred_uncertainty:
                            loss_hum_t = (loss_func(smpl_delta_t, gt_delta_t, reduction='none').reshape(bs, t, -1)*frame_mask).mean() * self.cfg.w_hum_t
                            loss_hum_r = (loss_func(smpl_delta_r, gt_delta_r, reduction='none').reshape(bs, t, -1)*frame_mask).mean() * self.cfg.w_hum_rot
                        else:
                            uncert_pose = F.softplus(out_dict['hum_pose_uncertainty']).unsqueeze(-1) + self.cfg.var_epsilon# (BT, 24, 1)
                            uncert_smpl_t = F.softplus(out_dict['hum_trans_uncertainty']) + self.cfg.var_epsilon
                            smpl_delta_r = smpl_delta_r.reshape(-1, 24, 6)
                            gt_delta_r = gt_delta_r.reshape(-1, 24, 6)

                            loss_hum_t = 0.5 * (loss_func(smpl_delta_t, gt_delta_t, reduction='none')/uncert_smpl_t + uncert_smpl_t.log()+ FIX_LOSS)
                            loss_hum_t = ((loss_hum_t * (uncert_smpl_t.detach()**self.cfg.beta_nll)).reshape(bs, t, -1)*frame_mask).mean() * self.cfg.w_hum_t
                            loss_hum_r = 0.5 * (loss_func(smpl_delta_r, gt_delta_r, reduction='none')/uncert_pose + uncert_pose.log()+ FIX_LOSS)
                            loss_hum_r = ((loss_hum_r * (uncert_pose.detach()**self.cfg.beta_nll)).reshape(bs, t, -1)*frame_mask).mean() * self.cfg.w_hum_rot
                            # keep track of the classic loss
                            with torch.no_grad():
                                loss_hum_t_raw = (loss_func(smpl_delta_t, gt_delta_t, reduction='none').reshape(bs, t, -1) * frame_mask).mean() * self.cfg.w_hum_t
                                loss_hum_r_raw = (loss_func(smpl_delta_r, gt_delta_r, reduction='none').reshape(bs, t, -1) * frame_mask).mean() * self.cfg.w_hum_rot
                                loss_dict[f'{key}/loss_hum_r_raw'] = loss_hum_r_raw
                                loss_dict[f'{key}/loss_hum_t_raw'] = loss_hum_t_raw
                        loss_dict[f'loss_hum_r_{key}'] = loss_hum_r
                        loss_dict[f'loss_hum_t_{key}'] = loss_hum_t

                        loss_hum_j = 0. # joints position loss
                        if self.cfg.w_hum_j > 0.:
                            betas, pred_smpl_pose, pred_smpl_r, pred_smpl_t = self.smpl_params_from_pred(batch, out_dict)

                            male_mask = batch['is_male'].reshape(-1).bool() # B*L, same shape as pred_smpl_pose
                            assert len(male_mask) == len(pred_smpl_pose)
                            idx_m, idx_f = male_mask.nonzero(as_tuple=True)[0], (~male_mask).nonzero(as_tuple=True)[0]
                            idx_list, jtrs_pr_list = [], []
                            J = 24 
                            if idx_m.numel() > 0:
                                idx_list.append(idx_m)
                                # use male smpl model to get joints 
                                jts_pr_m = self.smpl_male.get_joints(pose72to156(pred_smpl_pose[idx_m]), betas[idx_m], pred_smpl_t[idx_m]) # [:, :J] # take only the first 23 joints without wrists 
                                jtrs_pr_list.append(jts_pr_m)
                            if idx_f.numel() > 0:
                                idx_list.append(idx_f)
                                jts_pr_f = self.smpl_female.get_joints(pose72to156(pred_smpl_pose[idx_f]), betas[idx_f], pred_smpl_t[idx_f]) # [:, :J] # take only the first 23 joints without wrists 
                                jtrs_pr_list.append(jts_pr_f)
                            jts_pr_list = torch.cat(jtrs_pr_list, dim=0)
                            perm = torch.cat(idx_list, dim=0)    # original positions of each sub-batch
                            jts_pr = jts_pr_list[torch.argsort(perm)]        # (B, ...), restored to original order

                            jts_gt = batch['smpl_jtrs_gt'].reshape(-1, jts_pr.shape[-2], 3) # .reshape(-1, J, 3)
                            if not self.cfg.pred_uncertainty:
                                loss_hum_j = (loss_func(jts_pr, jts_gt, reduction='none').reshape(bs, t, -1)*frame_mask).mean() * self.cfg.w_hum_j
                            else:
                                loss_hum_j = 0.5 * (loss_func(jts_pr, jts_gt, reduction='none') / uncert_pose + uncert_pose.log() + FIX_LOSS)
                                loss_hum_j = ((loss_hum_j * (uncert_pose.detach() ** self.cfg.beta_nll)).reshape(bs, t, -1) * frame_mask).mean() * self.cfg.w_hum_j
                                with torch.no_grad():
                                    loss_hum_j_raw = (loss_func(jts_pr, jts_gt, reduction='none').reshape(bs, t,-1) * frame_mask).mean() * self.cfg.w_hum_j
                                    loss_dict[f'{key}/loss_hum_j_raw'] = loss_hum_j_raw

                        loss_hum_b = 0.
                        if self.cfg.w_hum_b > 0:
                            loss_hum_b = (loss_func(betas, batch['betas_gt'].reshape(-1, 10), reduction='none').reshape(bs, t, -1)*frame_mask).mean() * self.cfg.w_hum_b
                            loss_dict[f'{key}/loss_hum_b'] = loss_hum_b
                        loss_dict[f'loss_hum_j_{key}'] = loss_hum_j

                        # add velocity loss
                        loss_velo = 0.
                        if self.cfg.w_velo > 0:
                            # Oct29 midnight: v2: for human use the joint loss and object use the translation loss
                            jts_pr = jts_pr.reshape(bs, t, -1, 3)
                            jts_gt = batch['smpl_jtrs_gt'] 
                            velo_pr_hj = jts_pr[:, 1:] - jts_pr[:, :-1]
                            velo_gt_hj = jts_gt[:, 1:] - jts_gt[:, :-1]
                            velo_pr_ot = B_in_cams_interm[:, 1:, :3, 3] - B_in_cams_interm[:, :-1, :3, 3]
                            velo_gt_ot = pose_gt[:, 1:, :3, 3] - pose_gt[:, :-1, :3, 3]
                            loss_velo_hj = F.mse_loss(velo_pr_hj, velo_gt_hj, reduction='none').sum(-1).mean() 
                            loss_velo_ot = F.mse_loss(velo_pr_ot, velo_gt_ot, reduction='none').sum(-1).mean()  
                            loss_velo = (loss_velo_hj + loss_velo_ot) * self.cfg.w_velo
                            loss_dict[f'{key}/loss_velo'] = loss_velo
                        # contact prediction
                        loss_contact = 0.
                        if self.cfg.cont_out_dim > 0:
                            cont_gt = batch['contact_dist_gt'] # (B, T, 52)
                            cont_gt_hands = cont_gt[:, :, [22, 23+15]]
                            cont_pred = out_dict['contact'].reshape(bs, t, -1)
                            if self.cfg.cont_out_type == 'binary':
                                # bce loss 
                                loss_bce = (F.binary_cross_entropy_with_logits(cont_pred, (cont_gt_hands < self.cfg.cont_mask_thres).float(), reduction='none')*frame_mask ).mean() * self.cfg.w_contact
                                loss_dict[f'{key}/loss_contact'] = loss_bce
                                loss_contact = loss_bce
                            elif self.cfg.cont_out_type == 'distance':
                                # mse loss
                                loss_mse = (F.mse_loss(cont_pred, cont_gt_hands, reduction='none')*frame_mask).mean() * self.cfg.w_contact
                                loss_dict[f'{key}/loss_contact'] = loss_mse
                                loss_contact = loss_mse
                        
                        loss += loss_hum_t + loss_hum_r + loss_hum_j + loss_hum_b + loss_velo + loss_contact
                        print(f'step {self.train_state.step} hum_r:{loss_hum_r:.3f}, hum_t:{loss_hum_t:.3f}, hum_j: {loss_hum_j:.3f}, hum_b: {loss_hum_b:.3f}, velo: {loss_velo:.3f}, r_abs:{loss_r_abs:.3f}, t_abs:{loss_t_abs:.3f}, r: {loss_r:.3f}, t: {loss_t:.3f}, contact: {loss_contact:.3f}, tot: {loss:.3f}')
                    else:
                        raise NotImplementedError
                elif cfg.loss_type in ['l2-absrot-delta-humabs', 'l1-absrot-delta-humabs']:
                    assert self.cfg.w_hum_j > 0.
                    # now predicting abs
                    trans_pr = out_dict['body_transl'] # (BT, 3)
                    trans_gt = batch['smpl_transl_gt'].reshape(-1, 3)
                    # compute loss on parameters, joints
                    loss_hum_t = (loss_func(trans_pr, trans_gt, reduction='none').reshape(bs, t, -1)*frame_mask).mean() * self.cfg.w_hum_t
                    rotmat_gt = batch['smpl_rotmat_gt'].reshape(-1, 24, 3, 3)
                    rotmat_pr = out_dict['body_rotmat'] # (BT, 24, 3, 3)
                    loss_hum_r = (loss_func(rotmat_pr, rotmat_gt, reduction='none').reshape(bs, t, -1)*frame_mask).mean() * self.cfg.w_hum_rot
                    bs = batch['smpl_transl_gt'].shape[0]
                    betas = batch['betas_gt'].reshape(self.cfg.clip_len * bs, 10)
                    jts_pr = self.smpl_model.get_joints(rotmat_pr.reshape(self.cfg.clip_len * bs, 24*9), betas, trans_pr, axis2rot=False)  # (BT, J, 3)
                    jts_gt = batch['smpl_jtrs_gt'].reshape(-1, 24, 3)
                    loss_hum_j = (loss_func(jts_pr, jts_gt, reduction='none').reshape(bs, t, -1)*frame_mask).mean().item() * self.cfg.w_hum_j
                    loss += loss_hum_t + loss_hum_r + loss_hum_j

                    loss_dict[f'loss_hum_r_{key}'] = loss_hum_r
                    loss_dict[f'loss_hum_t_{key}'] = loss_hum_t
                    loss_dict[f'loss_hum_j_{key}'] = loss_hum_j


            if not self.cfg.no_wandb and self.accelerator.is_main_process:
                wandb.log(loss_dict, step=self.train_state.step)

        elif cfg['loss_type'] == 'l1+geo-acc':
            loss_t = torch.abs(trans_delta_pred - trans_delta_gt).mean()
            loss_r = torch.abs(rot_delta_pred - rot_delta_gt).mean() * cfg['w_rot']
            loss = loss_r + loss_t

            poseA = batch['pose_perturbed']
            B_in_cams, B_in_cams_gt = self.compute_abspose(poseA.shape[0], batch, cfg, poseA, rot_delta_pred, rot_delta_gt,
                                                           trans_delta_gt, trans_delta_pred)
            d1 = geodesic_distance(B_in_cams[:, 1:, :3, :3].reshape(-1, 3, 3),
                                          B_in_cams[:, :-1, :3, :3].reshape(-1, 3, 3))
            d2 = geodesic_distance(B_in_cams_gt[:, 1:, :3, :3].reshape(-1, 3, 3),
                                   B_in_cams_gt[:, :-1, :3, :3].reshape(-1, 3, 3))
            loss_acc = torch.abs(d1 - d2).mean() * self.cfg.lw_acc
            print(f'loss_acc: {loss_acc:.4f}, loss: {loss:.4f}')
            loss = loss + loss_acc

        elif cfg['loss_type'] == 'l2': # default L2
            loss_t = ((trans_delta_pred - trans_delta_gt) ** 2).mean()
            loss_r = ((rot_delta_pred - rot_delta_gt) ** 2).mean() * cfg['w_rot']
            loss = loss_r + loss_t

            # add acceleration loss
            if self.cfg.lw_acc>0:
                trans_delta_uno = trans_delta_pred * batch['mesh_diameter'].reshape(len(trans_delta_pred), -1) / 2.
                rot_delta_uno = so3_exp_map(rot_delta_pred * self.cfg['rot_normalizer']).permute(0, 2, 1)
                # now convert to (B, T...)
                poseA = batch['pose_perturbed'] # (B, T, 4, 4)
                poseB = batch['pose_gt']
                B, T = poseA.shape[:2]
                pose_corrected = Utils.egocentric_delta_pose_to_pose(poseA.reshape(-1, 4, 4), trans_delta=trans_delta_uno,
                                                          rot_mat_delta=rot_delta_uno)
                pose_corrected = pose_corrected.reshape(B, T, 4, 4)

                acc_t_gt = poseB[:, :-2, :3, 3] - 2 * poseB[:, 1:-1, :3, 3] + poseB[:, 2:, :3, 3]
                acc_t_pred = pose_corrected[:, :-2, :3, 3] - 2 * pose_corrected[:, 1:-1, :3, 3] + pose_corrected[:, 2:, :3, 3]

                axis_gt = so3_log_map(poseB[:, :, :3, :3].reshape(-1, 3, 3)).reshape(B, T, -1)
                axis_pr = so3_log_map(pose_corrected[:, :, :3, :3].reshape(-1, 3, 3)).reshape(B, T, -1)
                acc_r_gt = axis_gt[:, :-2] - 2 * axis_gt[:, 1:-1] + axis_gt[:, 2:]
                acc_r_pr = axis_pr[:, :-2] - 2 * axis_pr[:, 1:-1] + axis_pr[:, 2:]
                la_t = F.l1_loss(acc_t_pred, acc_t_gt).mean()
                la_r = F.l1_loss(acc_r_gt, acc_r_pr).mean()
                loss_acc = (la_r + la_t) * self.cfg.lw_acc
            loss = loss + loss_acc

            # this needs to be computed in the original pose space.
        else:
            raise RuntimeError

        return loss, loss_r, loss_t, loss_acc

    def forward_batch(self, batch, cfg, model, vis=False, ret_dict=False):
        "forward one batch"
        imgsB, imgsA = batch['input_rgbs'], batch['render_rgbs']
        xyzB, xyzA = batch['input_xyz'], batch['render_xyz']
        pose_perturbed = batch['poseA_norm']
        output = model(torch.cat([imgsA, xyzA], 2), torch.cat([imgsB, xyzB], 2), pose_perturbed, batch)
        trans_delta_gt = batch['delta_transl']  # (B, T, 3)
        mesh_radius = batch['mesh_diameter'] / 2.  # (B, T)
        trans_normalizer = batch['trans_normalizer']  # (B, T, 3)
        B, T = trans_delta_gt.shape[:2]
        # use diameter: the xyz map is normalized by object diameter
        if cfg['normalize_xyz']:
            trans_delta_gt *= 1 / mesh_radius.reshape(len(trans_delta_gt), T, -1)
        else:
            trans_delta_gt = trans_delta_gt / trans_normalizer
            if not (torch.abs(trans_delta_gt) <= 1 + 1e-3).all():
                logging.info("ERROR label")
        rot_delta_mat_gt = batch['delta_rot']
        rot_delta_gt = so3_log_map(rot_delta_mat_gt.reshape(B * T, 3, 3).permute(0, 2, 1))  # permute: pyt3d so3 uses col order
        rot_delta_gt = rot_delta_gt / cfg['rot_normalizer']  # random noise sample range.
        trans = output['trans'].float()  # (BT,3)
        rot = output['rot'].float()  # BT, 3
        trans_delta_pred = trans
        trans_delta_gt = trans_delta_gt.reshape(B * T, 3)  # FP was trained to predict

        # log error
        if self.accelerator.is_main_process and not cfg.no_wandb:
            with torch.no_grad():
                log_dict = {}
                poseA = batch['pose_perturbed']
                B_in_cams, B_in_cams_gt = self.compute_abspose(B, batch, cfg, poseA, rot, rot_delta_gt,
                                                               trans_delta_gt, trans_delta_pred, output)
                err_t = torch.sum((B_in_cams_gt[:, :, :3, 3] - B_in_cams[:, :, :3, 3]) ** 2, -1).sqrt().mean()
                err_r = geodesic_distance(B_in_cams_gt[:, :, :3, :3].reshape(-1, 3, 3),
                                          B_in_cams[:, :, :3, :3].reshape(-1, 3, 3)).mean()
                key = 'train' if model.training else 'val'
                log_dict[f'{key}/err_t'] = err_t
                log_dict[f'{key}/err_r'] = err_r
                log_dict[f'{key}/err_r_deg'] = err_r * 180/torch.pi

                # log contact accuracy 
                if self.cfg.cont_out_dim > 0:
                    cont_gt = batch['contact_dist_gt'] # (B, T, 52)
                    cont_gt_hands = cont_gt[:, :, [22, 23+15]]
                    cont_pred = output['contact'].reshape(B, T, -1)
                    if self.cfg.cont_out_type == 'binary':
                        cont_acc = (cont_pred > 0).float() == (cont_gt_hands < self.cfg.cont_mask_thres).float()
                    elif self.cfg.cont_out_type == 'distance':
                        cont_acc = (cont_pred < self.cfg.cont_mask_thres).float() == (cont_gt_hands < self.cfg.cont_mask_thres).float()
                    log_dict[f'{key}/cont_acc'] = cont_acc.float().mean()
                # log error of intermediate predictions
                if self.cfg['loss_type'] in ['l1-abs-delta', 'l1-absrot-delta', 'l1-absrot-delta-hum', 'l1-absrot-delta-humabs', 'l2-absrot-delta-humabs']:
                    B_in_cams_interm = self.abspose_from_relative(batch, cfg, poseA, output['rot'], output['trans'])
                    # also compute symmetries 
                    if self.cfg.symm_loss:
                        B_in_cams_gt_symm = batch['pose_gt_symm'] # (B, T, N, 4, 4)
                        err_r = geodesic_distance(B_in_cams_gt_symm[:, :, :, :3, :3].reshape(-1, 3, 3),
                                              B_in_cams_interm[:, :, None, :3, :3].repeat(1, 1, B_in_cams_gt_symm.shape[2], 1, 1).reshape(-1, 3, 3)).reshape(-1, B_in_cams_gt_symm.shape[2]).min(-1)[0]
                        err_r = err_r.mean()
                        # do the same for translation
                        err_t = torch.sum((B_in_cams_gt_symm[:, :, :, :3, 3].reshape(-1, 3) - B_in_cams_interm[:, :, None, :3, 3].repeat(1, 1, B_in_cams_gt_symm.shape[2], 1).reshape(-1, 3)) ** 2, -1).sqrt()
                        err_t = err_t.reshape(-1, B_in_cams_gt_symm.shape[2]).min(-1)[0].mean()
                    else:
                        err_t = torch.sum((B_in_cams_gt[:, :, :3, 3] - B_in_cams_interm[:, :, :3, 3]) ** 2, -1).sqrt().mean()
                        err_r = geodesic_distance(B_in_cams_gt[:, :, :3, :3].reshape(-1, 3, 3),
                                                B_in_cams_interm[:, :, :3, :3].reshape(-1, 3, 3)).mean()

                    key = 'train' if model.training else 'val'
                    log_dict[f'{key}/err_interm_t'] = err_t
                    log_dict[f'{key}/err_interm_r'] = err_r
                wandb.log(log_dict, step=self.train_state.step)

        # visualize input and output predictions
        if vis and self.accelerator.is_main_process:
            start = time.time()
            bid = 0  # batch id
            skip = 16  # log every N frame

            key = 'train' if self.model.training else 'val'
            log_dict = {}
            maskA, maskB, rgbsA, rgbsB, xyzA, xyzB = self.prepare_input_viz(batch, cfg)
            poseB = batch['pose_gt'] # this does not match B_in_cams_gt!
            # TODO: replace poseB with poseA + delta GT

            poseA = batch['pose_perturbed']
            to_origin = batch['to_origin'][bid].cpu().numpy()
            bbox = batch['obj_bbox_3d'][bid].cpu().numpy()
            clip_len = rgbsA.shape[1]

            # log error
            with torch.no_grad():
                B_in_cams, B_in_cams_gt = self.compute_abspose(B, batch, cfg, poseA, rot, rot_delta_gt,
                                                               trans_delta_gt, trans_delta_pred, output)
                err_t = torch.sum((B_in_cams_gt[:, :, :3, 3] - B_in_cams[:, :, :3, 3])**2, -1).sqrt().mean()
                err_r = geodesic_distance(B_in_cams_gt[:, :, :3, :3].reshape(-1, 3, 3), B_in_cams[:, :, :3, :3].reshape(-1, 3, 3)).mean()
                key = 'train' if model.training else 'val'
                log_dict[f'{key}/err_t'] = err_t
                log_dict[f'{key}/err_r'] = err_r

                # log SMPL evaluation
                NJ = 24
                if cfg.nlf_root is not None:
                    verts_smpl_gt, verts_smpl, jtrs_gt, jtrs_pr, pred_smpl_r, pred_smpl_t, jts_rot_pr, jts_rot_gt = self.compute_smpl_verts(
                        batch, output)
                    v2v = torch.sum((verts_smpl_gt - verts_smpl) ** 2, -1).sqrt().mean()
                    mpjpe = torch.sum((jtrs_pr - jtrs_gt) ** 2, -1).sqrt().mean()
                    mpjae = Utils.geodesic_distance_batch(jts_rot_pr, jts_rot_gt).mean()
                    ste = torch.sum((batch['smpl_transl_gt'].reshape(-1, 3) - pred_smpl_t) ** 2).sqrt().mean()
                    log_dict[f'{key}/v2v'] = v2v
                    log_dict[f'{key}/mpjpe'] = mpjpe
                    log_dict[f'{key}/mpjae'] = mpjae
                    log_dict[f'{key}/smpl_t'] = ste

            for i in range(0, clip_len, skip):
                comb, rgba, rgbb = self.visualize_rgbm(batch, bid, i, maskA, maskB, rgbsA, rgbsB)
                # add xyz as well
                xyza_vis = (np.clip(xyzA[bid, i].transpose(1, 2, 0)+0.5, 0, 1.)* 255).astype(np.uint8)
                xyzb_vis = (np.clip(xyzB[bid, i].transpose(1, 2, 0)+0.5, 0, 1.)* 255).astype(np.uint8)
                comb = np.concatenate([comb, np.concatenate([xyza_vis, xyzb_vis], 0)], axis=1)
                # Visualize pose predictions as well
                K = batch['K_rois'][bid, i].cpu().numpy()
                center_pose = B_in_cams_gt[bid, i].cpu().numpy() @ np.linalg.inv(to_origin)
                vis_gt, vis_input, vis_pred = rgbb.copy(), rgba.copy(), rgbb.copy()
                vis_gt = Utils.draw_posed_3d_box(K, img=vis_gt, ob_in_cam=center_pose, bbox=bbox, line_color=(0, 255, 0))
                vis_gt = Utils.draw_xyz_axis(vis_gt, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3,
                                             transparency=0, is_input_rgb=True)
                center_pose = poseA[0, i].cpu().numpy() @ np.linalg.inv(to_origin)
                vis_input = Utils.draw_posed_3d_box(K, img=vis_input, ob_in_cam=center_pose, bbox=bbox, line_color=(255, 0, 0))
                vis_input = Utils.draw_xyz_axis(vis_input, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3,
                                             transparency=0, is_input_rgb=True)

                pose = B_in_cams[bid, i].detach().cpu().numpy()
                center_pose = pose @ np.linalg.inv(to_origin)
                vis_pred = Utils.draw_posed_3d_box(K, img=vis_pred, ob_in_cam=center_pose, bbox=bbox, line_color=(0, 255, 255))
                vis_pred = Utils.draw_xyz_axis(vis_pred, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3,
                                                transparency=0, is_input_rgb=True)

                # now show an overlap
                vis_comb = vis_gt.copy()
                vis_comb = Utils.draw_posed_3d_box(K, img=vis_comb, ob_in_cam=center_pose, bbox=bbox,
                                                   line_color=(0, 255, 255))
                vis_comb = Utils.draw_xyz_axis(vis_comb, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3,
                                               transparency=0, is_input_rgb=True)

                # add contact text 
                if self.cfg.cont_out_dim > 0:
                    cont_gt = batch['contact_dist_gt'][bid, i, [22, 23+15]]
                    cont_text = f'lh: {cont_gt[0]:.3f}, rh: {cont_gt[1]:.3f}'
                    cv2.putText(vis_gt, cont_text, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
                    # add to vis_pred as well 
                    cont_pred = output['contact'].reshape(B, T, -1)[bid, i]
                    cont_text = f'lh: {cont_pred[0]:.3f}, rh: {cont_pred[1]:.3f}'
                    cv2.putText(vis_pred, cont_text, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
                pose_comb = np.concatenate([np.concatenate([vis_input, vis_pred], 1),
                                            np.concatenate([vis_gt, vis_comb], 1)], axis=0)
                comb = np.concatenate([comb, pose_comb], axis=1)

                # visualize PC
                pc_ab, pc_colors = self.visualize_xyz_map(bid, i, xyzA, xyzB)
                log_dict[f'{key}_xyz_{i}'] = wandb.Object3D(np.concatenate([pc_ab, pc_colors], 1),
                                                            caption=f'xyz: red-A green-B ')

                log_dict[f'{key}_all_{i}'] = wandb.Image(comb, caption='top-A bottom-B')
                outfile = osp.join(self.exp_dir, f'vis/{key}_xyz_{i}.ply')
                os.makedirs(osp.dirname(outfile), exist_ok=True)
                trimesh.PointCloud(pc_ab, colors=pc_colors).export(outfile)
                outfile = osp.join(self.exp_dir, f'vis/{key}_all_{i}.png')
                Image.fromarray(comb).save(outfile)

            if not self.cfg.no_wandb:
                wandb.log(log_dict, step=self.train_state.step)
            end = time.time()
            print(f'Step {self.train_state.step} {key} vis uploading finished after {end - start} seconds')

        if ret_dict:
            return rot, rot_delta_gt, trans_delta_gt, trans_delta_pred, output
        return rot, rot_delta_gt, trans_delta_gt, trans_delta_pred

    def prepare_input_viz(self, batch, cfg):
        rgbsA = (batch['render_rgbs'].cpu().numpy() * 255).astype(np.uint8)  # (B, T, 3, H, W)
        rgbsB = (batch['input_rgbs'].cpu().numpy() * 255).astype(np.uint8)  # (B, T, 3, H, W)
        xyzA = (batch['render_xyz'][:, :, :3].cpu().numpy())
        xyzB = (batch['input_xyz'][:, :, :3].cpu().numpy())
        if batch['input_xyz'].shape[2] in [5, 6]:
            maskA = (batch['render_xyz'][:, :, 3:].cpu().numpy() * 255).astype(np.uint8)
            maskB = (batch['input_xyz'][:, :, 3:].cpu().numpy() * 255).astype(np.uint8)
        elif cfg.mask_encode_type == 'one-channel':
            maskA = ((batch['render_xyz'][:, :, 3:].cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
            maskB = ((batch['input_xyz'][:, :, 3:].cpu().numpy() + 1) / 2 * 255).astype(np.uint8)
        else:
            maskA, maskB = None, None
        return maskA, maskB, rgbsA, rgbsB, xyzA, xyzB

    def compute_abspose(self, B, batch, cfg, poseA, rot, rot_delta_gt, trans_delta_gt, trans_delta_pred, out_dict=None):
        b, t = poseA.shape[:2]
        if self.cfg['loss_type'] == 'l1-abs':
            rot_pred = so3_exp_map(rot).permute(0, 2, 1)
            trans_pred = trans_delta_pred
            B_in_cams = torch.zeros_like(poseA)
            B_in_cams[:, :, :3, :3] = rot_pred.reshape(b, t, 3, 3)
            B_in_cams[:, :, :3, 3] = trans_pred.reshape(b, t, 3)
        elif self.cfg['loss_type'] in ['l1-abs-delta', 'l2-abs-delta']:
            if self.cfg.rot_rep == 'axis_angle':
                rot_pred = so3_exp_map(out_dict['rot_abs']).permute(0, 2, 1)
            elif self.cfg.rot_rep == '6d':
                rot_pred = geom_utils.rot6d_to_rotmat(out_dict['rot_abs'])
            else:
                raise NotImplementedError
            trans_pred = out_dict['trans_abs']
            B_in_cams = torch.zeros_like(poseA)
            B_in_cams[:, :, :3, :3] = rot_pred.reshape(b, t, 3, 3)
            B_in_cams[:, :, :3, 3] = trans_pred.reshape(b, t, 3)
        elif cfg['loss_type'] in ['l1-absrot-delta', 'l1-absrot-delta-hum', 'l2-absrot-delta-humabs', 'l1-absrot-delta-humabs']:
            # rot from abs, trans from delta
            B_in_cams = self.abspose_from_relative(batch, cfg, poseA, rot, trans_delta_pred)
            if self.cfg.rot_rep == 'axis_angle':
                rot_pred = so3_exp_map(out_dict['rot_abs']).permute(0, 2, 1)
            elif self.cfg.rot_rep == '6d':
                rot_pred = geom_utils.rot6d_to_rotmat(out_dict['rot_abs'])
            else:
                raise NotImplementedError
            B_in_cams[:, :, :3, :3] = rot_pred.reshape(b, t, 3, 3)
        else:
            B_in_cams = self.abspose_from_relative(batch, cfg, poseA, rot, trans_delta_pred)
        rot_delta_gt_rot = so3_exp_map(rot_delta_gt * cfg['rot_normalizer']).permute(0, 2, 1)
        B_in_cams_gt = Utils.egocentric_delta_pose_to_pose(poseA.reshape(-1, 4, 4),
                                                           trans_delta=trans_delta_gt * batch['mesh_diameter'].reshape((-1, 1)) / 2.,
                                                           rot_mat_delta=rot_delta_gt_rot).reshape(B, t, 4, 4)  # (BT, 4, 4)

        return B_in_cams, B_in_cams_gt

    def abspose_from_relative(self, batch, cfg, poseA, rot, trans_delta_pred):
        "compute abs pose from relative pose prediction"
        b, t = poseA.shape[:2]
        trans_delta_final = trans_delta_pred * batch['mesh_diameter'].reshape((-1, 1)) / 2.  # undo normalization
        rot_delta_final = so3_exp_map(rot * cfg['rot_normalizer']).permute(0, 2, 1)
        B_in_cams = Utils.egocentric_delta_pose_to_pose(poseA.reshape(-1, 4, 4), trans_delta=trans_delta_final,
                                                        rot_mat_delta=rot_delta_final).reshape(b, t, 4, 4)
        return B_in_cams

    def compute_smpl_verts(self, batch, out_dict):
        "compute smpl verts for GT and prediction"
        betas, pred_smpl_pose, pred_smpl_r, pred_smpl_t = self.smpl_params_from_pred(batch, out_dict)

        verts_smpl, jtrs_pr, _, _, jts_rot_pr = self.smpl_male(pose72to156(pred_smpl_pose), betas, pred_smpl_t, ret_glb_rot=True)
        verts_smpl_gt, jtrs_gt, _, _, jts_rot_gt = self.smpl_male(batch['smpl_poses_gt'].reshape(-1, 156), betas,
                                                                   batch['smpl_transl_gt'].reshape(-1, 3), ret_glb_rot=True)
        return verts_smpl_gt, verts_smpl, jtrs_gt, jtrs_pr, pred_smpl_r, pred_smpl_t, jts_rot_pr, jts_rot_gt

    def smpl_params_from_pred(self, batch, out_dict):
        "compute SMPL parameters from prediction, return in shape (BT, ...)"
        J, bid = 24, 0
        clip_len = self.cfg.clip_len
        bs = len(batch['nlf_transl'])
        if self.cfg.loss_type in ['l2-absrot-delta-humabs', 'l1-absrot-delta-humabs']:
            # predict abs pose already
            pred_smpl_r = out_dict['body_rotmat']
            pred_smpl_t = out_dict['body_transl']
        else:
            # additional visualization for human as well
            nlf_poses = batch['nlf_rotmat'].reshape(-1, J, 3, 3)  # B, T, J, 3, 3,
            pred_smpl_t = batch['nlf_transl'].reshape(-1, 3) + out_dict['hum_trans']
            delta_pr_r = geom_utils.rot6d_to_rotmat(out_dict['hum_pose'].reshape(-1, 6)).reshape(-1, J, 3, 3)

            pred_smpl_r = delta_pr_r @ nlf_poses
        pred_smpl_pose = geom_utils.rotation_matrix_to_angle_axis(pred_smpl_r.reshape(-1, 3, 3)).reshape(-1, J * 3)
        if 'hum_shape' in out_dict:
            betas = out_dict['hum_shape'] + batch['betas_nlf'].reshape(-1, 10)
        else:
            betas = batch['betas_nlf'].reshape(-1, 10) # use predicted NLF betas
        return betas, pred_smpl_pose, pred_smpl_r, pred_smpl_t

    @staticmethod
    def visualize_rgbm(batch, bid, i, maskA, maskB, rgbsA, rgbsB):
        rgba = rgbsA[bid, i].transpose(1, 2, 0)
        rgbb = rgbsB[bid, i].transpose(1, 2, 0)
        ab = np.concatenate([rgba, rgbb], axis=0)
        # log mask as well
        if batch['input_xyz'].shape[2] == 5:
            maska = maskA[bid, i].transpose(1, 2, 0)  # already (H, W, 2)
            maskb = maskB[bid, i].transpose(1, 2, 0)
            comb = np.concatenate([np.concatenate([maska, np.zeros_like(maska[:, :, 0:1])], -1),
                                   np.concatenate([maskb, np.zeros_like(maskb[:, :, 0:1])], -1)], 0)
            comb = np.concatenate([ab, comb], axis=1)
        elif batch['input_xyz'].shape[2] == 4:
            # one channel
            maska_h = maskA[bid, i, 0][:, :, None].repeat(3, -1)
            maska_o = maskA[bid, i, 0][:, :, None].repeat(3, -1)
            maskb_h = maskB[bid, i, 0][:, :, None].repeat(3, -1)
            maskb_o = maskB[bid, i, 0][:, :, None].repeat(3, -1)
            comb = np.concatenate([maska_h, maskb_h], axis=0)
            comb = np.concatenate([ab, comb], axis=1)
        elif batch['input_xyz'].shape[2] == 6:
            # do nothing
            maska = maskA[bid, i].transpose(1, 2, 0) # already (H, W, 3)
            maskb = maskB[bid, i].transpose(1, 2, 0)
            comb = np.concatenate([maska, maskb], axis=0)
            comb = np.concatenate([ab, comb], axis=1)
        else:
            comb = ab
        return comb, rgba, rgbb

    @staticmethod
    def visualize_xyz_map(bid, i, xyzA, xyzB):
        mask_xyza = np.abs(xyzA[bid, i, 2]) > 0.001  # avoid all zeros
        pc_a = xyzA[bid, i].transpose(1, 2, 0)[mask_xyza].reshape((-1, 3))
        mask_xyzb = np.abs(xyzB[bid, i, 2]) > 0.001
        pc_b = xyzB[bid, i].transpose(1, 2, 0)[mask_xyzb].reshape((-1, 3))
        pc_ab = np.concatenate([pc_a, pc_b], axis=0)
        red = np.array([[255, 0, 0]]).repeat(len(pc_a), 0)
        green = np.array([[0, 255, 0]]).repeat(len(pc_b), 0)
        pc_colors = np.concatenate([red, green], 0)
        return pc_ab, pc_colors

def main2():
    # 1. Create the base config from the structured dataclass
    # This holds all the defaults.
    cfg = get_config()

    trainer = Trainer(cfg)
    trainer.train()


def get_config():
    base_conf = OmegaConf.structured(TrainTemporalRefinerConfig)
    cfg_cli = OmegaConf.from_cli()
    # 2. Load the config from the YAML file
    # This holds our overrides.
    if 'config' in cfg_cli:
        file_conf = OmegaConf.load(cfg_cli.config)
        # 3. Merge the two configurations.
        # The values in `file_conf` will overwrite the defaults in `base_conf`.
        cfg: TrainTemporalRefinerConfig = OmegaConf.merge(base_conf, file_conf)
        print("Overriding config from file", cfg_cli.config)
    else:
        cfg = base_conf
    # merge with command line args
    cfg = OmegaConf.merge(cfg, cfg_cli)
    return cfg


if __name__ == "__main__":
    main2()

