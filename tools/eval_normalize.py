# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
usage: python tools/eval_saved.py split_file=splits/demo-behave.json result_dir=output/opt/cari4d-release+step031397_demo-hy3d3-optv2
"""
import glob
import json
import pickle
import time

import cv2
import sys, os

import imageio
import os.path as osp
import pickle as pkl

import trimesh

sys.path.append(os.getcwd())
import torch
from tqdm import tqdm
import numpy as np
from tools.eval_base import ModelEvaluator
from behave_data.const import BEHAVE_ROOT
from behave_data.utils import load_template, get_render_template_path_from_seq
from behave_data.const import _sub_gender, get_hy3d_mesh_file
from learning.training.trainer import Trainer, get_config
import Utils
from lib_smpl import get_smpl, pose72to156
from learning.training.training_config import TrainTemporalRefinerConfig


def chamfer_distance(s1, s2, w1=1., w2=1.):
    """
    :param s1: B x N x 3
    :param s2: B x M x 3
    :param w1: weight for distance from s1 to s2
    :param w2: weight for distance from s2 to s1
    this requires pytorch3d specific installation? 
    """
    from pytorch3d.ops import knn_points

    assert s1.is_cuda and s2.is_cuda
    closest_dist_in_s2 = knn_points(s1, s2, K=1)
    closest_dist_in_s1 = knn_points(s2, s1, K=1)

    return (closest_dist_in_s2.dists**0.5 * w1).squeeze(-1), (closest_dist_in_s1.dists**0.5 * w2).squeeze(-1)

class NormalizedEvaluator(ModelEvaluator):
    @torch.no_grad()
    def eval_1seq(self, cfg: TrainTemporalRefinerConfig, data_gt_chunk, data_pr_chunk, errors_all, smoothnet_obj, smoothnet_smpl, trainer, smooth=False, smooth_smplt=False):
        "normalize and then compute v2v, mpjpe, etc. "
        if smooth:
            print("SMPL pose and object pose smoothed!")
            pose_pr_smooth = smoothnet_obj.smooth_seq(data_pr_chunk['pose_abs'], data_pr_chunk['frames'])
            data_pr_chunk['pose_abs'] = pose_pr_smooth # update the prediction

            smpl_smooth = smoothnet_smpl.smooth_seq(data_pr_chunk['smpl_pose'],
                                                    data_pr_chunk['smpl_t'],
                                                    data_pr_chunk['betas'],
                                                    data_pr_chunk['frames'])

            data_pr_chunk['smpl_pose'] = torch.from_numpy(smpl_smooth['poses']).cuda()  # update the pose
            data_pr_chunk['betas'] = torch.from_numpy(smpl_smooth['betas']).cuda()  # update the betas
            if not smooth_smplt:
                print("Not smoothing SMPL translation")
                smpl_smooth['trans'] = data_pr_chunk['smpl_t'].cpu().numpy()  # use the original translation instead of smoothed one
            data_pr_chunk['smpl_t'] = torch.from_numpy(smpl_smooth['trans']).cuda()  # update the translation as well

        else:
            pose_pr_smooth = data_pr_chunk['pose_abs']
            smpl_smooth = {
                'poses': data_pr_chunk['smpl_pose'].cpu().numpy(),
                'trans': data_pr_chunk['smpl_t'].cpu().numpy(),
                'betas': data_pr_chunk['betas'].cpu().numpy(),
            }

        # get the 1st frame SMPL
        # compute GT human + object templates
        seq_name = data_pr_chunk['frames'][0].split('/')[0]
        gender = _sub_gender[seq_name.split('_')[1]]
        smpl_model = get_smpl(gender, True).cuda()
        verts_smpl, jtrs_pr, _, _, jts_rot_pr = smpl_model(torch.from_numpy(pose72to156(smpl_smooth['poses'])).cuda(),
                                                                   torch.from_numpy(smpl_smooth['betas']).cuda(),
                                                                   torch.from_numpy(smpl_smooth['trans']).cuda(),
                                                                   ret_glb_rot=True)
        verts_smpl_gt, jtrs_gt, _, _, jts_rot_gt = smpl_model(data_gt_chunk['smpl_pose'],
                                                                      data_gt_chunk['betas'],
                                                                      data_gt_chunk['smpl_t'],
                                                                      ret_glb_rot=True)
        # now compute procrustes alignment from pr to GT using the vertices and trimehs alignment
        if cfg.align2gt:
            # align the first frame only 
            mat, transformed, cost1 = trimesh.registration.procrustes(verts_smpl.cpu().numpy()[0], verts_smpl_gt.cpu().numpy()[0])

            # apply transformation to the obj pose 
            align2gt = torch.from_numpy(mat).cuda().float()
            pose_pr_smooth = torch.stack([torch.matmul(align2gt, p) for p in pose_pr_smooth], 0)
            # now apply to human as well 
            # align human vertices, joints, rotations, and smpl_t  
            align2gt_batch = align2gt[None].repeat(len(verts_smpl), 1, 1)
            verts_smpl = torch.matmul(verts_smpl, align2gt_batch[:, :3, :3].permute(0, 2, 1)) + align2gt_batch[:, :3, 3][:, None]
            jtrs_pr = torch.matmul(jtrs_pr, align2gt_batch[:, :3, :3].permute(0, 2, 1)) + align2gt_batch[:, :3, 3][:, None]
            jts_rot_pr_glb = torch.matmul(jts_rot_pr[:, 0], align2gt_batch[:, :3, :3].permute(0, 2, 1))
            jts_rot_pr[:, 0, :3, :3] = jts_rot_pr_glb
            data_pr_chunk['smpl_t'] = torch.matmul(data_pr_chunk['smpl_t'][:, None], align2gt_batch[:, :3, :3].permute(0, 2, 1)) + align2gt_batch[:, :3, 3][:, None]
            # squeeze smpt 1st dim
            data_pr_chunk['smpl_t'] = data_pr_chunk['smpl_t'].squeeze(1)
        
        self.add_obj_errors(data_gt_chunk, errors_all, pose_pr_smooth)

        # compute GT object
        N = 8196
        L = len(data_pr_chunk['pose_abs'])
        seq, _ = data_pr_chunk['frames'][0].split('/')
        obj_name = seq.split('_')[2]
        if 'Date' in seq_name or 'ICap' in seq_name:
            # BEHAVE or intercap dataset
            obj_template = load_template(obj_name, cent=False, dataset_path=BEHAVE_ROOT)
        else:
            # Other datasets
            template_file = get_render_template_path_from_seq(seq_name)
            obj_template = trimesh.load(template_file, process=False)
        cent_gt_template = np.mean(obj_template.vertices, 0)
        obj_template.vertices = obj_template.vertices - cent_gt_template 
        obj_pts_gt = torch.from_numpy(obj_template.sample(N)).float().cuda()[None].expand(L, -1, -1)
        if cfg.use_hy3d:
            # load from hy3d mesh
            file_hy3d = get_hy3d_mesh_file(seq_name, meshes_root='data/cari4d-demo/meshes')
            obj_temp_pr = trimesh.load(file_hy3d, process=False)
            obj_pts_pr = torch.from_numpy(obj_temp_pr.sample(N)).float().cuda()[None].expand(L, -1, -1)
            print("Using hy3d mesh from ", file_hy3d)
        else:
            obj_pts_pr = obj_pts_gt # same as GT
            print("Using GT mesh")
        obj_pts_pr = torch.matmul(obj_pts_pr, pose_pr_smooth[:, :3, :3].permute(0, 2, 1)) + pose_pr_smooth[:, :3, 3][:, None]
        obj_pts_gt = torch.matmul(obj_pts_gt, data_gt_chunk['pose_abs'][:, :3, :3].permute(0, 2, 1)) + data_gt_chunk['pose_abs'][:, :3, 3][:, None]

        # sample surface points to do normalization
        hum_pts_gt = torch.from_numpy(np.stack([trimesh.Trimesh(v, smpl_model.faces).sample(N) for v in verts_smpl_gt.cpu().numpy()])).float().cuda()
        hum_pts_pr = torch.from_numpy(np.stack([trimesh.Trimesh(v, smpl_model.faces).sample(N) for v in verts_smpl.cpu().numpy()])).float().cuda()
        pts_gt_comb = torch.cat([hum_pts_gt, obj_pts_gt], 1)
        pts_pr_comb = torch.cat([hum_pts_pr, obj_pts_pr], 1)
        undo_normalize = not cfg.eval_normalize 
        if not undo_normalize:
            cent_gt = torch.mean(pts_gt_comb, 1)
            radius_gt = torch.sqrt(torch.max(torch.sum((pts_gt_comb - cent_gt[:, None]) ** 2, -1), -1)[0])  # T,
        else:
            cent_gt = torch.zeros(pts_gt_comb.shape[0], 3).cuda()
            radius_gt = torch.ones(pts_gt_comb.shape[0]).cuda() * 0.5 
        pts_gt_norm = (pts_gt_comb - cent_gt[:, None]) / (radius_gt[:, None, None] * 2)
        pts_pr_norm = (pts_pr_comb - cent_gt[:, None]) / (radius_gt[:, None, None] * 2)

        # now normalize everything
        verts_smpl = (verts_smpl - cent_gt[:, None]) / (radius_gt[:, None, None] * 2)
        verts_smpl_gt = (verts_smpl_gt - cent_gt[:, None]) / (radius_gt[:, None, None] * 2)
        jtrs_pr = (jtrs_pr - cent_gt[:, None]) / (radius_gt[:, None, None] * 2)
        jtrs_gt = (jtrs_gt - cent_gt[:, None]) / (radius_gt[:, None, None] * 2)

        J_eval = 22 # only eval the first 23 joints now
        jtrs_pr_rela = jtrs_pr[:, :J_eval] - jtrs_pr[:, 0:1]
        jtrs_gt_rela = jtrs_gt[:, :J_eval] - jtrs_gt[:, 0:1]
        v2v = torch.sum((verts_smpl_gt - verts_smpl) ** 2, -1).sqrt().mean(-1).cpu().numpy()
        mpjpe_all = torch.sum((jtrs_pr_rela - jtrs_gt_rela) ** 2, -1).sqrt().reshape(-1, J_eval).cpu().numpy()
        mpjpe = mpjpe_all.mean(-1)
        smpl_t = torch.sum((data_gt_chunk['smpl_t'] - data_pr_chunk['smpl_t']) ** 2, -1).sqrt().cpu().numpy()
        mpjae_all = Utils.geodesic_distance_batch(jts_rot_pr[:, :J_eval].reshape(-1, 3, 3), jts_rot_gt[:, :J_eval].reshape(-1, 3, 3)).reshape(-1, J_eval).cpu().numpy()  # T*J
        mpjae = mpjae_all.mean(-1)
        errors_all['v2v'].append(v2v)
        errors_all['mpjpe'].append(mpjpe)
        errors_all['smpl_t'].append(smpl_t)
        errors_all['mpjae'].append(mpjae)
        errors_all['mpjae_all'].append(mpjae_all)
        errors_all['mpjpe_all'].append(mpjpe_all)

        # evaluate acceleration error of translation of human and object, all tensors have shape (T, ...) where T is the seq length
        acc_t_gt = data_gt_chunk['smpl_t'][:-2] - 2 * data_gt_chunk['smpl_t'][1:-1] + data_gt_chunk['smpl_t'][2:]
        acc_t_pr = data_pr_chunk['smpl_t'][:-2] - 2 * data_pr_chunk['smpl_t'][1:-1] + data_pr_chunk['smpl_t'][2:]
        acc_t = torch.sum((acc_t_pr - acc_t_gt) ** 2, -1).sqrt().cpu().numpy()
        # append 1st to front and last to end
        acc_t = np.concatenate([acc_t[0:1], acc_t, acc_t[-1:]], 0)
        errors_all['acc_ht'].append(acc_t)
        acc_ot_gt = data_gt_chunk['pose_abs'][:-2, :3, 3] - 2 * data_gt_chunk['pose_abs'][1:-1, :3, 3] + data_gt_chunk['pose_abs'][2:, :3, 3]
        acc_ot_pr = data_pr_chunk['pose_abs'][:-2, :3, 3] - 2 * data_pr_chunk['pose_abs'][1:-1, :3, 3] + data_pr_chunk['pose_abs'][2:, :3, 3]
        acc_o = torch.sum((acc_ot_pr - acc_ot_gt) ** 2, -1).sqrt().cpu().numpy()
        # append 1st to front and last to end
        acc_o = np.concatenate([acc_o[0:1], acc_o, acc_o[-1:]], 0)
        errors_all['acc_ot'].append(acc_o)

        # also add acc for human joints and jitters 
        acc_hj_gt = jtrs_gt[:-2, :J_eval] - 2 * jtrs_gt[1:-1, :J_eval] + jtrs_gt[2:, :J_eval]
        acc_hj_pr = jtrs_pr[:-2, :J_eval] - 2 * jtrs_pr[1:-1, :J_eval] + jtrs_pr[2:, :J_eval]
        acc_hj = torch.sum((acc_hj_pr - acc_hj_gt) ** 2, -1).sqrt().mean(-1).cpu().numpy()
        # append 1st to front and last to end
        acc_hj = np.concatenate([acc_hj[0:1], acc_hj, acc_hj[-1:]], 0)
        errors_all['acc_hj'].append(acc_hj)
        # jitter as the pr in 3rd derivative
        jitter_hj_pr = jtrs_pr[3:, :J_eval] - 3 * jtrs_pr[2:-1, :J_eval] + 3 * jtrs_pr[1:-2, :J_eval] - jtrs_pr[:-3, :J_eval]
        jitter_hj = torch.norm(jitter_hj_pr, dim=-1).mean(-1).cpu().numpy()
        # repeat 1st element twice to front and last element once to back
        jitter_hj = np.concatenate([jitter_hj[0:1].repeat(2), jitter_hj, jitter_hj[-1:]], 0)
        errors_all['jitter_hj'].append(jitter_hj)
        jitter_ot = torch.norm(data_pr_chunk['pose_abs'][3:, :3, 3] - 3 * data_pr_chunk['pose_abs'][2:-1, :3, 3] + 3 * data_pr_chunk['pose_abs'][1:-2, :3, 3] - data_pr_chunk['pose_abs'][:-3, :3, 3], dim=-1).cpu().numpy()
        # repeat 1st element twice to front and last element once to back
        jitter_ot = np.concatenate([jitter_ot[0:1].repeat(2), jitter_ot, jitter_ot[-1:]], 0)
        errors_all['jitter_ot'].append(jitter_ot)

        # evaluate contacts if given 
        if 'contact_dist' in data_pr_chunk:
            contact_dist_gt = data_gt_chunk['contact_dist']
            contact_dist_pr = data_pr_chunk['contact_dist']
            contact_dim = contact_dist_gt.shape[-1] # check the dimension and select only the distance part 
            if contact_dim == 24 + 8:
                # only first 8 dims are the really contact distance
                contact_dist_gt = contact_dist_gt[:, :8]
                contact_dist_pr = contact_dist_pr[:, :8]
            dist_mask = torch.ones_like(contact_dist_gt).to(contact_dist_gt.device) if cfg.contact_dist_thres < 10 else contact_dist_gt < cfg.contact_dist_thres
            err_contact = torch.abs(contact_dist_gt - contact_dist_pr).cpu().numpy() * dist_mask.cpu().numpy()
            errors_all['cont_all'].append(err_contact) # (BT, J)
        elif 'contact_logits' in data_pr_chunk:
            # evaluate contact accuracy
            contact_logits_gt = data_gt_chunk['contact_logits']
            contact_logits_pr = data_pr_chunk['contact_logits']
            if cfg.cont_out_type == 'binary':
                contact_acc = (contact_logits_pr > 0).float() == (contact_logits_gt < cfg.cont_mask_thres).float()
            elif cfg.cont_out_type == 'distance':
                # BT, 2 
                contact_acc = (contact_logits_pr < cfg.cont_mask_thres).float() == (contact_logits_gt < cfg.cont_mask_thres).float()
            else:
                raise ValueError(f'Invalid contact output type: {cfg.cont_out_type}')
            errors_all['cont_all'].append(contact_acc.float().mean(-1).cpu().numpy())
            print("Contact accuracy: ", contact_acc.float().mean().cpu().numpy())
        else:
            errors_all['cont_all'].append(np.zeros((len(mpjpe_all),))) # dummy result  

        CUBE_SIDE_LEN = 1.0
        th_list = [CUBE_SIDE_LEN/100]

        chamf, scores = self.compute_fscores(pts_gt_norm, pts_pr_norm, th_list)
        errors_all['cd_comb'].append(chamf.cpu().numpy())
        errors_all['fscore_comb'].append(scores[:, 0, 0].cpu().numpy())

        chamf, scores = self.compute_fscores(pts_gt_norm[:, :N], pts_pr_norm[:, :N], th_list)
        errors_all['cd_hum'].append(chamf.cpu().numpy())
        errors_all['fscore_hum'].append(scores[:, 0, 0].cpu().numpy())
        chamf, scores = self.compute_fscores(pts_gt_norm[:, N:], pts_pr_norm[:, N:], th_list)
        errors_all['cd_obj'].append(chamf.cpu().numpy())
        errors_all['fscore_obj'].append(scores[:, 0, 0].cpu().numpy())

        # evaluate hand contact accuracy 
        jtrs_hand_gt = jtrs_gt[:, [22, 23+15]]
        jtrs_hand_pr = jtrs_pr[:, [22, 23+15]] 
        # compute their distance to object points 
        dist_hand_gt = chamfer_distance(jtrs_hand_gt, pts_gt_norm[:, N:])[0]
        dist_hand_pr = chamfer_distance(jtrs_hand_pr, pts_pr_norm[:, N:])[0]
        cont_thres = 0.015 # contact distance threshold 
        cont_acc = (dist_hand_pr < cont_thres).float() == (dist_hand_gt < cont_thres).float()
        errors_all['cont_hand'].append(cont_acc.float().mean(-1).cpu().numpy())

    def compute_fscores(self, pts_gt_norm, pts_pr_norm, th_list):
        "return: (L, 3, len(th_list))"
        L = len(pts_gt_norm)
        d1, d2 = chamfer_distance(pts_gt_norm, pts_pr_norm)
        chamf = d1.mean(-1) + d2.mean(-1)
        scores = torch.zeros(L, 3, len(th_list)).cuda()
        for i, th in enumerate(th_list):
            recall = torch.sum(d2 < th, -1) / d2.shape[-1]
            precision = torch.sum(d1 < th, -1) / d1.shape[-1]

            mask = recall + precision == 0
            fscore = 2 * recall * precision / (recall + precision)
            fscore[mask] = 0.0
            scores[:, :, i] = torch.stack([fscore, precision, recall], -1)
        return chamf, scores

    def get_error_keys(self):
        ""
        err_keys = ['rot', 'transl', 'mpjpe', 'v2v', 'mpjae', 'smpl_t', 'cd_comb', 'fscore_comb', 
        'cd_hum', 'fscore_hum', 'cd_obj', 'fscore_obj', 'mpjae_all', 'mpjpe_all', 'cont_all', 
        'acc_ht', 'acc_ot', 'acc_hj', 'jitter_hj', 'jitter_ot', 'cont_hand']
        return err_keys

    def get_summary_string(self, cfg, errors_all, seqs_test, trainer):
        "add more metrics"
        ss = (
            f"{cfg.exp_name}/{osp.basename(trainer.ckpt_file)} {len(seqs_test)} seqs {len(errors_all['rot'])} examples: rot={np.mean(errors_all['rot']) * 180 / np.pi:.4f}, "
            f"trans={np.mean(errors_all['transl']):.4f}, "
            f"v2v={np.mean(errors_all['v2v']):.4f}, mpjae={np.mean(errors_all['mpjae']) * 180 / np.pi:.4f}, mpjpe={np.mean(errors_all['mpjpe']):.4f}, "
            f"smpl_t={np.mean(errors_all['smpl_t']):.4f}, ")
        return ss

    def evaluate(self, cfg):
        ""
        # add additional configs
        if cfg.align2gt:
            cfg.identifier += '-align2gt'
        if cfg.eval_input:
            cfg.identifier += '-fp-nlf'

        # get pth files
        cfg.result_dir = cfg.result_dir[:-1] if cfg.result_dir[-1] == '/' else cfg.result_dir
        save_name_old = osp.basename(cfg.result_dir)
        save_name = f'{save_name_old}{cfg.identifier}'

        err_keys = self.get_error_keys()
        errors_all = {k:[] for k in err_keys}

        frames_all = []
        seqs_test = []

        seqs = json.load(open(cfg.split_file))['test']
        count = 0
        for seq in tqdm(seqs):
            pth_file = f'{cfg.result_dir}/{seq}.pth'
            if not osp.isfile(pth_file):
                print(f'{pth_file} not found, skipping')
                continue
            print(f'Evaluating {pth_file}...')
            count += 1
            pth_data = torch.load(pth_file, map_location='cuda')
            data_pr = pth_data['pr']
            if cfg.eval_input:
                data_pr = pth_data['data_in'] # evaluate input

            gt_pth_file = f'output/gt/{seq}.pth'
            data_gt = torch.load(gt_pth_file, map_location='cuda', weights_only=False)['gt']
            # cut data_pr
            for k, v in data_pr.items():
                if isinstance(v, torch.Tensor):
                    data_pr[k] = v[:len(data_gt['frames'])]

            self.eval_1seq(cfg, data_gt, data_pr, errors_all, None, None, None, False, smooth_smplt=cfg.smooth_smplt)
            frames_all.extend(data_pr['frames'])
            seqs_test.append(osp.basename(pth_file).split('.')[0])
        if len(errors_all[err_keys[0]]) == 0:
            print("No errors found, skipping results accumulation")
            return
        outfile = self.save_output(cfg, errors_all, frames_all, save_name, seqs_test, cfg)


def main():
    cfg = get_config()
    evaluator = NormalizedEvaluator(cfg)
    evaluator.evaluate(cfg)

if __name__ == '__main__':
    main()