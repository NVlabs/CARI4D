# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
example commands:

python tools/eval_base.py config=learning/configs/chair-dinov2-abspose.yml exp_name=chair-dinov2-abs6d split_file=splits/date03.json pose_init_type=random-keyframes
no_wandb=False subtract_transl=True render_root=/home/xianghuix/datasets/foundpose_train/behave-fppose

chair-dinov2-abs6d/step000600.pth 8 seqs 11328 examples: rot=3.94294, trans=0.11807
chair-dinov2-abs6d-norm/step001200.pth 8 seqs 11328 examples: rot=1.01162, trans=0.03347
chair-dinov2-abs6d-norm/step000600.pth 8 seqs 11328 examples: rot=2.32023, trans=0.05591
chair-dinov2-abs6d-norm9d/step000600.pth 8 seqs 11328 examples: rot=0.96399, trans=0.05406

chair-dinov2-dt-fppose-l1/step001200.pth 8 seqs 11328 examples: rot=0.72732, trans=0.04186
chair-dinov2-nospatial-fp/step000900.pth 8 seqs 11328 examples: rot=0.86969, trans=0.02051 (3 attn layer)
chair-dinov2-dt-fppose/step001100.pth 8 seqs 11328 examples: rot=1.16438, trans=0.06162

chair-dinov2-rot80-dt/step001800.pth 8 seqs 11328 examples: rot=1.50219, trans=0.07342
r80-dt-1ch/step001700.pth 8 seqs 11328 examples: rot=1.28175, trans=0.06725
r80-dt-nomask/step001000.pth 8 seqs 11328 examples: rot=1.40369, trans=0.06447
r80-dt-occ/step000900.pth 8 seqs 11328 examples: rot=1.56501, trans=0.06953
"""
import glob
import json
import pickle
import joblib
import time

import sys, os

import os.path as osp

sys.path.append(os.getcwd())
import torch
from tqdm import tqdm
import numpy as np

from learning.training.trainer import Trainer, get_config
import Utils
from tools import geometry_utils as geom_utils
from datetime import datetime
from behave_data.const import get_test_view_id

class ModelEvaluator:
    def __init__(self, cfg):
        # some fixed config
        self.cfg = cfg
        cfg.no_wandb = True
        cfg.job = 'test'  # no shuffle
        if cfg.ckpt_file is None:
            cfg.ckpt_file = 'none'

    @torch.no_grad()
    def evaluate(self, cfg):
        ""
        cfg.num_workers = 36

        # add additional configs
        if cfg.align2gt:
            cfg.identifier += '-align2gt'
        if cfg.eval_input:
            cfg.identifier += '-fp-nlf'

        d = json.load(open(cfg.split_file))
        seqs_train, seqs_test = d['train'], d['test']
        trainer = Trainer(cfg)  # this will load a pre-trained checkpoint
        dataloader_val = trainer.val_dataloader
        trainer.model.eval()

        save_name = self.get_save_name(cfg, trainer)
        outfiles = glob.glob(f'results/{osp.splitext(osp.basename(cfg.split_file))[0]}+{save_name}*.json')
        if len(outfiles) >0 and not cfg.redo:
            print(f'{outfiles} done, skipping')
            return
        print("start evaluating, results will be saved to {}".format(f'results/{osp.splitext(osp.basename(cfg.split_file))[0]}+{save_name}*.json'))

        preds_rot, preds_trans = [], []
        input_rot, input_trans = [], []
        pose_pred = []
        errors_rot, errors_trans = 0, 0
        rot_list, trans_list = [], []
        v2vs, mpjpes, mpjaes, smpl_t_err = 0, 0, 0, 0
        v2v_errs, mpjpes_errs, mpjaes_errs, smpl_t_errs = [], [], [], []
        jaes, jpes, raw_jaes = [], [], [] # raw joint angle errors

        run_smooth = cfg.run_smooth
        smoothnet_obj =  None
        smoothnet_smpl =  None

        keys = ['pose_abs', 'smpl_pose', 'smpl_t', 'frames', 'betas', 'uncertainty', 'pose_symm', 'contact_logits']
        data_gt, data_pr, data_in = {k:[] for k in keys}, {k:[] for k in keys}, {k:[] for k in keys}
        count = 0
        frames = []
        if cfg.eval_input:
            print("Evaluating input FP + NLF!")
            assert 'fp-nlf' in cfg.identifier, "Identifier must contain 'fp-nlf' to evaluate input FP + NLF!"
        for bs_id, batch in enumerate(tqdm(dataloader_val)):
            #
            time_start = time.time()
            rot, rot_delta_gt, trans_delta_gt, trans_delta_pred, out_dict = trainer.forward_batch(batch, cfg, trainer.model, ret_dict=True, vis=False)

            poseA = batch['pose_perturbed']
            poseB = batch['pose_gt']
            B, T = poseA.shape[:2]
            time_end = time.time()
            B_in_cams, B_in_cams_gt = trainer.compute_abspose(poseA.shape[0], batch, cfg, poseA, rot,
                                                              rot_delta_gt,
                                                              trans_delta_gt, trans_delta_pred, out_dict)
            if cfg.use_intermediate:
                B_in_cams = trainer.abspose_from_relative(batch, cfg, poseA, rot, trans_delta_pred)
            B_in_cams = B_in_cams.reshape(-1, 4, 4)  # Backward compatible
            # compute error

            files = [batch['image_files'][ti][bi] for bi in range(B) for ti in range(T)]

            data_pr['pose_abs'].append(B_in_cams)
            data_pr['frames'].extend(files)
            data_gt['pose_abs'].append(poseB.reshape(-1, 4, 4))
            data_gt['frames'].extend(files)
            data_in['pose_abs'].append(poseA.reshape(-1, 4, 4))
            data_in['frames'].extend(files)
            data_in['pose_symm'].append(batch['pose_gt_symm'].reshape(len(files), -1, 4, 4)) # need to have format (BT,...) for later operations
            data_gt['pose_symm'].append(batch['pose_gt_symm'].reshape(len(files), -1, 4, 4))
            data_pr['pose_symm'].append(batch['pose_gt_symm'].reshape(len(files), -1, 4, 4))

            # contact prediction 
            if 'contact' in out_dict:
                data_pr['contact_logits'].append(out_dict['contact'].reshape(B*T, -1))
                data_gt['contact_logits'].append(batch['contact_dist_gt'].reshape(B*T, -1)[:, [22, 23+15]])
                data_in['contact_logits'].append(batch['contact_dist_gt'].reshape(B*T, -1)[:, [22, 23+15]])
            else:
                # append all zeros dummy data
                data_pr['contact_logits'].append(torch.zeros(B*T, 2))
                data_gt['contact_logits'].append(torch.zeros(B*T, 2))
                data_in['contact_logits'].append(torch.zeros(B*T, 2))
                    
            frames.extend(files)
            pose_pred.append(B_in_cams.cpu().numpy())

            # evaluate human prediction
            if cfg.nlf_root is not None:
                NJ = 24
                betas, pred_smpl_pose, pred_smpl_r, pred_smpl_t = trainer.smpl_params_from_pred(batch, out_dict) # TODO: use predicted betas for data_pr
                data_pr['smpl_pose'].append(pred_smpl_pose) # (BT, 72)
                data_pr['smpl_t'].append(pred_smpl_t)
                data_pr['betas'].append(batch['betas_nlf'].reshape(-1, 10)) # TODO: use predicted NLF betas
                data_gt['smpl_pose'].append(batch['smpl_poses_gt'].reshape(-1, 156))
                data_gt['smpl_t'].append(batch['smpl_transl_gt'].reshape(-1, 3))
                data_gt['betas'].append(batch['betas_gt'].reshape(-1, 10))

                # Input data
                data_in['smpl_pose'].append(batch['nlf_poses'].reshape(-1, 156))
                data_in['smpl_t'].append(batch['nlf_transl'].reshape(-1, 3))
                data_in['betas'].append(batch['betas_nlf'].reshape(-1, 10))

            # add uncertainty prediction 
            uncertainties = torch.zeros(len(B_in_cams), 2 + 1 + 24)
            if 'rot_uncertainty' in out_dict:
                # (BT, 2 + 1 + 24), add predictions 
                uncertainties = torch.cat([out_dict['rot_uncertainty'], out_dict['trans_uncertainty'], out_dict['hum_pose_uncertainty'], out_dict['hum_trans_uncertainty']], -1).detach().cpu()
            data_pr['uncertainty'].append(uncertainties)
            data_gt['uncertainty'].append(uncertainties)
            data_in['uncertainty'].append(uncertainties)

        # now compute avg:
        data_pr = {k:torch.cat(v, 0) if k !='frames' and len(v) > 0 else v for k, v in data_pr.items()}
        data_gt = {k:torch.cat(v, 0) if k !='frames' and len(v) > 0 else v for k, v in data_gt.items()}
        data_in = {k:torch.cat(v, 0) if k !='frames' and len(v) > 0 else v for k, v in data_in.items()}

        if cfg.eval_input:
            print("Evaluating input FP + NLF!")
            data_pr = data_in # evaluate FP + NLF

        data_pr_chunk = {k:[] for k in keys}
        seq_name, prev_idx = '', 0
        res_dir = '/home/xianghuix/datasets/behave/foundpose-input/e2etracker/results'
        outfile = f'{res_dir}/{save_name}/{seq_name}.pkl'
        os.makedirs(osp.dirname(outfile), exist_ok=True)

        err_keys = self.get_error_keys()
        errors_all = {k:[] for k in err_keys}
        for i, frame in enumerate(tqdm(data_gt['frames'])):
            seq, ftime = frame.split('/')
            if seq != seq_name:
                curr_idx = i  # now cut the chunks
                if curr_idx > prev_idx:
                    # update and save
                    data_gt_chunk = {k: data_gt[k][prev_idx:curr_idx].clone() if k not in ['frames'] else data_gt[k][prev_idx:curr_idx] for k in keys}
                    data_pr_chunk = {k: data_pr[k][prev_idx:curr_idx].clone() if k not in ['frames'] else data_pr[k][prev_idx:curr_idx] for k in keys}
                    data_in_chunk = {k: data_in[k][prev_idx:curr_idx].clone() if k not in ['frames'] else data_in[k][prev_idx:curr_idx] for k in keys}
                    outfile = f'{res_dir}/{save_name}/{seq_name}.pth'
                    torch.save({"gt": data_gt_chunk, "pr": data_pr_chunk, "in": data_in_chunk}, outfile) # so that the smoothed results are updated 
                    
                    if not cfg.inf_only:
                        self.eval_1seq(cfg, data_gt_chunk, data_pr_chunk, errors_all, smoothnet_obj, smoothnet_smpl, trainer,
                                       run_smooth, smooth_smplt=cfg.smooth_smplt)
                    else:
                        print("Only inference, no metrics computation")
                    
                    prev_idx = curr_idx

                seq_name = seq

        # Eval the last seq
        curr_idx = len(data_gt['frames'])  # now cut the chunks
        data_gt_chunk = {k: data_gt[k][prev_idx:curr_idx].clone() if k not in ['frames'] else data_gt[k][prev_idx:curr_idx] for k in keys}
        data_pr_chunk = {k: data_pr[k][prev_idx:curr_idx].clone() if k not in ['frames'] else data_pr[k][prev_idx:curr_idx] for k in keys}
        data_in_chunk = {k: data_in[k][prev_idx:curr_idx].clone() if k not in ['frames'] else data_in[k][prev_idx:curr_idx] for k in keys}
        outfile = f'{res_dir}/{save_name}/{seq_name}.pth'
        torch.save({"gt": data_gt_chunk, "pr": data_pr_chunk, "in": data_in_chunk}, outfile)
        print('results saved to {}'.format(outfile))
        self.eval_1seq(cfg, data_gt_chunk, data_pr_chunk, errors_all, smoothnet_obj, smoothnet_smpl, trainer, run_smooth, False)
        outfile = self.save_output(cfg, errors_all, frames, save_name, seqs_test, trainer)

        # visualize the results as video 
        if cfg.render_video:
            from tools.viz_pred import PredVisualizer
            defaults = PredVisualizer.get_default_args()
            defaults.error_file = outfile.replace('.json', '.pkl').replace('results/', 'results/raw/')
            viz = PredVisualizer(defaults)
            defaults.use_sel_view = cfg.use_sel_view
            for seq in seqs_test[:20]: # render 20 maximum 
                defaults.kid = get_test_view_id(seq) if cfg.use_sel_view else 1
                defaults.kid = cfg.cam_id if defaults.kid is None else defaults.kid 
                outfile = f'{res_dir}/{save_name}/{seq}.pth'
                if not osp.isfile(outfile):
                    print(f'{outfile} not found, skipping')
                    continue
                defaults.pred_file = outfile
                viz.visualize(defaults)

    def get_error_keys(self):
        err_keys = ['rot', 'transl', 'mpjpe', 'v2v', 'mpjae', 'smpl_t']
        return err_keys

    def get_save_name(self, cfg, trainer):
        save_name = cfg.exp_name + '+' + osp.splitext(osp.basename(trainer.ckpt_file))[0] + cfg.identifier
        save_name = save_name if cfg.save_name == 'test' else cfg.save_name  # allow overriding
        return save_name

    def save_output(self, cfg, errors_all, frames, save_name, seqs_test, trainer):
        try:
            errors_all = {k: np.concatenate(v) if len(v) > 0 else np.array([0.]) for k, v in errors_all.items()}
        except:
            breakpoint()
        # save results
        ss = self.get_summary_string(cfg, errors_all, seqs_test, trainer)
        print(ss)
        ts = datetime.now().strftime('%Y%m%d-%H%M%S')
        outfile = f'results/{osp.splitext(osp.basename(cfg.split_file))[0]}+{save_name}+{ts}.json'
        os.makedirs(osp.dirname(outfile), exist_ok=True)
        os.makedirs(osp.dirname(outfile)+'/raw', exist_ok=True)
        avg_dict = {k: float(np.mean(errors_all[k])) for k in errors_all.keys()}
        json_dict = {
            **avg_dict,
            'summary': ss,
            'seqs': seqs_test,
            'total': len(errors_all['rot']),
        }
        json.dump(
            json_dict, open(outfile, 'w'), indent=2
        )
        pkl_dict = {
            **avg_dict,
            'seqs': seqs_test,
            'total': len(errors_all['rot']),
            'frames': frames,
            'cfg': cfg,
            **{k + "_errors": errors_all[k] for k in errors_all.keys()}
        }
        joblib.dump(
            pkl_dict, outfile.replace('.json', '.pkl').replace('results/', 'results/raw/')
        )
        print('all done, saved to', outfile)
        return outfile

    def get_summary_string(self, cfg, errors_all, seqs_test, trainer):
        ss = (
            f"{cfg.exp_name}/{osp.basename(trainer.ckpt_file)} {len(seqs_test)} seqs {len(errors_all['rot'])} examples: rot={np.mean(errors_all['rot']) * 180 / np.pi:.4f}, "
            f"trans={np.mean(errors_all['transl']):.4f}, "
            f"v2v={np.mean(errors_all['v2v']):.4f}, mpjae={np.mean(errors_all['mpjae']) * 180 / np.pi:.4f}, mpjpe={np.mean(errors_all['mpjpe']):.4f}, "
            f"smpl_t={np.mean(errors_all['smpl_t']):.4f}")
        return ss

    def eval_1seq(self, cfg, data_gt_chunk, data_pr_chunk, errors_all, smoothnet_obj, smoothnet_smpl, trainer, smooth=False, smooth_smplt=False):
        if smooth:
            pose_pr_smooth = smoothnet_obj.smooth_seq(data_pr_chunk['pose_abs'], data_pr_chunk['frames'])
            data_pr_chunk['pose_abs'] = pose_pr_smooth # update the prediction 
        else:
            pose_pr_smooth = data_pr_chunk['pose_abs']

        self.add_obj_errors(data_gt_chunk, errors_all, pose_pr_smooth)
        # SMPL evaluation
        if cfg.nlf_root is not None:
            if smooth:
                smpl_smooth = smoothnet_smpl.smooth_seq(data_pr_chunk['smpl_pose'],
                                                        data_pr_chunk['smpl_t'],
                                                        data_pr_chunk['betas'],
                                                        data_pr_chunk['frames'])
                
                data_pr_chunk['smpl_pose'] = torch.from_numpy(smpl_smooth['poses']).cuda() # update the pose 
                data_pr_chunk['betas'] = torch.from_numpy(smpl_smooth['betas']).cuda() # update the betas 
                if not smooth_smplt:
                    print("Not smoothing SMPL translation")
                    smpl_smooth['trans'] = data_pr_chunk['smpl_t'].cpu().numpy() # use the original translation instead of smoothed one 
                data_pr_chunk['smpl_t'] = torch.from_numpy(smpl_smooth['trans']).cuda() # update the translation as well 

            else:
                smpl_smooth = {
                    'poses': data_pr_chunk['smpl_pose'].cpu().numpy(),
                    'trans': data_pr_chunk['smpl_t'].cpu().numpy(),
                    'betas': data_pr_chunk['betas'].cpu().numpy(),
                }

            self.add_smpl_errors(data_gt_chunk, data_pr_chunk, errors_all, smpl_smooth, trainer)

    def add_smpl_errors(self, data_gt_chunk, data_pr_chunk, errors_all, smpl_smooth, trainer):
        NJ = 24
        verts_smpl, jtrs_pr, _, _, jts_rot_pr = trainer.smpl_model(torch.from_numpy(smpl_smooth['poses']).cuda(),
                                                                   torch.from_numpy(smpl_smooth['betas']).cuda(),
                                                                   torch.from_numpy(smpl_smooth['trans']).cuda(),
                                                                   ret_glb_rot=True)
        verts_smpl_gt, jtrs_gt, _, _, jts_rot_gt = trainer.smpl_model(data_gt_chunk['smpl_pose'],
                                                                      data_gt_chunk['betas'],
                                                                      data_gt_chunk['smpl_t'],
                                                                      ret_glb_rot=True)
        v2v = torch.sum((verts_smpl_gt - verts_smpl) ** 2, -1).sqrt().mean(-1).cpu().numpy()
        # for mpjpe: subtract the root joint first, https://github.com/xiexh20/behave-dataset/blob/main/challenges/lib/metrics.py#L180
        jtrs_pr = jtrs_pr - jtrs_pr[:, 0:1]
        jtrs_gt = jtrs_gt - jtrs_gt[:, 0:1]
        mpjpe = torch.sum((jtrs_pr - jtrs_gt) ** 2, -1).sqrt().reshape(-1, NJ).mean(-1).cpu().numpy()
        smpl_t = torch.sum((data_gt_chunk['smpl_t'] - data_pr_chunk['smpl_t']) ** 2, -1).sqrt().cpu().numpy()
        mpjae = Utils.geodesic_distance_batch(jts_rot_pr.reshape(-1, 3, 3), jts_rot_gt.reshape(-1, 3, 3)).reshape(-1, NJ).mean(-1).cpu().numpy()  # T*J
        errors_all['v2v'].append(v2v)
        errors_all['mpjpe'].append(mpjpe)
        errors_all['smpl_t'].append(smpl_t)
        errors_all['mpjae'].append(mpjae)

    def add_obj_errors(self, data_gt_chunk, errors_all, pose_pr_smooth):
        pose_o_gt = data_gt_chunk['pose_abs']
        errors_r, errors_t = [], []
        if 'pose_symm' in data_gt_chunk:
            pose_o_gt_symm = data_gt_chunk['pose_symm'] # (BT, N, 4, 4)
            for i in range(pose_o_gt_symm.shape[1]):
                err_r = geom_utils.geodesic_distance(pose_pr_smooth[:, :3, :3], pose_o_gt_symm[:, i, :3, :3].reshape(-1, 3, 3))
                errors_r.append(err_r)
                err_t = torch.sqrt(torch.sum((pose_pr_smooth[:, :3, 3].reshape(-1, 3) - pose_o_gt_symm[:, i, :3, 3].reshape(-1, 3)) ** 2, -1))
                errors_t.append(err_t)
            err_r = torch.stack(errors_r, -1).min(-1)[0] # (B, N) -> (B,)
            et = torch.stack(errors_t, -1).min(-1)[0] # (B, N) -> (B,)
            print("Using symmetry pose for evaluation")
        else:
            err_r = geom_utils.geodesic_distance(pose_o_gt[:, :3, :3].reshape(-1, 3, 3).float(), pose_pr_smooth[:, :3, :3].float())
            et = torch.sqrt(torch.sum((pose_o_gt[:, :3, 3].reshape(-1, 3) - pose_pr_smooth[:, :3, 3]) ** 2, -1))

        errors_all['rot'].append(err_r.cpu().numpy())
        errors_all['transl'].append(et.cpu().numpy()) 



def eval_hvopnet():
    'evaluate results from HVOPNet /home/xianghuix/datasets/behave/fp-nlf-hvopnet/'
    from scipy.spatial.transform import Rotation
    cfg = get_config()
    evaluator = ModelEvaluator(cfg)
    keys = ['pose_abs', 'smpl_pose', 'smpl_t', 'frames', 'betas']
    data_gt, data_pr, data_in = {}, {}, {}
    seqs = json.load(open('splits/date03-9seqs.json'))['test']
    trainer = Trainer(cfg)

    err_keys = ['rot', 'transl', 'mpjpe', 'v2v', 'mpjae', 'smpl_t']
    errors_all = {k: [] for k in err_keys}
    frames = []
    for seq in tqdm(seqs):
        file_hvopnet = f'/home/xianghuix/datasets/behave/fp-gtsmpl-hvopnet-th0.7/{seq}_params.pkl'
        file_packed = f'/home/xianghuix/datasets/behave/behave-packed/{seq}_GT-packed.pkl'
        data_hvopnet = joblib.load(file_hvopnet)
        data_packed = joblib.load(file_packed)
        L = len(data_packed['frames'])
        pose_abs = np.eye(4)[None].repeat(L, 0)
        pose_abs[:, :3, :3] = data_hvopnet['obj_angles']
        pose_abs[:, :3, 3] = data_hvopnet['obj_trans']
        data_pr['pose_abs'] = torch.from_numpy(pose_abs).cuda().float()
        pose_abs = np.eye(4)[None].repeat(L, 0)
        pose_abs[:, :3, :3] = Rotation.from_rotvec(data_packed['obj_angles']).as_matrix()
        pose_abs[:, :3, 3] = data_packed['obj_trans']
        data_gt['pose_abs'] = torch.from_numpy(pose_abs).cuda().float()

        poses_full = data_packed['poses'].astype(np.float32)
        poses_72 = np.concatenate([poses_full[:, :69], poses_full[:, 69 + 45:69 + 48]], axis=1)
        data_gt['smpl_pose'] = torch.from_numpy(poses_72).cuda().float()
        poses_full = data_hvopnet['poses'].astype(np.float32)
        poses_72 = np.concatenate([poses_full[:, :69], poses_full[:, 69 + 45:69 + 48]], axis=1)
        data_pr['smpl_pose'] = torch.from_numpy(poses_72).cuda().float()

        # Use GT betas
        data_pr['betas'] = torch.from_numpy(data_packed['betas']).cuda().float()
        data_gt['betas'] = torch.from_numpy(data_packed['betas']).cuda().float()
        data_gt['smpl_t'] = torch.from_numpy(data_packed['trans']).cuda().float()
        data_pr['smpl_t'] = torch.from_numpy(data_hvopnet['trans']).cuda().float()

        data_pr['frames'] = [osp.join(seq, x) for x in data_packed['frames']]
        data_gt['frames'] = [osp.join(seq, x) for x in data_packed['frames']]

        evaluator.eval_1seq(cfg, data_gt, data_pr, errors_all, None, None, trainer, False, False)
        frames.extend(data_gt['frames'])
    save_name = evaluator.get_save_name(cfg, trainer) + '_GT-SMPL-HVOPNet-th0.7'
    outfile = evaluator.save_output(cfg, errors_all, frames, save_name, seqs, trainer)



def main():
    cfg = get_config()
    evaluator = ModelEvaluator(cfg)
    evaluator.evaluate(cfg)


if __name__ == '__main__':
    main()