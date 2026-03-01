# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
precompute contacts for the dataset
"""
import sys, os
sys.path.append(os.getcwd())
import json
import os
import os.path as osp
import joblib
import igl
import numpy as np
import time
import torch
from tqdm import tqdm
from lib_smpl import get_smpl, pose72to156
from scipy.spatial.transform import Rotation as R
import trimesh


class ContactPrecompute:
    def __init__(self):
        self.smpl_male = get_smpl('male', hands=True)
        self.smpl_female = get_smpl('female', hands=True)


    def compute(self, args):
        "load GT and NLF data, together with object shapes, compute contact distances"
        fp_root = args.fp_root
        nlf_root = args.nlf_path
        packed_root = '/home/xianghuix/datasets/behave/behave-packed'
        outdir = args.outdir
        os.makedirs(outdir, exist_ok=True)
        
        seq = osp.basename(args.video).split('.')[0]
        outfile = osp.join(outdir, f'{seq}_contact-jts.npz')
        if osp.isfile(outfile):
            print(f'{outfile} already exists, skipping')
            return

        # preload object templates
        from behave_data.utils import load_templates_all, load_template

        if args.wild_video:
            from behave_data.const import get_hy3d_mesh_file
            file_hy3d = get_hy3d_mesh_file(seq)
            template = trimesh.load(file_hy3d, process=False)
            template.vertices = template.vertices - np.mean(template.vertices, 0)
            template_gt = template.copy()
        else:
            if 'hy3d' in fp_root:
                # load hy3d mesh 
                from behave_data.const import get_hy3d_mesh_file
                file_hy3d = get_hy3d_mesh_file(seq)
                template = trimesh.load(file_hy3d, process=False)
            else:
                template = load_template(seq.split('_')[2], cent=False, dataset_path='/home/xianghuix/datasets/behave')
            template_gt = load_template(seq.split('_')[2], cent=False, dataset_path='/home/xianghuix/datasets/behave')
            center_gt = np.mean(template_gt.vertices, 0)
            template.vertices = template.vertices - center_gt
            template_gt.vertices = template_gt.vertices - center_gt

        # simplify object tempalte if it is too dense
        if len(template.vertices) > 8000:
            from open3d.utility import Vector3dVector, Vector3iVector
            import open3d
            N = len(template.vertices)
            mesh_full = open3d.geometry.TriangleMesh(Vector3dVector(template.vertices), Vector3iVector(template.faces))

            mesh_simplifed = mesh_full.simplify_quadric_decimation(5000, 0.5)
            template = trimesh.Trimesh(np.array(mesh_simplifed.vertices), np.array(mesh_simplifed.triangles))
            print(f'Simplified template from {N} to {len(template.vertices)} vertices')
        
        nlf_file = osp.join(nlf_root, f'{seq}_params.pkl')
        fp_file = osp.join(fp_root, f'{seq}_all.pkl')
        packed_file = osp.join(packed_root, f'{seq}_GT-packed.pkl')
        if not osp.isfile(packed_file):
            assert args.wild_video, f'{packed_file} does not exist, must be wild video!'
            packed_file = f'/home/xianghuix/datasets/behave/behave-packed/Date03_Sub03_chairwood_hand_GT-packed.pkl'
            packed_data = joblib.load(packed_file)
            packed_data_cut = {}
            for k, v in packed_data.items():
                packed_data_cut[k] = v[:5] # only keep minimum dummy data 
            packed_data = packed_data_cut
        else:
            packed_data = joblib.load(packed_file)
        # if file does not exist, skip
        if not osp.isfile(nlf_file) or not osp.isfile(fp_file) or not osp.isfile(packed_file):
            print(f'{nlf_file} or {fp_file} or {packed_file} does not exist, skipping')
            return  

        nlf_data = joblib.load(nlf_file)
        fp_data = joblib.load(fp_file)
        

        # get GT SMPL joints
        gender = packed_data['gender']
        body_model = self.smpl_male if gender == 'male' else self.smpl_female
        jts_smpl_gt = body_model.get_joints(torch.from_numpy(packed_data['poses']).float(),
                                        torch.from_numpy(packed_data['betas']).float(), torch.from_numpy(packed_data['trans']).float())    
        obj_angles = packed_data['obj_angles']
        obj_transls = packed_data['obj_trans']
        pose_gt = np.eye(4)[None].repeat(len(obj_angles), 0)
        pose_gt[:, :3, :3] = R.from_rotvec(obj_angles).as_matrix()
        pose_gt[:, :3, 3] = obj_transls
        
        obj_name = seq.split('_')[2]
        obj_template = template
        obj_verts = np.array(template_gt.vertices).copy()[None].repeat(len(packed_data['poses']), 0) # GT always use template 
        verts_gt = np.matmul(obj_verts, pose_gt[:, :3, :3].transpose(0, 2, 1)) + pose_gt[:, :3, 3][:, None]

        # now compute NLF joints from nlf_data
        nlf_poses = nlf_data['poses']
        nlf_poses = np.stack([pose72to156(nlf_poses[:, x, :72]) for x in range(4)], 1)
        fp_poses = fp_data['fp_poses']
        dists_nlf_fp = []
        closest_points = []
        num_cams = 6 if 'ICapS' in seq else 4
        kids = range(num_cams) if 'hy3d' not in fp_root else range(1)
        for k in kids:
            jts_smpl_nlf = body_model.get_joints(torch.from_numpy(nlf_poses[:, k]).float(),
                                        torch.from_numpy(nlf_data['betas'][:, k]).float(), torch.from_numpy(nlf_data['transls'][:, k]).float())
            obj_verts_fp = np.array(obj_template.vertices).copy()[None].repeat(len(fp_poses), 0)
            verts_fp = np.matmul(obj_verts_fp, fp_poses[:, k, :3, :3].transpose(0, 2, 1)) + fp_poses[:, k, :3, 3][:, None]
            dists_k = []
            closest_points_k = []
            for vobj, vsmpl in zip(tqdm(verts_fp), jts_smpl_nlf.cpu().numpy()):
                res = igl.signed_distance(vsmpl, vobj, np.array(obj_template.faces).astype(int))
                dist = res[0]
                dists_k.append(np.abs(dist))
                closest_points_k.append(res[2])
            dists_nlf_fp.append(np.array(dists_k))
            closest_points.append(np.stack(closest_points_k, 0))
        dists_nlf_fp = np.stack(dists_nlf_fp, 0) if 'hy3d' not in fp_root else np.stack(dists_nlf_fp*num_cams, 0)
        closest_points = np.stack(closest_points, 0) if 'hy3d' not in fp_root else np.stack(closest_points*num_cams, 0) # (4, T, N, 3)
        

        # compute GT contact distances
        if args.wild_video:
            dists_gt = dists_nlf_fp[0]
            closest_points_gt = closest_points[0]
        else:
            dists_gt = []
            closest_points_gt = []
            for vobj, vsmpl in zip(tqdm(verts_gt), jts_smpl_gt.cpu().numpy()):
                res = igl.signed_distance(vsmpl, vobj, np.array(template_gt.faces).astype(int))
                dist = res[0]
                closest_points_gt.append(res[2])
                dists_gt.append(np.abs(dist))
            dists_gt = np.array(dists_gt)
            closest_points_gt = np.stack(closest_points_gt, 0)
        np.savez(outfile, dists_noisy=dists_nlf_fp, dists_gt=dists_gt, closest_points_noisy=closest_points, closest_points_gt=closest_points_gt)
        print(f'Saved {outfile}')

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-v', '--video', help='path to a video file')
    parser.add_argument('-o', '--outdir', help='path to save the results', default='/home/xianghuix/datasets/behave/contact-jts')
    parser.add_argument('-fp', '--fp_root', help='path to the FP results', default='/home/xianghuix/datasets/behave/fp-unidepth-jump-all-aligned')
    parser.add_argument('-nlf', '--nlf_path', help='path to the NLF results', default='/home/xianghuix/datasets/behave/nlf-smplh-gender')
    parser.add_argument('-wild', '--wild_video', help='whether the video is wild video', action='store_true')
    args = parser.parse_args()
    
    cp = ContactPrecompute()
    cp.compute(args)
    print('all done')