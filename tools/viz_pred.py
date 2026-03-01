# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
visualize predictions

to render IMHD videos:

python tools/viz_pred.py -pf outputs/results/HORefine-Jloss-filter-unidepth-allobj-symm-abs+step024978_abs-hy3d3-icapk0-unnorm/ICapS02_sub09_obj01_Seg_0.pth --data_source intercap --video /home/xianghuix/datasets/behave/unidepth/ICapS02_sub09_obj01_Seg_0.0.color.mp4

"""
import glob

import cv2
import sys, os

import imageio
import os.path as osp

sys.path.append(os.getcwd())
import torch
from tqdm import tqdm
import numpy as np
import nvdiffrast.torch as dr
import joblib
import trimesh
from pytorch3d.renderer import look_at_view_transform
from behave_data.const import get_test_view_id
import Utils

from prep.fp_behave import FPBehaveVideoProcessor
from behave_data.const import _sub_gender
from lib_smpl import get_smpl, pose72to156, colors24 


class PredVisualizer(FPBehaveVideoProcessor):
    "load pred from pth file"
    def __init__(self, args):
        # lightweight init: only camera intrinsics and helpers
        self.args = args
        self.scale_ratio = 4
        self.video = args.video

        self.camera_K = self.init_camera_K() # to be consistent with dataset format 
        self.scale_ratio = 4
        print(self.camera_K, 'scale ratio', self.scale_ratio)
        self.camera_K[:2] /= self.scale_ratio
        self.init_others()

    @staticmethod
    def get_parser():
        parser = FPBehaveVideoProcessor.get_parser()
        parser.add_argument('-pf', '--pred_file')
        parser.add_argument('--chunk_size', type=int, default=600)
        parser.add_argument('--out_root', type=str,
                            default='/home/xianghuix/datasets/behave/foundpose-input/e2etracker/viz')
        parser.add_argument('-ef', '--error_file', type=str, default=None )
        parser.add_argument('--use_sel_view', action='store_true')
        parser.add_argument('--no_sphere', action='store_true')
        parser.add_argument('--filter_oneeuro', action='store_true')

        return parser

    @staticmethod
    def get_default_args():
        from argparse import Namespace
        return Namespace(
        video=None,
        outpath='',
        fps=30,
        tstart=3.0,
        tend=None,
        redo=False,
        kid=1,
        start=0,
        end=-1,
        nodepth=True,
        pred_file=None,
        chunk_size=600,
        wild_video=False,
        out_root='/home/xianghuix/datasets/behave/foundpose-input/e2etracker/viz',
        use_sel_view=False,
        no_sphere=False,
        data_source='behave',
        filter_oneeuro=False,
    )

    def get_video_path(self, seq):
        return f'/home/xianghuix/datasets/behave/videos/{seq}.{self.args.kid}.color.mp4'

    def get_chunk_num(self):
        return 5000000

    def init_others(self):
        self.smpl_female = get_smpl('female', True).cuda()
        self.smpl_male = get_smpl('male', True).cuda()
        self.glctx = dr.RasterizeCudaContext()
        self._mesh_cache = {}
        self._mesh_cache_gt = {}
        # cached unit icosphere template for fast sphere rendering (uniform color)
        try:
            sphere_mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
            self._sphere_template = Utils.make_mesh_tensors(sphere_mesh)
        except Exception:
            self._sphere_template = None

    def visualize(self, args):
        "render H+O with batch precompute and video-based RGB, write mp4 per sequence"
        out_root = args.out_root
        os.makedirs(out_root, exist_ok=True)
        out_data = torch.load(args.pred_file, map_location='cpu')
        data_gt, data_pr, data_in = out_data['gt'], out_data['pr'], out_data['in']

        frames = data_pr['frames']
        K = self.camera_K.copy()

        end = len(frames) if args.end is None else args.end
        frames = frames[:end]

        # helper to load combined mesh tensors (SMPL parts + object) with UV texture
        def get_mesh_tensors(video_prefix, obj_name, combine_smpl=True, use_hy3d=False):
            if not use_hy3d:
                if obj_name in self._mesh_cache_gt:
                    return self._mesh_cache_gt[obj_name]
            else:
                if obj_name in self._mesh_cache:
                    return self._mesh_cache[obj_name]
            if combine_smpl:
                mesh_tensors, meshes = Utils.load_smpl_obj_uvmap(video_prefix, use_hy3d=use_hy3d, human_texture='part')
                verts_list = meshes.verts_list()
                smpl_n = verts_list[0].shape[0]
                obj_base = verts_list[1].to('cuda').float()
                # center object vertices using simplified template center (dataset convention)
                from behave_data.utils import load_template as load_template_simple
                try:
                    simp = load_template_simple(obj_name, cent=False, dataset_path='/home/xianghuix/datasets/behave')
                except Exception as e:
                    print(f'Error loading template {obj_name}: {e}')
                    from behave_data.const import get_hy3d_mesh_file
                    hy3d_file = get_hy3d_mesh_file(video_prefix)
                    simp = trimesh.load(hy3d_file, process=False)
                cent_np = np.mean(simp.vertices, axis=0)
                cent = torch.as_tensor(cent_np, device='cuda', dtype=torch.float)
                obj_base = obj_base - cent
                cache = {
                    'mesh_tensors': mesh_tensors,
                    'obj_base': obj_base,
                    'smpl_n': smpl_n,
                    'obj_name': obj_name,
                    'cent': cent,
                }
            else:
                if use_hy3d:
                    mesh_file = get_hy3d_mesh_file(video_prefix)
                    if mesh_file is None:
                        return None
                else:
                    mesh_file = f'/home/xianghuix/datasets/behave/objects/{obj_name}/{obj_name}.obj'
                mesh = trimesh.load(mesh_file, process=False)
                mesh_tensors = Utils.make_mesh_tensors(mesh)
                obj_base = mesh_tensors['pos']
                from behave_data.utils import load_template as load_template_simple
                simp = load_template_simple(obj_name, cent=False, dataset_path='/home/xianghuix/datasets/behave')
                cent_np = np.mean(simp.vertices, axis=0)
                cent = torch.as_tensor(cent_np, device='cuda', dtype=torch.float)
                obj_base = obj_base - cent
                cache = {
                    'mesh_tensors': mesh_tensors,
                    'obj_base': obj_base,
                    'smpl_n': 0,
                    'obj_name': obj_name,
                    'cent': cent,
                }
                print("object center: ", cent_np)
            return cache
        has_smpl = True
        if has_smpl:
            if args.filter_oneeuro:
                from tools.filter_oneeuro import filter_3axis
                smplt_pr_filter = filter_3axis(data_pr['smpl_t'].cpu().numpy())
                data_pr['smpl_t'] = torch.from_numpy(smplt_pr_filter).to('cuda').float()
                print('smpl_t filtered')

                # also filter the object translation, object rotation 
                obj_transl = data_pr['pose_abs'][:, :3, 3]
                obj_transl_filter = filter_3axis(obj_transl.cpu().numpy())
                data_pr['pose_abs'][:, :3, 3] = torch.from_numpy(obj_transl_filter).to('cuda').float()
                print('object translation filtered')

            smpl_pose_pr_all = data_pr['smpl_pose'].to('cuda').float()
            smpl_pose_pr_all = pose72to156(smpl_pose_pr_all) if smpl_pose_pr_all.shape[1] == 72 else smpl_pose_pr_all
            smpl_t_pr_all = data_pr['smpl_t'].to('cuda').float()
            betas_pr_all = data_pr['betas'].to('cuda').float()
            smpl_pose_gt_all = data_gt['smpl_pose'].to('cuda').float()
            smpl_pose_gt_all = pose72to156(smpl_pose_gt_all) if smpl_pose_gt_all.shape[1] == 72 else smpl_pose_gt_all
            smpl_t_gt_all = data_gt['smpl_t'].to('cuda').float()
            betas_gt_all = data_gt['betas'].to('cuda').float()
            smpl_pose_in_all = data_in['smpl_pose'].to('cuda').float()
            smpl_pose_in_all = pose72to156(smpl_pose_in_all) if smpl_pose_in_all.shape[1] == 72 else smpl_pose_in_all
            smpl_t_in_all = data_in['smpl_t'].to('cuda').float()
            betas_in_all = data_in['betas'].to('cuda').float()
        
        # ensure tensors (single sequence)
        pose_pr_all = data_pr['pose_abs'].to('cuda').float()
        pose_gt_all = data_gt['pose_abs'].to('cuda').float()
        pose_in_all = data_in['pose_abs'].to('cuda').float() 
        use_hy3d = 'hy3d' in args.pred_file

        # single sequence and contiguous indices
        seq, _ = frames[0].split('/')
        inds_all = list(range(len(frames)))
        end = len(inds_all) if args.end is None else args.end
        inds_all = inds_all[:end]

        # prepare mesh and video
        gender = _sub_gender[seq.split('_')[1]]
        body_model = self.smpl_male if gender == 'male' else self.smpl_female
        obj_name = seq.split('_')[2]
        video_prefix = seq 
        cache = get_mesh_tensors(video_prefix, obj_name, combine_smpl=has_smpl, use_hy3d=use_hy3d)
        if args.data_source == 'intercap' or args.wild_video:
            # this one does not have texture in the GT mesh, but still needs hy3d to align 
            cache_gt = get_mesh_tensors(video_prefix, obj_name, combine_smpl=has_smpl, use_hy3d=True)
        else:
            cache_gt = get_mesh_tensors(video_prefix, obj_name, combine_smpl=has_smpl, use_hy3d=False) # if not args.wild_video else get_mesh_tensors(video_prefix, obj_name, combine_smpl=has_smpl, use_hy3d=True)
        
        mesh_tensors = cache['mesh_tensors']
        mesh_tensors_gt = cache_gt['mesh_tensors']
        obj_base = cache['obj_base']
        obj_base_gt = cache_gt['obj_base']
        gt_to_perturb_pose = np.eye(4)
        gt_to_perturb_pose[:3, 3] = cache['cent'].cpu().numpy() #if args.data_source != 'intercap' else - cache['cent'].cpu().numpy() # for FP pose, which is w.r.t. non-centered object 
        print("Using center: ", gt_to_perturb_pose, 'mesh center: ', cache['cent'].cpu().numpy())

        # Load input data 
        # read tools/render_fp_nlf.py to see how to load the SMPL parameters and vertices from pkl file
        # read tools/render_fp_smpl.py to see how to load FP object parameters from pkl file
        # compute vertices from the parameters, and later use that to render SMPL and object
        if args.use_sel_view:
            view = get_test_view_id(seq)
            kid = view if view is not None else 1
        else:
            kid = 1 if args.data_source == 'behave' and not args.wild_video else 0 # TODO: check this 
            
        kid = 1 if args.data_source == 'intercap' else kid 
        video_path = args.video if args.video is not None else f'/home/xianghuix/datasets/behave/videos/{seq}.{kid}.color.mp4'
        if '2023'  in seq:
            kid = 0 
            video_path = f'/home/xianghuix/datasets/IMHD2/videos-behave/{seq}.{kid}.color.mp4'
        print(f'Using kid: {kid}, video path: {video_path}')
        # NLF SMPL params (poses, betas, transls) for selected Kinect
        nlf_file = f'/home/xianghuix/datasets/behave/nlf/{seq}_params.pkl'
        has_nlf = osp.isfile(nlf_file)
        if has_nlf:
            nlf_data = joblib.load(nlf_file)
            nlf_frames = nlf_data['frames'] if 'frames' in nlf_data else [f't{t:08.3f}' for t in range(len(nlf_data['poses']))]
            time_to_nlf = {ft: i for i, ft in enumerate(nlf_frames)}
            thetas_all = nlf_data['poses'][:, kid]
            betas_all = nlf_data['betas'][:, kid]
            transls_all = nlf_data['transls'][:, kid]
            # use time-averaged betas as in tools/render_fp_nlf.py
            betas_avg_all = np.mean(betas_all, axis=0, keepdims=True).repeat(thetas_all.shape[0], axis=0)
        else:
            time_to_nlf, thetas_all, betas_all, transls_all, betas_avg_all = {}, None, None, None, None

        # FoundationPose object poses (camera coordinates) per Kinect
        fp_file = f'/home/xianghuix/datasets//behave/fp-hy3d2-unidepth/{seq}_all.pkl'
        has_fp = osp.isfile(fp_file)
        if has_fp:
            fp_data = joblib.load(fp_file)
            fp_frames = fp_data['frames']
            fp_poses_all = fp_data['fp_poses']  # (T, K, 4, 4)
            # map time -> pose for target kid
            time_to_fp = {ft: fp_poses_all[i, kid] for i, ft in enumerate(fp_frames)}
        else:
            time_to_fp = {}

        
        from behave_data.video_reader import VideoController
        from behave_data.utils import availabe_kindata
        seq = osp.basename(args.pred_file).split('.')[0]
        self.video = video_path
        video_path = self.video
        kids, comb = availabe_kindata(video_path, kinect_count=4)
        controllers = [VideoController(video_path.replace(f'.{kid}.', f'.{k}.')) for k in kids]

        # determine render size from first frame
        first_time = float(frames[0].split('/')[1].replace('t', ''))
        first_img = controllers[kid].get_closest_frame(controllers[kid].get_closest_time(first_time))
        h0, w0 = first_img.shape[:2]
        H, W = h0 // self.scale_ratio, w0 // self.scale_ratio
        projection_mat = torch.as_tensor(
            Utils.projection_matrix_from_intrinsics(K, height=H, width=W, znear=0.001, zfar=100).reshape(1, 4, 4),
            device='cuda', dtype=torch.float)
        ob_in_glcams = torch.tensor(Utils.glcam_in_cvcam, device='cuda', dtype=torch.float).reshape(1, 4, 4)
        mtx = projection_mat @ ob_in_glcams

        # writer
        pred_dir = osp.basename(osp.dirname(args.pred_file))
        save_name = f'{pred_dir}_nosphere{args.no_sphere}-1euro-{args.filter_oneeuro}+{seq}.mp4'
        out_path = osp.join(out_root, save_name)
        vw = imageio.get_writer(out_path, 'FFMPEG', fps=30)
        self.side_view_z = None 

        # helper
        def batch_render(verts_comb, crop_y0_ratio=0.15, crop_x0_ratio=0.15, crop_x1_ratio=0.85, mesh_tensors=None, spheres=None):
            B = verts_comb.shape[0]
            device = verts_comb.device

            # build front-view projection matrices (batched)
            ob_in_glcams = torch.tensor(Utils.glcam_in_cvcam, device=device, dtype=torch.float)[None].repeat(B, 1, 1)
            projs = [Utils.projection_matrix_from_intrinsics(K, height=H, width=W, znear=0.001, zfar=100) for _ in range(B)]
            projection_mat = torch.as_tensor(np.stack(projs, axis=0), device=device, dtype=torch.float)
            mtx_front = projection_mat @ ob_in_glcams

            # FRONT VIEW (base geometry RGB)
            pos_homo_front = Utils.to_homo_torch(verts_comb)
            pos_clip_front = (mtx_front[:, None] @ pos_homo_front[..., None])[..., 0]
            color_front_base = Utils.nvdiff_rasterize(self.glctx, mesh_tensors, pos_clip_front, (H, W))

            # SIDE VIEW transform (look-at 90 deg azim)
            if self.side_view_z is None:
                self.side_view_z = torch.mean(verts_comb[:, :, 2])
            z_now = torch.mean(verts_comb[:, :, 2])
            if abs(z_now - self.side_view_z) > 0.5:
                self.side_view_z = z_now
            z = self.side_view_z
            at = torch.tensor([[0.0, 0.0, z]], device=device, dtype=torch.float)
            dist = z*1.75 if args.wild_video else z*1.3
            R, T = look_at_view_transform(dist=dist, elev=0, azim=75, at=at, up=((0, 1, 0),))
            view_mat = torch.eye(4, device=device, dtype=torch.float)
            view_mat[:3, :3] = R[0]
            view_mat[:3, 3] = T[0]
            verts_side = torch.matmul(verts_comb, view_mat[:3, :3]) + view_mat[:3, 3]
            pos_homo_side = Utils.to_homo_torch(verts_side)
            pos_clip_side = (mtx_front[:, None] @ pos_homo_side[..., None])[..., 0]
            color_side_base = Utils.nvdiff_rasterize(self.glctx, mesh_tensors, pos_clip_side, (H, W))

            # Optional: spheres (centers, radii, colors)
            def prepare_sphere_batch(centers_bxs, radii_bxs, colors_s_or_bxs):
                # centers_bxs: (B, S, 3), radii_bxs: (B, S), colors: (S,3) or (B,S,3)
                if self._sphere_template is None:
                    return None
                pos_unit = self._sphere_template['pos']  # (Nv,3)
                faces_unit = self._sphere_template['faces']  # (Nf,3)
                Nv = pos_unit.shape[0]
                Nf = faces_unit.shape[0]
                B_local, S = centers_bxs.shape[:2]
                # broadcast radii
                if radii_bxs.dim() == 1:
                    radii_bxs = radii_bxs[None].repeat(B_local, 1)
                # colors per-sphere (S,3). If batch-provided, use first batch for efficiency
                if colors_s_or_bxs.dim() == 3:
                    colors_s = colors_s_or_bxs[0]
                else:
                    colors_s = colors_s_or_bxs
                # verts_cam for all spheres
                verts_spheres_cam = []
                for b in range(B_local):
                    centers_bs = centers_bxs[b]  # (S,3)
                    radii_bs = radii_bxs[b][:, None]  # (S,1)
                    # (S, Nv, 3)
                    v_bs = centers_bs[:, None, :] + radii_bs[:, None, :] * pos_unit[None, :, :]
                    v_bs = v_bs.reshape(-1, 3)  # (S*Nv, 3)
                    verts_spheres_cam.append(v_bs)
                verts_spheres_cam = torch.stack(verts_spheres_cam, 0).to(device=device, dtype=torch.float)
                # faces with offsets (shared across batch)
                faces_all = []
                for s in range(S):
                    faces_all.append(faces_unit + s * Nv)
                faces_all = torch.cat(faces_all, 0)
                # vertex colors per-vertex (S*Nv,3) using per-sphere uniform color
                vc_all = torch.cat([colors_s[s].reshape(1, 1, 3).repeat(1, Nv, 1) for s in range(S)], 1)[0]
                vc_all = vc_all.to(device=device, dtype=torch.float)
                mesh_spheres = {
                    'faces': faces_all.to(device=device, dtype=torch.int),
                    'vertex_color': vc_all,
                }
                return verts_spheres_cam, mesh_spheres

            if spheres is not None and 'centers' in spheres and 'radii' in spheres and 'colors' in spheres and spheres['centers'] is not None and spheres['radii'] is not None:
                centers = spheres['centers']  # (B,S,3) or (S,3)
                if centers.dim() == 2:
                    centers = centers[None].repeat(B, 1, 1)
                radii = spheres['radii']  # (B,S) or (S)
                colors = spheres['colors']  # (S,3) or (B,S,3)

                verts_sph_front, mesh_sph = prepare_sphere_batch(centers.to(device), radii.to(device), colors.to(device)) if self._sphere_template is not None else (None, None)
                if verts_sph_front is not None:
                    # FRONT spheres RGB
                    pos_homo_front_s = Utils.to_homo_torch(verts_sph_front)
                    pos_clip_front_s = (mtx_front[:, None] @ pos_homo_front_s[..., None])[..., 0]
                    color_front_s = Utils.nvdiff_rasterize(self.glctx, mesh_sph, pos_clip_front_s, (H, W))
                    # SIDE spheres RGB
                    verts_sph_side = torch.matmul(verts_sph_front, view_mat[:3, :3]) + view_mat[:3, 3]
                    pos_homo_side_s = Utils.to_homo_torch(verts_sph_side)
                    pos_clip_side_s = (mtx_front[:, None] @ pos_homo_side_s[..., None])[..., 0]
                    color_side_s = Utils.nvdiff_rasterize(self.glctx, mesh_sph, pos_clip_side_s, (H, W))

                    # Composite RGB by overdrawing spheres where they have coverage
                    def composite_over(base, add):
                        # replace this with mixed render 
                        mask = (add.sum(dim=-1, keepdim=True) > 0)
                        return torch.where(mask, add*0.7 + base*0.3, base)

                    rend_front = composite_over(color_front_base, color_front_s)
                    rend_side = composite_over(color_side_base, color_side_s)
                else:
                    rend_front = color_front_base
                    rend_side = color_side_base
            else:
                rend_front = color_front_base
                rend_side = color_side_base

            # Crop both views before concatenation
            y0 = int(H * crop_y0_ratio)
            x0 = int(W * crop_x0_ratio)
            x1 = int(W * crop_x1_ratio)
            rend_front = rend_front[:, y0:H, x0:x1]
            rend_side = rend_side[:, y0:H, x0:x1]
            rend_combined = torch.cat([rend_front, rend_side], dim=1)
            rend_combined = (rend_combined * 255).byte().cpu().numpy()
            return rend_combined

        # Load errors from pkl file
        errors_all = None
        if args.error_file is not None:
            if args.error_file.endswith('.json'):
                # convert to pkl file
                args.error_file = args.error_file.replace('.json', '.pkl').replace('results/', 'results/raw/')
            errors_all = joblib.load(args.error_file)
            
        # chunked processing to avoid OOM
        cs = int(getattr(args, 'chunk_size', 300))
        N = len(inds_all)
        # add contact text
        packed_file = f'/home/xianghuix/datasets/behave/behave-packed/{seq}_GT-packed.pkl'
        if osp.isfile(packed_file):
            packed_data = joblib.load(packed_file)
            dists_h2o_gt = packed_data['dists_h2o'][0]
        else:
            dists_h2o_gt = np.zeros((len(inds_all), 52))
            print(f'{packed_file} does not exist, using dummy dists_h2o_gt')
        try:
            for s in tqdm(range(0, N, cs), desc=f'visualizing {seq} in chunks of {cs}'):
                e = min(s + cs, N)
                inds = inds_all[s:e]

                pose_pr = pose_pr_all[inds]
                pose_gt = pose_gt_all[inds]
                pose_in = pose_in_all[inds]
                # object verts
                R_pr = pose_pr[:, :3, :3]
                t_pr = pose_pr[:, :3, 3]
                R_in = pose_in[:, :3, :3]
                
                R_gt = pose_gt[:, :3, :3]
                t_gt = pose_gt[:, :3, 3]
                t_in = pose_in[:, :3, 3]
                verts_obj_pr = torch.matmul(obj_base[None], R_pr.permute(0, 2, 1)) + t_pr[:, None]
                verts_obj_gt = torch.matmul(obj_base_gt[None], R_gt.permute(0, 2, 1)) + t_gt[:, None]
                verts_obj_in = torch.matmul(obj_base[None], R_in.permute(0, 2, 1)) + t_in[:, None]
                if has_smpl:
                    vs_pr, _j_pr, _a, _b = body_model(smpl_pose_pr_all[inds], betas_pr_all[inds], smpl_t_pr_all[inds])
                    vs_gt, _j_gt, _a2, _b2 = body_model(smpl_pose_gt_all[inds], betas_gt_all[inds], smpl_t_gt_all[inds])
                    vs_in, _j_in, _a3, _b3 = body_model(smpl_pose_in_all[inds], betas_in_all[inds], smpl_t_in_all[inds])
                    verts_comb_pr = torch.cat([vs_pr, verts_obj_pr], dim=1)
                    verts_comb_gt = torch.cat([vs_gt, verts_obj_gt], dim=1)
                    verts_comb_in = torch.cat([vs_in, verts_obj_in], dim=1)
                else:
                    verts_comb_pr = verts_obj_pr
                    verts_comb_gt = verts_obj_gt
                    verts_comb_in = verts_obj_in
                # Optional baseline: NLF (SMPL) + FP (object) rendering
                have_baseline = has_smpl and has_nlf and has_fp
                assert not have_baseline, 'TODO: implement NLF + FP rendering'
                if have_baseline:
                    # build FP object poses for current chunk
                    pose_fp_chunk = []
                    for gi in inds:
                        _, fname = frames[gi].split('/')
                        pose_fp_chunk.append(time_to_fp.get(fname, np.eye(4, dtype=np.float32)))
                    pose_fp_chunk = torch.from_numpy(np.matmul(np.stack(pose_fp_chunk, 0), gt_to_perturb_pose)).to('cuda').float()
                    R_fp = pose_fp_chunk[:, :3, :3]
                    t_fp = pose_fp_chunk[:, :3, 3]
                    verts_obj_fp = torch.matmul(obj_base[None], R_fp.permute(0, 2, 1)) + t_fp[:, None]

                    # build NLF SMPL verts for current chunk
                    nlf_idx = []
                    for gi in inds:
                        _, fname = frames[gi].split('/')
                        nlf_idx.append(time_to_nlf.get(fname, -1))
                    # prepare tensors; for missing indices, fill zeros to avoid errors
                    valid_mask = torch.as_tensor([i >= 0 for i in nlf_idx], device='cuda')
                    idx_arr = np.array([i if i >= 0 else 0 for i in nlf_idx], dtype=np.int64)
                    thetas_chunk = torch.from_numpy(thetas_all[idx_arr]).to('cuda').float()
                    betas_chunk = torch.from_numpy(betas_avg_all[idx_arr]).to('cuda').float()
                    trans_chunk = torch.from_numpy(transls_all[idx_arr]).to('cuda').float()
                    vs_nlf, _jn, _an, _bn = body_model(thetas_chunk, betas_chunk, trans_chunk)
                    # zero out invalid ones
                    if (~valid_mask).any():
                        vs_nlf = vs_nlf * valid_mask[:, None, None]
                    verts_comb_nf = torch.cat([vs_nlf, verts_obj_fp], dim=1)
                    rend_nf = batch_render(verts_comb_nf, crop_y0_ratio=0.15, crop_x0_ratio=0.15, crop_x1_ratio=0.85)
                else:
                    rend_nf = None

                # add spheres
                sphere_indices = torch.from_numpy(np.array([7, 8, 10, 11, 20, 21])).to('cuda') # only feet and hands
                centers = _j_pr[:, sphere_indices]
                colors24_th = torch.from_numpy(colors24).to('cuda')/255.
                if 'contact_dist' in data_pr and not args.no_sphere:
                    contact_dim = data_pr['contact_dist'].shape[-1]
                    assert contact_dim in [8, 24, 52, 32], f"Invalid contact dimension: {contact_dim}"
                    if contact_dim in [24, 52]:
                        sphere_indices = torch.arange(contact_dim).to('cuda')
                        contact_dist_pr = data_pr['contact_dist'].cuda()[inds][:, sphere_indices]  # (B, J)
                        contact_dist_gt = data_gt['contact_dist'].cuda()[inds][:, sphere_indices]  # (B, J)
                    elif contact_dim == 8:
                        # only the 8 ending joints 
                        sphere_indices = torch.tensor([7, 8, 10, 11, 20, 21, 22, 23+15]).to('cuda')
                        contact_dist_pr = data_pr['contact_dist'].cuda()[inds] # use all of them 
                        contact_dist_gt = data_gt['contact_dist'].cuda()[inds] # use all of them 
                    elif contact_dim == 32:
                        # only the fist 8 points 
                        sphere_indices = torch.tensor([7, 8, 10, 11, 20, 21, 22, 23+15]).to('cuda')
                        contact_dist_pr = data_pr['contact_dist'].cuda()[inds][:, :8] # use first 8 points 
                        contact_dist_gt = data_gt['contact_dist'].cuda()[inds][:, :8] # use first 8 points 
                    
                    contact_dist_pr[contact_dist_pr > 0.2] = 0.001 # don't use these large values 
                    spheres_pr = {
                        'centers': _j_pr[:, sphere_indices],
                        'radii': contact_dist_pr,
                        'colors': colors24_th[sphere_indices][None].repeat(centers.shape[0], 1, 1),
                    }
                    
                    contact_dist_gt[contact_dist_gt > 0.2] = 0.001
                    spheres_gt = {
                        'centers': _j_gt[:, sphere_indices],
                        'radii': contact_dist_gt,
                        'colors': colors24_th[sphere_indices][None].repeat(centers.shape[0], 1, 1),
                    }
                    print(f"{len(sphere_indices)} contact spheres added")
                else:
                    spheres_pr, spheres_gt = None, None
                
                crop_y0_ratio=0.15  if "Date03" in seq and not args.wild_video else 0 # only cut for behave but not wild video  
                cut_x0_ratio = 0.15 if not args.wild_video else 0
                cut_x1_ratio = 0.85 if not args.wild_video else 1.0
                rend_pr = batch_render(verts_comb_pr, crop_y0_ratio=crop_y0_ratio, crop_x0_ratio=cut_x0_ratio, crop_x1_ratio=cut_x1_ratio, mesh_tensors=mesh_tensors, spheres=spheres_pr)
                rend_gt = batch_render(verts_comb_gt, crop_y0_ratio=crop_y0_ratio, crop_x0_ratio=cut_x0_ratio, crop_x1_ratio=cut_x1_ratio, mesh_tensors=mesh_tensors_gt, spheres=spheres_gt)
                rend_in = batch_render(verts_comb_in, crop_y0_ratio=crop_y0_ratio, crop_x0_ratio=cut_x0_ratio, crop_x1_ratio=cut_x1_ratio, mesh_tensors=mesh_tensors)
                # overlay and write
                for j, gi in enumerate(tqdm(inds)):
                    _, fname = frames[gi].split('/')
                    t = float(fname.replace('t', '')) if args.data_source == 'behave' and not args.wild_video else float(fname)
                    actual_times = np.array([controllers[x].get_closest_time(t) for x in kids])
                    best_kid = np.argmin(np.abs(actual_times - t))
                    actual_time = actual_times[best_kid]
                    color = controllers[kid].get_closest_frame(actual_time)
                    color = cv2.resize(color, (W, H))
                    if fname == 't0005.800':
                        verts_comb_pr_i = verts_comb_pr[j]
                        trimesh.PointCloud(verts_comb_pr_i[:, :3].cpu().numpy()).export(f'outputs/viz/{seq}_t0005.800_pr.ply')
                        print("Object Pose:", pose_pr[j], 'first 10 verts:', obj_base[:10])

                    rp = rend_pr[j]  # already cropped inside renderer: (2*Hc, Wc, 3)
                    rg = rend_gt[j]
                    ri = rend_in[j]
                    
                    # crop color to match renderer crop
                    y0 = int(crop_y0_ratio * H)  # only cut for behave but not wild video  
                    x0 = int(cut_x0_ratio * W)
                    x1 = int(cut_x1_ratio * W)
                    color_c = color[y0:H, x0:x1]
                    Hc, Wc = color_c.shape[:2]

                    # build input panel from cropped halves (front + side both use same crop)
                    input_panel = np.concatenate([color_c, color_c], axis=0)

                    # overlay front view (top half), keep side view (bottom half) as pure render
                    alpha = 0.7
                    pred_panel = input_panel.copy()
                    rp_front, rp_side = rp[:Hc], rp[Hc:]
                    maskp_top = (rp_front.sum(axis=-1, keepdims=True) > 0)
                    pred_panel[:Hc][maskp_top[..., 0]] = (
                        alpha * rp_front[maskp_top[..., 0]] + (1 - alpha) * pred_panel[:Hc][maskp_top[..., 0]]
                    ).astype(np.uint8)
                    pred_panel[Hc:] = rp_side

                    gt_panel = input_panel.copy()
                    rg_front, rg_side = rg[:Hc], rg[Hc:]
                    maskg_top = (rg_front.sum(axis=-1, keepdims=True) > 0)
                    gt_panel[:Hc][maskg_top[..., 0]] = (
                        alpha * rg_front[maskg_top[..., 0]] + (1 - alpha) * gt_panel[:Hc][maskg_top[..., 0]]
                    ).astype(np.uint8)
                    gt_panel[Hc:] = rg_side

                    # add nlf + fp (data_in) panel
                    in_panel = input_panel.copy()
                    ri_front, ri_side = ri[:Hc], ri[Hc:]
                    maski_top = (ri_front.sum(axis=-1, keepdims=True) > 0)
                    in_panel[:Hc][maski_top[..., 0]] = (
                        alpha * ri_front[maski_top[..., 0]] + (1 - alpha) * in_panel[:Hc][maski_top[..., 0]]
                    ).astype(np.uint8)
                    in_panel[Hc:] = ri_side

                    if rend_nf is not None:
                        rn = rend_nf[j]
                        nf_panel = input_panel.copy()
                        rn_front, rn_side = rn[:Hc], rn[Hc:]
                        maskn_top = (rn_front.sum(axis=-1, keepdims=True) > 0)
                        nf_panel[:Hc][maskn_top[..., 0]] = (
                            alpha * rn_front[maskn_top[..., 0]] + (1 - alpha) * nf_panel[:Hc][maskn_top[..., 0]]
                        ).astype(np.uint8)
                        nf_panel[Hc:] = rn_side
                        views = [input_panel, nf_panel, pred_panel, gt_panel] if not args.wild_video else [input_panel, nf_panel, pred_panel]
                        comb = np.concatenate(views, axis=1)
                        # labels based on cropped sizes
                        y_top = 40
                        y_bot = Hc + 40
                        cv2.putText(comb, 'input-front', (Wc // 6, y_top), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                        cv2.putText(comb, 'nlf+fp-front', (Wc + Wc // 6, y_top), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                        cv2.putText(comb, 'pred-front', (2 * Wc + Wc // 6, y_top), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                        cv2.putText(comb, 'gt-front', (3 * Wc + Wc // 6, y_top), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                        cv2.putText(comb, 'input-side', (Wc // 6, y_bot), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                        cv2.putText(comb, 'nlf+fp-side', (Wc + Wc // 6, y_bot), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                        cv2.putText(comb, 'pred-side', (2 * Wc + Wc // 6, y_bot), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                        cv2.putText(comb, 'gt-side', (3 * Wc + Wc // 6, y_bot), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                        cv2.putText(comb, fname, (2 * Wc-Wc//6, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                    else:
                        views = [input_panel, in_panel, pred_panel, gt_panel] if not args.wild_video else [input_panel, in_panel, pred_panel]
                        comb = np.concatenate(views, axis=1)
                        y_top = 40
                        y_bot = Hc + 40
                        # add text for the data_in panel as well, so now each row is 4 elements, the position should be divied by 8 instead of 6 now 
                        cv2.putText(comb, 'input-front', (Wc // 6, y_top), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                        cv2.putText(comb, 'NLF+FP-front', (Wc + Wc // 6, y_top), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                        cv2.putText(comb, 'pred-front', (2 * Wc + Wc // 6, y_top), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                        cv2.putText(comb, 'gt-front', (3 * Wc + Wc // 6, y_top), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                        cv2.putText(comb, 'input-side', (Wc // 6, y_bot), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                        cv2.putText(comb, 'NLF+FP-side', (Wc + Wc // 6, y_bot), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                        cv2.putText(comb, 'pred-side', (2 * Wc + Wc // 6, y_bot), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                        cv2.putText(comb, 'gt-side', (3 * Wc + Wc // 6, y_bot), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                        cv2.putText(comb, fname, (Wc, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
                    
                    # add error text 
                    if errors_all is not None:
                        err_idx = errors_all['frames'].index(f'{seq}/{fname}')
                        err_rot = errors_all['rot_errors'][err_idx] * 180 / np.pi
                        err_t = errors_all['transl_errors'][err_idx] * 100 
                        err_v2v = errors_all['v2v_errors'][err_idx] * 100 
                        err_mpjae = errors_all['mpjae_errors'][err_idx] * 180 / np.pi
                        err_mpjpe = errors_all['mpjpe_errors'][err_idx] * 100 
                        err_smpl_t = errors_all['smpl_t_errors'][err_idx] * 100 
                        err_text = f'obj_R: {err_rot:.3f}, obj_T: {err_t:.3f}, v2v: {err_v2v:.3f}, mpjae: {err_mpjae:.3f}, mpjpe: {err_mpjpe:.3f}, smpl_t: {err_smpl_t:.3f}'
                        cv2.putText(comb, err_text, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    
                    # add contact dist text
                    dist_hands = dists_h2o_gt[gi, [22, 23+15]]
                    dist_text = f'lhand: {dist_hands[0]:.3f}, rhand: {dist_hands[1]:.3f}'
                    cv2.putText(comb, dist_text, (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    vw.append_data(comb)

                # free chunk tensors
                del verts_comb_pr, verts_comb_gt, verts_obj_pr, verts_obj_gt, rend_pr, rend_gt
                if 'have_baseline' in locals() and have_baseline:
                    del verts_obj_fp, verts_comb_nf, rend_nf, pose_fp_chunk, vs_nlf, thetas_chunk, betas_chunk, trans_chunk
                torch.cuda.empty_cache()
        finally:
            vw.close()
            print(f'saved to {out_path}')
    
    




def main():
    parser = PredVisualizer.get_parser()
    args = parser.parse_args()
    args.nodepth = True

    visualizer = PredVisualizer(args)
    if osp.isfile(args.pred_file):
        visualizer.visualize(args)
    else:
        files = sorted(glob.glob(args.pred_file))
        print(f'found {len(files)} files')
        for file in files:
            args.pred_file = file
            visualizer.visualize(args)

    
    


if __name__ == '__main__':
    main()