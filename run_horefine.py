# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import json
import os, sys

sys.path.append(os.getcwd())

import os.path as osp
import joblib
import numpy as np
import torch
import cv2
import imageio
import time
from tqdm import tqdm
import nvdiffrast.torch as dr
from pytorch3d.renderer import look_at_view_transform
from scipy.spatial.transform import Rotation as R
from typing import Any

import Utils
from Utils import load_smpl_obj_uvmap
from tools import img_utils
from behave_data.utils import load_kinect_poses_back, init_video_controllers
from lib_smpl import get_smpl
from lib_smpl.body_landmark import BodyLandmarks
from behave_data.const import _sub_gender
from behave_data.behave_video import load_masks
from prep.render_fp_nlf import BehaveFPNLFRenderer
from tools.eval_base import ModelEvaluator
from learning.training.trainer import Trainer
from lib_smpl import pose156to72, pose72to156, SMPL_ASSETS_ROOT
import h5py


class HORefineRunner(BehaveFPNLFRenderer):
    "refine both human and object"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prepare_video_mask_loader(self, args, kids, video_prefix, cfg):
        args.nodepth = False
        controllers, _ = init_video_controllers(args, args.video, kids)

        # read h5 file
        h5_path = f'{cfg.masks_root}/{video_prefix}_masks_k{args.cam_id}.h5'
        print(f'loading masks from {h5_path}')
        tar_mask = h5py.File(h5_path, 'r')
        return controllers, tar_mask

    @torch.no_grad()
    def run(self, args, cfg):
        "refine both human and object of one video given by cfg.video"
        # step 1: initialize a trainer the same way as in tools/eval_base.py, this will load the model and set to eval mode
        self.cfg = cfg
        cfg.job = 'test-only'
        cfg.no_wandb = True
        trainer = Trainer(cfg)
        trainer.model.eval()
        evaluator = ModelEvaluator(cfg)
        args.cam_id = cfg.cam_id
        err_keys = ['rot', 'transl', 'mpjpe', 'v2v', 'mpjae', 'smpl_t']
        errors_all = {k: [] for k in err_keys}

        # step 2: load FP pose and NLF SMPL poses from files
        args.video = cfg.video
        self.run_1seq(args, cfg, evaluator, trainer, errors_all, [])

    @torch.no_grad()
    def run_1seq(self, args, cfg, evaluator, trainer, errors_all, frames_all):
        device = 'cuda'
        video_prefix = osp.basename(args.video).split('.')[0]
        seq_name = video_prefix
        save_name = evaluator.get_save_name(cfg, trainer)
        pth_file = f'{cfg.outpath}/{save_name}/{seq_name}.pth'
        if osp.isfile(pth_file):
            print(f'{pth_file} already exists, skipping')
            return

        fp_root = cfg.fp_root
        fp_data = joblib.load(osp.join(fp_root, f'{seq_name}_all.pkl'))
        fp_poses = fp_data['fp_poses']  # (T, K, 4, 4)
        fp_frames = fp_data['frames']
        kids = [cfg.cam_id]
        w2c_rots = [np.eye(3) for _ in kids]
        w2c_trans = [np.zeros(3) for _ in kids]
        obj_name = seq_name.split('_')[2]
        mesh_tensors, meshes = load_smpl_obj_uvmap(seq_name, use_hy3d=True, meshes_root=cfg.hy3d_meshes_root)
        meshes_any: Any = meshes
        glctx = dr.RasterizeCudaContext()
        render_size = (args.rend_size, args.rend_size)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        # object-only tensors for mask computation/render (no texture path to avoid None textures)
        obj_idx = 1
        verts_obj_base_t = meshes_any[obj_idx].verts_padded()[0].to(device).float()
        tex = meshes[obj_idx].textures
        uv = tex.verts_uvs_padded()[0]
        uv[:, 1] = 1 - uv[:, 1]
        mesh_tensors_obj = {
            "tex": tex.maps_padded().to(device).float(),  # this must be (1, H, W, 3)
            'uv_idx': torch.tensor(tex.faces_uvs_padded()[0], device=device, dtype=torch.int),
            # correct way to get face uvs
            'uv': uv.to(device).float(),
            'pos': meshes[obj_idx].verts_padded()[0].to(device).float(),
            'faces': torch.tensor(meshes[obj_idx].faces_padded()[0], device=device, dtype=torch.int),
            'vnormals': meshes[obj_idx].verts_normals_padded()[0].to(device).float(),
        }
        # step 3: load one batch of images and human object masks and compute ROI/Kroi
        t0 = time.time()
        controllers, tar_mask = self.prepare_video_mask_loader(args, kids, video_prefix, cfg)
        t1 = time.time()
        print(f'mask pre-loading time: {t1 - t0:.4f}s')
        gt_to_perturb_pose = np.eye(4)

        if not cfg.wild_video:
            center = np.zeros(3) # the mesh is already aligned with GT, hence no recentering needed
        else:
            center = np.mean(verts_obj_base_t.detach().cpu().numpy(), axis=0)  # simply the center
        print("Using center: ", center)
        gt_to_perturb_pose[:3, 3] = center
        verts_obj_base = verts_obj_base_t.detach().cpu().numpy() - center  # need to subtract center
        enum_idx, kid = 0, cfg.cam_id
        
        nlf_data = joblib.load(f'{cfg.nlf_root}/{video_prefix}_params.pkl')

        # prepare dummy GT data
        packed = {}
        packed['obj_angles'] = R.from_matrix(np.eye(3)).as_rotvec()[None].repeat(len(nlf_data['poses']), 0)
        packed['obj_trans'] = np.zeros((len(nlf_data['poses']), 3))
        # convert from (T, K...) to (T,...)
        packed['poses'] = nlf_data['poses'][:, 0].copy()
        packed['betas'] = nlf_data['betas'][:, 0].copy()
        packed['trans'] = nlf_data['transls'][:, 0].copy()
        packed['frames'] = nlf_data['frames']
        
        frames_packed = packed['frames']
        # take the frames as the joint set
        frames_packed = [x for x in frames_packed if x in nlf_data['frames']]
        body_model = get_smpl(_sub_gender[video_prefix.split('_')[1]], hands=True).to(device)
        betas_avg = np.mean(packed['betas'].reshape((-1, 10)), 0)
        mesh_diameter = self.get_smpl_diameter(betas_avg, body_model)
        clip_len = int(getattr(cfg, 'clip_len', 16))
        start = getattr(args, 'start', 0)
        end = min(start + clip_len, len(fp_frames))
        # Load packed GT once for frame availability and later GT usage
        # Write visualization video: overlay front-view rendering on input RGB
        out_root = cfg.video_out
        os.makedirs(out_root, exist_ok=True)
        save_name = evaluator.get_save_name(cfg, trainer)
        out_path = osp.join(out_root, f'{save_name}+{seq_name}_it{cfg.refine_iters}.mp4')
        vw = imageio.get_writer(out_path, fps=30)
        landmark = BodyLandmarks(SMPL_ASSETS_ROOT)
        iterations = cfg.refine_iters
        H_full, W_full = None, None
        # For evaluation
        keys = ['pose_abs', 'smpl_pose', 'smpl_t', 'frames', 'betas', 'verts', 'contact_logits']
        data_gt, data_pr, data_in = {k: [] for k in keys}, {k: [] for k in keys}, {k: [] for k in keys}
        self.side_view_z = None # to render side view

        vis_input = True
        if vis_input:
            out_path = osp.join(out_root, f'{save_name}+{seq_name}_it{cfg.refine_iters}_input.mp4')
            vw_input = imageio.get_writer(out_path, fps=15)
        for start in tqdm(range(0, len(frames_packed), clip_len)):
            end = min(start + clip_len, len(frames_packed))
            if end - start < clip_len:
                print(f'not sufficient frames for one window: {start}->{end}, updating clip length and window to {end - start}')
                # update everything
                clip_len = end - start 
                window = clip_len
                cfg.clip_len = clip_len
                cfg.window = clip_len
            clip_len = end - start  # actual clip length
            pose_init_list, poseA_norm_list, K_rois, frames_used = [], [], [], []
            poses_perturbed = []
            full_colors = []  # store original RGB frames to reuse in visualization

            prep = {}
            # NLF-based SMPL fields: nlf_rotmat, nlf_transl, betas_gt, joints_nlf
            nlf_inds = np.array([nlf_data['frames'].index(x.split('/')[-1]) for x in frames_packed[start:end]])

            poses_nlf_init = nlf_data['poses'][nlf_inds, enum_idx].astype(np.float32)  # (T, 72)
            trans_nlf_init = nlf_data['transls'][nlf_inds, enum_idx].astype(np.float32)  # (T, 3)
            betas = nlf_data['betas'][:, enum_idx].copy()  # TODO: replace with GT DATA!
            betas_avg = np.mean(betas, axis=0)[None].repeat(len(poses_nlf_init), axis=0)

            # GT params
            poses_full = packed['poses'][start:end].astype(np.float32)
            betas_gt = packed['betas'][start:end].astype(np.float32)
            verts_nlf_render_init = body_model(torch.from_numpy(poses_nlf_init).to(device),  # match
                                               torch.from_numpy(betas_gt).to(device),  # match
                                               torch.from_numpy(trans_nlf_init).to(device))[0].cpu().numpy()

            # joints from landmarks
            joints_nlf_np = landmark.get_body_kpts_batch(verts_nlf_render_init)  # (T, 25, 3)
            prep['joints_nlf'] = torch.from_numpy(joints_nlf_np)[None].to(device).float()
            # rotation matrices per joint
            NJ = 52
            nlf_rot_np_init = R.from_rotvec(poses_nlf_init.reshape(-1, 3)).as_matrix().astype(np.float32).reshape(-1,
                                                                                                                  NJ, 3,
                                                                                                                  3)
            prep['nlf_rotmat'] = torch.from_numpy(nlf_rot_np_init).to(device).float()[:, :24]  # (BT, J, 3, 3)
            prep['nlf_transl'] = torch.from_numpy(trans_nlf_init)[None].to(device).float()  # matches
            prep['betas_gt'] = torch.from_numpy(betas_gt)[None].to(device).float()
            prep['betas_nlf'] = torch.from_numpy(betas_avg)[None].to(device).float()
            prep['nlf_poses'] = torch.from_numpy(poses_nlf_init).to(device).float()

            # Load RGB and masks
            input_rgbms, input_xyzs, bboxes = [], [], []
            for i in tqdm(range(start, end)):
                frame_time = frames_packed[i]
                if frame_time not in fp_frames:
                    print(f'Frame {frame_time} not found in FP frames!')
                    continue
                idx_fp = fp_frames.index(frame_time)
                frames_used.append(f'{seq_name}/{frame_time}')
                pose_fp = np.matmul(fp_poses[idx_fp, enum_idx], gt_to_perturb_pose)

                mask_h, mask_o = load_masks(video_prefix, frame_time, kid, tar_mask)
                if mask_h is None:
                    continue

                # compute crop params from mask
                bmin, bmax = img_utils.masks2bbox([mask_h, mask_o])
                center_2d = (bmax + bmin) / 2
                radius = np.max(bmax - bmin) * 1.1 / 2
                top_left = center_2d - radius
                bottom_right = center_2d + radius
                K_roi = self.Kroi_from_corners(bottom_right, top_left) # TODO: reload this each time when new video seq come in, especially for wild video!  

                # Load RGB and depth for input
                t2 = time.time()
                t = float(frame_time[1:])
                actual_times = np.array([controllers[x].get_closest_time(t) for x, _ in enumerate(kids)])
                best_kid = np.argmin(np.abs(actual_times - t))
                actual_time = actual_times[best_kid]
                color, depth = controllers[enum_idx].get_closest_frame(actual_time)
                H_full, W_full = color.shape[:2]
                t3 = time.time()
                full_colors.append(np.asarray(color))
                # ensure proper stacking with correct dtype
                color_np = np.asarray(color, dtype=np.uint8)
                mask_h_np = mask_h.astype(np.uint8)
                mask_o_np = mask_o.astype(np.uint8)
                color = np.concatenate([color_np, mask_h_np[:, :, None], mask_o_np[:, :, None]], axis=-1)
                bbox = np.hstack((top_left.astype(np.float32), bottom_right.astype(np.float32)))
                dmap_xyz, rgbm = self.crop_color_dmap(bbox, color, depth, render_size)  # (H, W, 3), (H, W, 5)

                input_rgbms.append(rgbm)
                input_xyzs.append(dmap_xyz)
                K_rois.append(K_roi)
                poses_perturbed.append(pose_fp.copy())
                bboxes.append(bbox)
            # Initialization
            B_in_cams_init = np.stack(poses_perturbed, axis=0).copy()
            B_in_cams = torch.from_numpy(np.stack(poses_perturbed, axis=0).copy()).to(device).float()[None]
            verts_nlf_render = verts_nlf_render_init
            poses_nlf, betas_nlf, trans_nlf = poses_nlf_init, betas_gt, trans_nlf_init

            # Render batch, which could be repeated iteratively
            for it in range(iterations):
                B_in_cams = B_in_cams.cpu().numpy()[0]
                verts_obj_batch = [np.matmul(verts_obj_base, pose_fp[:3, :3].T) + pose_fp[:3, 3] for pose_fp in B_in_cams]
                verts_hum_batch = verts_nlf_render.copy()

                # Final input to the network
                input_rgbs_final, input_xyz_final = [], []
                render_rgbs, render_xyz = [], []
                for fi in range(len(verts_obj_batch)):  # this is super fast, 4s
                    # step 4: render combined (SMPL + object) inside ROI
                    vh = verts_hum_batch[fi]  # (N_h, 3)
                    vo = verts_obj_batch[fi]
                    mesh_tensors['pos'] = torch.from_numpy(np.concatenate([vh, vo], 0)).float().cuda()
                    mesh_tensors_obj['pos'] = torch.from_numpy(vo).float().cuda()
                    bbox2d_ori = torch.tensor([[0, 0., render_size[0], render_size[1]]], device=device).repeat(1, 1)
                    extra = {}

                    t2 = time.time()
                    # TODO: do batch rendering
                    rgb_r, depth_r, _ = Utils.nvdiffrast_render(K=np.stack([K_rois[fi]], 0), H=render_size[1],
                                                                W=render_size[0],
                                                                ob_in_cams=torch.as_tensor(np.eye(4)[None]).float(),
                                                                context='cuda',
                                                                get_normal=False, glctx=glctx,
                                                                mesh_tensors=mesh_tensors,
                                                                output_size=render_size, bbox2d=bbox2d_ori,
                                                                use_light=True,
                                                                extra=extra)
                    rgb_obj, depth_obj, _ = Utils.nvdiffrast_render(K=np.stack([K_rois[fi]], 0), H=render_size[1],
                                                                    W=render_size[0],
                                                                    ob_in_cams=torch.as_tensor(np.eye(4)[None]).float(),
                                                                    context='cuda',
                                                                    get_normal=False, glctx=glctx,
                                                                    mesh_tensors=mesh_tensors_obj,
                                                                    output_size=render_size, bbox2d=bbox2d_ori,
                                                                    use_light=True, extra=extra)
                    rgbs = (rgb_r.cpu().numpy() * 255).astype(np.uint8)
                    dmaps = depth_r.cpu().numpy()

                    dmap_full = depth_r[0].cpu().numpy()
                    dmap_obj = depth_obj[0].cpu().numpy()
                    mask_rend_o = (dmap_obj <= dmap_full) & (dmap_obj > 0)
                    mask_o_full = dmap_obj > 0

                    dmap_xyz_init_np = Utils.depth2xyzmap(dmap_full, K_rois[fi])
                    # unify to torch then store as numpy (H, W, 3)
                    dmap_xyz_init = torch.from_numpy(dmap_xyz_init_np).permute(2, 0, 1).float()  # (3, H, W)
                    # store temporarily as dict to allow mask augmentation below
                    # render_xyz.append(dmap_xyz_init_t.cpu().numpy())
                    t3 = time.time()

                    input_data = {
                        'rgbmB': input_rgbms[fi].cpu().numpy().astype(np.uint8).copy(),
                        'xyzB': input_xyzs[fi].cpu().numpy().astype(np.float16).copy()
                    }
                    render_data = {
                        'rgba': rgbs[0].copy(),
                        'depth': dmaps[0].astype(np.float16).copy(),
                        'K_roi': K_rois[fi],
                        'bbox': bboxes[fi],
                        'mask_o': np.stack([mask_rend_o, mask_o_full], -1).copy(),
                    }
                    rgb_render = render_data['rgba']
                    if vis_input:
                        vis = np.concatenate([input_data['rgbmB'][:, :, :3], render_data['rgba'][:, :, :3], 
                        cv2.addWeighted(input_data['rgbmB'][:, :, :3], 0.5, render_data['rgba'][:, :, :3], 0.5, 0)], 1)
                        vw_input.append_data(vis) # rendering is correct 

                    dmap_xyz, dmap_xyz_a, rgb = trainer.dataset_test.process_input(dmap_xyz_init.clone(), fi,
                                                                                   input_data, mesh_diameter, trans_nlf,
                                                                                   poses_perturbed[fi], render_data,
                                                                                   render_data['rgba'], frames_used[-1],
                                                                                   kid)
                    render_rgbs.append(rgb_render.copy().transpose(2, 0, 1) / 255.)
                    render_xyz.append(dmap_xyz_a)
                    input_xyz_final.append(dmap_xyz)
                    input_rgbs_final.append(rgb)

                # step 5: organize data into a batch (single item)
                poseA_norm = B_in_cams.copy()
                poseA_norm[:, :3, 3] *= 2 / mesh_diameter

                mesh_diam_tensor = \
                torch.as_tensor([mesh_diameter] * len(poses_perturbed), device=device, dtype=torch.float)[None]
                trans_norm = torch.as_tensor(np.array(args.trans_normalizer), device=device, dtype=torch.float).repeat(
                    len(poses_perturbed), 1)[None]

                obj_name = seq_name.split('_')[2]
                batch = {
                    **prep, # this contains human pose information
                    "input_rgbs": torch.stack(input_rgbs_final, 0).cuda().float()[None],  # this is rgbB
                    'render_rgbs': torch.from_numpy(np.stack(render_rgbs, axis=0)).cuda().float()[None],
                    # this is rgbAs
                    'input_xyz': torch.stack(input_xyz_final, 0).float().cuda()[None],
                    'render_xyz': torch.stack(render_xyz, 0).float().cuda()[None],
                    'mesh_diameter': mesh_diam_tensor,  # not exactly right
                    'trans_normalizer': trans_norm.reshape(1, len(poses_perturbed), 3),
                    'poseA_norm': torch.from_numpy(poseA_norm).float().cuda()[None],
                    'pose_perturbed': torch.from_numpy(B_in_cams)[None].float().cuda(),
                    # matches

                    # dummy data
                    'delta_transl': torch.zeros((1, len(poses_perturbed), 3), device=device),
                    'delta_rot': torch.zeros((1, len(poses_perturbed), 3, 3), device=device),

                    'K_rois': torch.from_numpy(np.stack(K_rois)).float().cuda()[None],  # match

                    'smpl_poses_gt': torch.from_numpy(poses_full).float().cuda()[None],
                    # this does not align with dataloader one
                    'smpl_transl_gt': torch.from_numpy(packed['trans'][start:end]).float().cuda()[None],  # match
                    'betas_gt': torch.from_numpy(betas_gt).float().cuda()[None],
                }

                # Compute GT object pose in camera coordinates (B, T, 4, 4)
                # We already have R_cam and t_cam below when preparing GT rendering. Reuse packed loaded above.
                angles_gt = packed['obj_angles'][start:end].astype(np.float32)
                transl_gt = packed['obj_trans'][start:end].astype(np.float32)
                R_wc = torch.from_numpy(w2c_rots[enum_idx]).to(device).float()
                t_wc = torch.from_numpy(w2c_trans[enum_idx]).to(device).float()
                R_obj = torch.from_numpy(R.from_rotvec(angles_gt).as_matrix()).to(device).float()  # (T, 3, 3)
                t_world = torch.from_numpy(transl_gt).to(device).float()  # (T, 3)
                R_cam = torch.matmul(R_wc[None].expand(end - start, -1, -1), R_obj)  # (T, 3, 3)
                t_cam = torch.matmul(t_world, R_wc.T) + t_wc  # (T, 3)
                pose_gt_mat = torch.eye(4, device=device, dtype=torch.float)[None].repeat(end - start, 1, 1)
                pose_gt_mat[:, :3, :3] = R_cam
                pose_gt_mat[:, :3, 3] = t_cam
                batch['pose_gt'] = pose_gt_mat[None].clone()  # (B, T, 4, 4)

                poseA = batch['pose_perturbed']
                poseB = batch['pose_gt']
                # compute real delta
                batch['delta_transl'] = poseB[:, :, :3, 3] - poseA[:, :, :3, 3]
                batch['delta_rot'] = torch.matmul(poseB[:, :, :3, :3], poseA[:, :, :3, :3].permute(0, 1, 3, 2))

                # the code after this line is correct!
                rot, rot_delta_gt, trans_delta_gt, trans_delta_pred, output = trainer.forward_batch(batch, cfg,
                                                                                                    trainer.model,
                                                                                                    ret_dict=True,
                                                                                                    vis=False)

                # update object pose
                prep = {}
                B_in_cams, _ = trainer.compute_abspose(poseA.shape[0], batch, cfg, poseA, rot,
                                                       rot_delta_gt,
                                                       trans_delta_gt, trans_delta_pred, output)
                if cfg.use_intermediate:
                    B_in_cams = trainer.abspose_from_relative(batch, cfg, poseA, rot, trans_delta_pred)

                #### Start of update SMPL pose
                betas, pred_smpl_pose, pred_smpl_r, pred_smpl_t = trainer.smpl_params_from_pred(batch, output)
                pred_smpl_pose = pose72to156(pred_smpl_pose)
                # still use the old NLF translation

                verts_nlf_render = body_model(pred_smpl_pose,  # match
                                              torch.from_numpy(betas_gt).to(device),  # match
                                              torch.from_numpy(trans_nlf_init).float().to(device)
                                              )[0].cpu().numpy()
                prep['nlf_transl'] = torch.from_numpy(trans_nlf_init).float().to(device)[None].to(device).float()  # matches

                # joints from landmarks
                joints_nlf_np = landmark.get_body_kpts_batch(verts_nlf_render)  # (T, 25, 3)
                prep['joints_nlf'] = torch.from_numpy(joints_nlf_np)[None].to(device).float()
                # rotation matrices per joint
                poses_nlf = pred_smpl_pose.cpu().numpy()  # (T, 72)
                nlf_rot_np = R.from_rotvec(poses_nlf.reshape(-1, 3)).as_matrix().astype(np.float32).reshape(-1, 52, 3, 3)[:, :24]# prediction has only 24 joints 
                prep['nlf_rotmat'] = torch.from_numpy(nlf_rot_np).to(device).float()  # (BT, J, 3, 3)

                prep['betas_gt'] = torch.from_numpy(betas_gt)[None].to(device).float()
                poses_nlf, trans_nlf = poses_nlf, prep['nlf_transl'].cpu().numpy()[0]  # TODO: update betas if needed
                #### End of update SMPL pose

                # Re-init batch
                print(f'iteration {it} done')
            # step 7: visualize predictions by rendering SMPL + object in batch (similar to tools/viz_pred.py)

            K = self.K_full.copy()
            scale_ratio = 2
            K[:2] /= scale_ratio
            H, W = H_full // scale_ratio, W_full // scale_ratio

            # Build combined vertices in camera space for batch
            # Object verts from predicted pose
            obj_base_centered = verts_obj_base_t - torch.as_tensor(center, device=device, dtype=torch.float)
            R_pred = B_in_cams[0, :, :3, :3].to(device).float()  # (T, 3, 3)
            t_pred = B_in_cams[0, :, :3, 3].to(device).float()  # (T, 3)
            obj_verts_pr = torch.matmul(obj_base_centered[None].expand(end - start, -1, -1),
                                        R_pred.permute(0, 2, 1)) + t_pred[:, None]

            verts_pr = body_model(pred_smpl_pose.to(device),
                                          batch['betas_gt'].reshape(-1, 10).to(device),
                                          pred_smpl_t.to(device))[0]

            verts_comb_pr = torch.cat([verts_pr, obj_verts_pr], dim=1)  # (T, N_total, 3)

            # Convert to clip space and render batch at once
            mtx_front, rend_pr, rend_pr_side, view_mat = self.render_front_side(H, K, W, glctx, mesh_tensors,
                                                                                verts_comb_pr)

            # TODO: Render input
            verts_in = verts_nlf_render_init
            verts_obj_batch = [np.matmul(verts_obj_base, pose_fp[:3, :3].T) + pose_fp[:3, 3] for pose_fp in
                               B_in_cams_init]
            verts_comb_in = torch.from_numpy(np.concatenate([verts_in, np.stack(verts_obj_batch)], 1)).float().to(
                device)  # (T, N_total, 3)
            _, rend_in, rend_in_side, _ = self.render_front_side(H, K, W, glctx, mesh_tensors, verts_comb_in)

            files = frames_used
            data_pr['pose_abs'].append(B_in_cams.reshape(-1, 4, 4))
            data_pr['frames'].extend(files)
            data_gt['pose_abs'].append(poseB.reshape(-1, 4, 4))
            data_gt['frames'].extend(files)
            data_in['pose_abs'].append(poseA.reshape(-1, 4, 4))
            data_in['frames'].extend(files)

            data_pr['smpl_pose'].append(pred_smpl_pose)  # (BT, 156)
            data_pr['smpl_t'].append(pred_smpl_t)
            data_pr['betas'].append(batch['betas_gt'].reshape(-1, 10))
            data_gt['smpl_pose'].append(batch['smpl_poses_gt'].reshape(-1, 156))
            data_gt['smpl_t'].append(batch['smpl_transl_gt'].reshape(-1, 3))
            data_gt['betas'].append(batch['betas_gt'].reshape(-1, 10))
            data_pr['verts'].append(verts_pr)
            data_in['verts'].append(torch.from_numpy(verts_in).to(device).float())

            # add contact logits
            if 'contact' in output:
                data_pr['contact_logits'].append(output['contact'].reshape(-1, 2))
                data_gt['contact_logits'].append(output['contact'].reshape(-1, 2))
                data_in['contact_logits'].append(output['contact'].reshape(-1, 2))
                print("contact logits added")
            else:
                # add dummy data 
                data_pr['contact_logits'].append(torch.zeros(len(files), 2))
                data_gt['contact_logits'].append(torch.zeros(len(files), 2))
                data_in['contact_logits'].append(torch.zeros(len(files), 2))

            frames_all.extend(files) # to accumulate for all

            # Input data
            data_in['smpl_pose'].append(torch.from_numpy(poses_nlf_init).to(device).float().reshape(-1, 156))
            data_in['smpl_t'].append(torch.from_numpy(trans_nlf_init).to(device).float().reshape(-1, 3))
            data_in['betas'].append(torch.from_numpy(betas_avg).to(device).float().reshape(-1, 10))
            # Prepare GT SMPL and object for visualization
            if not cfg.wild_video:
                smpl_t_world = packed['trans'][start:end].astype(np.float32)
                vs_gt_world = body_model(torch.from_numpy(poses_full).to(device),
                                         batch['betas_gt'].reshape(-1, 10).to(device),
                                         torch.from_numpy(smpl_t_world).to(device))[0]
                data_gt['verts'].append(vs_gt_world)

                # GT object verts in camera
                angles_gt = packed['obj_angles'][start:end].astype(np.float32)
                transl_gt = packed['obj_trans'][start:end].astype(np.float32)
                R_obj = torch.from_numpy(R.from_rotvec(angles_gt).as_matrix()).to(device).float()  # (T, 3, 3)
                t_world = torch.from_numpy(transl_gt).to(device).float()  # (T, 3)
                obj_verts_gt = torch.matmul(obj_base_centered[None].expand(end - start, -1, -1),
                                            R_obj.permute(0, 2, 1)) + t_world[:, None]
                verts_comb_gt = torch.cat([vs_gt_world, obj_verts_gt], dim=1)
                _, rend_gt, rend_gt_side, _ = self.render_front_side(H, K, W, glctx, mesh_tensors, verts_comb_gt)

            # visualize input
            if cfg.viz_input:
                maskA, maskB, rgbsA, rgbsB, xyzA, xyzB = trainer.prepare_input_viz(batch, cfg)

            try:
                for j in tqdm(range(end - start)):
                    frame_time = fp_frames[start + j]
                    # reuse preloaded color
                    color = cv2.resize(full_colors[j], (W, H))
                    in_comb = self.comb_front_side(color, rend_in[j], rend_in_side[j])
                    pr_comb = self.comb_front_side(color, rend_pr[j], rend_pr_side[j])
                    rgb_comb = self.comb_front_side(color, color, color)
                    combs = [rgb_comb, in_comb, pr_comb]
                    if not cfg.wild_video:
                        gt_comb = self.comb_front_side(color, rend_gt[j], rend_gt_side[j])
                        combs.append(gt_comb)
                    comb = np.concatenate(combs, axis=1)
                    cv2.putText(comb, frame_time+ f' idx {j+start}', (comb.shape[1] // 4, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                (0, 255, 255), 2)

                    if cfg.viz_input:
                        bid = 0
                        comb_in, rgba, rgbb = trainer.visualize_rgbm(batch, bid, j, maskA, maskB, rgbsA, rgbsB)
                        xyza_vis = (np.clip(xyzA[bid, j].transpose(1, 2, 0) + 0.5, 0, 1.) * 255).astype(np.uint8)
                        xyzb_vis = (np.clip(xyzB[bid, j].transpose(1, 2, 0) + 0.5, 0, 1.) * 255).astype(np.uint8)
                        comb_in = np.concatenate([comb_in, np.concatenate([xyza_vis, xyzb_vis], 0)], axis=1)
                        hc, (hi, wi) = comb.shape[0], comb_in.shape[:2]
                        comb_in = cv2.resize(comb_in, (int(wi*hc/hi), hc))
                        comb = np.concatenate([comb, comb_in], axis=1)

                    vw.append_data(comb)
            finally:
                pass
        vw.close()
        print(f'visualization saved to {out_path}')
        # save result as one pth file 
        pth_file = f'{cfg.outpath}/{save_name}/{seq_name}.pth'
        os.makedirs(osp.dirname(pth_file), exist_ok=True)
        data_pr = {k: torch.cat(v, 0) if k != 'frames' and len(v) > 0 else v for k, v in data_pr.items()}
        data_gt = {k: torch.cat(v, 0) if k != 'frames' and len(v) > 0 else v for k, v in data_gt.items()}
        data_in = {k: torch.cat(v, 0) if k != 'frames' and len(v) > 0 else v for k, v in data_in.items()}
        torch.save({"gt": data_gt, "pr": data_pr, "in": data_in}, pth_file)
        print(f'result saved to {pth_file}')

    def comb_front_side(self, color, rp, rp_side):
        "rp: front view, rp_side: side view"
        mask_p = (rp.sum(axis=-1, keepdims=True) > 0)
        alpha = 0.7
        pred_top = color.copy()
        pred_top[mask_p[..., 0]] = (alpha * rp[mask_p[..., 0]] + (1 - alpha) * pred_top[mask_p[..., 0]]).astype(np.uint8)
        # cut
        if not self.cfg.wild_video:
            h, w = color.shape[:2]
            x1, x2 = int(w*0.15), int(w*0.85)
            y1, y2 = int(h*0.15), int(h*1)
            pr_comb = np.concatenate((pred_top[y1:y2, x1:x2], rp_side[y1:y2, x1:x2]), axis=0)
        else:
            # no cut
            pr_comb = np.concatenate((pred_top, rp_side), axis=0)
        return pr_comb

    def render_front_side(self, H, K, W, glctx, mesh_tensors, verts_comb_pr):
        device = verts_comb_pr.device

        projection_mat = torch.as_tensor(
            Utils.projection_matrix_from_intrinsics(K, height=H, width=W, znear=0.001, zfar=100).reshape(1, 4, 4),
            device=device, dtype=torch.float)
        ob_in_glcams = torch.tensor(Utils.glcam_in_cvcam, device=device, dtype=torch.float).reshape(1, 4, 4)
        mtx_front = (projection_mat @ ob_in_glcams).repeat(len(verts_comb_pr), 1, 1)  # (T, 4, 4)
        pos_homo = Utils.to_homo_torch(verts_comb_pr)
        pos_clip = (mtx_front[:, None] @ pos_homo[..., None])[..., 0]
        rend_batch = Utils.nvdiff_rasterize(glctx, mesh_tensors, pos_clip, (H, W))  # (T, H, W, 3)
        rend_batch = (rend_batch * 255).byte().cpu().numpy()
        # Side view transformation
        if self.side_view_z is None:
            self.side_view_z = torch.mean(verts_comb_pr[:, :, 2])
        z_now = torch.mean(verts_comb_pr[:, :, 2])
        if abs(z_now - self.side_view_z) > 1.0:
            self.side_view_z = z_now
        z = self.side_view_z
        
        at = torch.tensor([[0.0, 0.0, z]], device=device, dtype=torch.float)
        Rv, Tv = look_at_view_transform(dist=z*1.3, elev=0, azim=75, at=at, up=((0, 1, 0),), device=device)
        view_mat = torch.eye(4, device=device, dtype=torch.float)
        view_mat[:3, :3] = Rv[0]
        view_mat[:3, 3] = Tv[0]
        verts_pr_side = torch.matmul(verts_comb_pr, view_mat[:3, :3]) + view_mat[:3, 3]
        pos_homo_side = Utils.to_homo_torch(verts_pr_side)
        pos_clip_side = (mtx_front[:, None] @ pos_homo_side[..., None])[..., 0]
        rend_batch_side = Utils.nvdiff_rasterize(glctx, mesh_tensors, pos_clip_side, (H, W))
        rend_batch_side = (rend_batch_side * 255).byte().cpu().numpy()
        return mtx_front, rend_batch, rend_batch_side, view_mat


def main():
    from argparse import Namespace
    from glob import glob

    from learning.training.trainer import get_config
    cfg = get_config()

    # check if the video is a path or a pattern to files 
    if osp.isfile(cfg.video):
        videos = [cfg.video]
    else:
        videos = sorted(glob(cfg.video))
    print(f'In total {len(videos)} videos')
    for video in videos:
        cfg.video = video

        args = Namespace(
            # from fp_behave.py
            video=cfg.video,
            outpath='/home/xianghuix/datasets/behave/fp',
            fps=30,
            tstart=3.0,
            tend=None,
            redo=False,
            kid=1,
            start=0,
            end=None,  # override (-1) with your snippet's default
            nodepth=False,

            # from your provided defaults
            packed_path='/home/xianghuix/datasets/behave/behave-packed/',
            dataset_path='/home/xianghuix/datasets/behave/',
            output_dir='/home/xianghuix/data/foundpose_train/behave',
            h5_path='/home/xianghuix/data/behave_release/30fps-h5',
            shard_num=5,
            trans_normalizer=[0.02, 0.02, 0.05],
            rot_normalizer=20.0,
            rend_size=224,
            skip=1,
            add_rgb=False,
            data_source='behave',
        )
        args.wild_video = cfg.wild_video
        args.cam_id = cfg.cam_id

        runner = HORefineRunner(args)
        runner.run(args, cfg)


if __name__ == '__main__':
    main()

