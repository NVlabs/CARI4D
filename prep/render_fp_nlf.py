# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import glob
import pickle
import sys, os
import time

import imageio
import joblib
import os.path as osp

sys.path.append(os.getcwd())
import cv2
import numpy as np
from tqdm import tqdm
import h5py
import nvdiffrast.torch as dr
sys.path.append(os.getcwd())
import torch, trimesh, json
import Utils
from lib_smpl import get_smpl
from tools import img_utils
from behave_data.utils import get_intrinsics_unified
from behave_data.const import get_test_view_id
from behave_data.utils import init_video_controllers, load_kinect_poses_back
from behave_data.behave_video import load_masks

from prep.prerender_behave import BehaveRenderer

class BehaveFPNLFRenderer(BehaveRenderer):
    def render_seq(self, args):
        ""
        video_prefix = osp.basename(args.video).split('.')[0]
        seq_name = video_prefix
        _, _, center = self.load_obj_template(args, seq_name, ret_center=True)  # temp_orig is centralized
        gt_to_perturb_pose = np.eye(4)
        kids = [x for x in range(4)]
        kids = [x for x in range(6)] if 'ICap' in seq_name else kids

        # Load FP results
        fp_root = args.fp_root
        fp_data = joblib.load(osp.join(fp_root, f'{seq_name}_all.pkl'))
        fp_poses = fp_data['fp_poses']
        fp_frames = fp_data['frames']  # first delete the data, and then update
        parts_vis, penetration = fp_data['visibilities'] if 'visibilities' in fp_data else np.zeros(
            (len(fp_poses), len(kids), 25)), fp_data['penetration'] if 'penetration' in fp_data else np.zeros(
            (len(fp_poses), len(kids)))

        trans_normalizer = np.array(args.trans_normalizer)
        rot_normalizer = args.rot_normalizer
        print(f"Normalization parameters: {trans_normalizer}, rot_normalizer: {rot_normalizer}")

        if args.data_source in ['behave', 'procigen']:
            w2c_rots, w2c_trans = load_kinect_poses_back(
                osp.join(args.dataset_path, 'calibs', video_prefix.split('_')[0], 'config'), kids)
        elif args.data_source in ['intercap', 'hodome', 'imhd']:
            gt_packed = joblib.load(f'/home/xianghuix/datasets/behave/behave-packed/{video_prefix}_GT-packed.pkl')
            w2c_rots, w2c_trans = gt_packed['extrinsics'][:, :3, :3], gt_packed['extrinsics'][:, :3, 3]

        K_all = [get_intrinsics_unified(args.data_source, video_prefix, kid, args.wild_video) for kid in kids]
        K_all = np.array(K_all)

        end = len(fp_frames) if args.end is None else args.end
        mesh_tensors, mesh_tensors_obj, meshes, obj_idx = self.load_mesh_tensors(args)
        verts_obj_temp = meshes[obj_idx].verts_packed().cpu().numpy() - center  # use the object template, so it aligns with the pose no matter it is HY3D or GT mesh template
        
        outfile = osp.join(args.output_dir, f'{seq_name}_render.h5')
        os.makedirs(osp.dirname(outfile), exist_ok=True)
        if not osp.isfile(outfile):
            h5_writer = h5py.File(outfile, 'w')
            old_keys = []
        else:
            try:
                h5_writer = h5py.File(outfile, 'r+')  # both read and write
                old_keys = set(h5_writer.keys())
            except Exception as e:
                print(f'Error opening {outfile}: {e}')
                h5_writer = h5py.File(outfile, 'w')
                old_keys = []

        glctx = dr.RasterizeCudaContext()
        render_size = (args.rend_size, args.rend_size)

        # load RGB image and masks as well
        controllers, tar_mask = self.prepare_video_mask_loader(args, kids, video_prefix)

        # load GT pose and compute SMPL verts
        mesh_diameter, packed_data, verts_all = self.prepare_smpl_verts(video_prefix)
        key = seq_name + '_w2c'
        if key not in old_keys:
            h5_writer.create_dataset(key, data=np.void(pickle.dumps({"rot": w2c_rots, "trans": w2c_trans,
                                                                     'mesh_diameter': mesh_diameter,
                                                                     'trans_normalizer': trans_normalizer,
                                                                     'rot_normalizer': rot_normalizer}, 0)))
            print(f'{key} created successfully')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        vw = imageio.get_writer(outfile.replace('.h5', '.mp4'), 'FFMPEG', fps=4) if args.debug else None

        chunk_size = 30

        # for hy3d and test set, only render specific view, for this we create dummy data for other views
        if 'hy3d3' in args.fp_root:
            fp_poses = fp_poses.repeat(len(kids), 1) if fp_poses.shape[
                                                            1] == 1 else fp_poses  # (N, 1, 4, 4) -> (N, 4, 4, 4)
            # get the selected view
            selected_views = json.load(open('splits/selected-views-map.json'))
            video_prefix = osp.basename(args.video).split('.')[0]
            if video_prefix not in selected_views and args.data_source == 'behave':
                raise ValueError(
                    f'{video_prefix} not in selected_views, please double check!')  # for other data sources, allow to render all views
            args.camera_ids = [int(selected_views[video_prefix][1])] if args.data_source == 'behave' else kids  # test
            # no need to repeat NLF as all views are done
            print("rendering only one view: ", args.camera_ids)
            print('fp_poses shape: ', fp_poses.shape, 'verts_all shape: ', verts_all.shape)

        render_views = kids if not args.use_sel_view else [get_test_view_id(video_prefix)]
        print('render_views: ', render_views)
        viz_count = 0
        for i in tqdm(range(args.start, end, args.skip)):
            frame_time = fp_frames[i]
            frame_key = f'{seq_name}+{frame_time}'
            # render 4 views all at once
            poses = [np.matmul(fp_poses[i, k], gt_to_perturb_pose) for k in kids]
            # add RGB
            t = float(frame_time[1:]) if args.data_source == 'behave' else float(frame_time)
            actual_times = np.array([controllers[x].get_closest_time(t) for x in kids])
            best_kid = np.argmin(np.abs(actual_times - t))
            actual_time = actual_times[best_kid]

            if frame_time not in packed_data['frames']:
                print(f'{frame_time} does not appear in packed data')
                continue
            idx_gt = packed_data['frames'].index(frame_time)
            verts_obj = np.array(verts_obj_temp).copy()

            for k in render_views:
                # format view specific intrinsics
                focal = np.array([K_all[k][0, 0], K_all[k][1, 1]])
                principal_point = np.array([K_all[k][0, 2], K_all[k][1, 2]])
                if k not in args.camera_ids:
                    continue
                render_key = frame_key + f'_k{k}_perturb_0'
                input_key = frame_key + f'_k{k}_input'
                if render_key in old_keys and input_key in old_keys:
                    continue
                mask_h, mask_o = load_masks(video_prefix, frame_time, k, tar_mask[k])
                if mask_h is None:
                    continue

                bmin, bmax = img_utils.masks2bbox([mask_h, mask_o])
                center = (bmax + bmin) / 2
                radius = np.max(bmax - bmin) * 1.1 / 2
                top_left = center - radius
                bottom_right = center + radius
                K_roi = self.Kroi_from_corners(bottom_right, top_left, focal=focal, principal_point=principal_point)
                Krois = [K_roi]
                bboxes = [np.concatenate([top_left, bottom_right], 0)]

                # now render
                if render_key not in old_keys:
                    vo = np.matmul(verts_obj, poses[k][:3, :3].T) + poses[k][:3, 3]
                    if len(verts_all.shape) == 3:
                        # GT SMPL
                        vh = np.matmul(verts_all[idx_gt].copy(), w2c_rots[k].T) + w2c_trans[k]
                    elif len(verts_all.shape) == 4:
                        vh = verts_all[k, idx_gt].copy()  # (N, 3)
                    else:
                        raise ValueError(f'Unknown verts shape {verts_all.shape}!')
                    mesh_tensors['pos'] = torch.from_numpy(np.concatenate([vh, vo], 0)).float().cuda()
                    mesh_tensors_obj['pos'] = torch.from_numpy(vo).float().cuda()
                    extra = {}
                    bbox2d_ori = torch.tensor([[0, 0., render_size[0], render_size[1]]]).repeat(1, 1)

                    rgb_r, depth_r, normal_r = Utils.nvdiffrast_render(K=np.stack(Krois, 0), H=render_size[1],
                                                                       W=render_size[0],
                                                                       ob_in_cams=torch.as_tensor(
                                                                           np.eye(4)[None]).float(), context='cuda',
                                                                       get_normal=False, glctx=glctx,
                                                                       mesh_tensors=mesh_tensors,
                                                                       output_size=render_size, bbox2d=bbox2d_ori,
                                                                       use_light=True,
                                                                       extra=extra)  # (B, H, W, C) # this is only one percent utilization if render one image
                    rgb_obj, depth_obj, normal_obj = Utils.nvdiffrast_render(K=np.stack(Krois, 0), H=render_size[1],
                                                                             W=render_size[0],
                                                                             ob_in_cams=torch.as_tensor(
                                                                                 np.eye(4)[None]).float(),
                                                                             context='cuda',
                                                                             get_normal=False, glctx=glctx,
                                                                             mesh_tensors=mesh_tensors_obj,
                                                                             output_size=render_size, bbox2d=bbox2d_ori,
                                                                             use_light=True, extra=extra)

                    t3 = time.time()
                    rgbs = (rgb_r.cpu().numpy() * 255).astype(np.uint8)
                    dmaps = depth_r.cpu().numpy()

                    dmap_full = depth_r[0].cpu().numpy()
                    dmap_obj = depth_obj[0].cpu().numpy()
                    mask_rend_o = (dmap_obj <= dmap_full) & (dmap_obj > 0)
                    mask_o_full = dmap_obj > 0

                    data_key = frame_key + f'_k{k}_perturb_0'
                    h5_writer.create_dataset(data_key,
                                             data=np.void(pickle.dumps({
                                                 'rgba': rgbs[0],
                                                 'depth': dmaps[0].astype(np.float16),
                                                 'pose': poses[k],
                                                 'K_roi': Krois[0],
                                                 'bbox': bboxes[0],
                                                 'mask_o': np.stack([mask_rend_o, mask_o_full], -1),
                                             }, 0)))

                # rgb and depth
                data_key = frame_key + f'_k{k}_input'
                if data_key in old_keys:
                    continue

                color, depth = controllers[k].get_closest_frame(actual_time)
                color = np.concatenate([color, mask_h[:, :, None], mask_o[:, :, None]], -1)
                dmap_xyz, rgbm = self.crop_color_dmap(bboxes[0], color, depth, render_size, K_full=K_all[k])
                h5_writer.create_dataset(data_key,
                                         data=np.void(pickle.dumps({
                                             'rgbmB': rgbm.cpu().numpy().astype(np.uint8),
                                             'xyzB': dmap_xyz.cpu().numpy().astype(np.float16)
                                         }, 0)))

                # visualize
                if vw is not None and render_key not in old_keys and viz_count < 20:  # only save 20 example viz
                    viz_count += 1
                    xyzA = Utils.depth2xyzmap(dmaps[0], Krois[0]) * 2 / mesh_diameter
                    xyzB = dmap_xyz.cpu().numpy() * 2 / mesh_diameter
                    mask = (rgbm[:, :, 3] > 0) | (rgbm[:, :, 4] > 0)
                    z_median = np.median(xyzA[:, :, 2][mask])
                    xyzA[:, :, 2][mask] = xyzA[:, :, 2][mask] - z_median
                    xyzB[:, :, 2][mask] = xyzB[:, :, 2][mask] - z_median
                    xyza_vis = (np.clip(xyzA + 0.5, 0, 1.) * 255).astype(np.uint8)
                    xyzb_vis = (np.clip(xyzB + 0.5, 0, 1.) * 255).astype(np.uint8)
                    err = np.abs(xyzb_vis - xyza_vis)
                    xyz_diff = (np.clip(err / 5.0, 0, 1.) * 255).astype(np.uint8)

                    mask_viz = np.concatenate([rgbm[:, :, 3:5], np.zeros_like(rgbm[:, :, 3:4])], -1).astype(np.uint8)
                    viz1 = np.concatenate([rgbs[0], (rgb_obj.cpu().numpy() * 255).astype(np.uint8)[0],
                                           mask_rend_o[:, :, None].astype(np.uint8).repeat(3, -1) * 255, mask_viz], 1)
                    viz2 = np.concatenate(
                        [cv2.addWeighted(rgbm[:, :, :3].numpy().astype(np.uint8), 0.5, rgbs[0], 0.5, 0), xyza_vis,
                         xyzb_vis, xyz_diff], 1)
                    viz = np.concatenate([viz1, viz2], 0)
                    cv2.putText(viz, frame_key + f' k{k} vis {parts_vis[i, k][-1]:.3f} pen {penetration[i, k]:.3f}',
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    vw.append_data(viz.astype(np.uint8))

        print('all done')
        if vw is not None:
            vw.close()
            print('video saved to', outfile.replace('.h5', '.mp4'))

            
    def load_mesh_tensors(self, args):
        video_prefix = osp.basename(args.video).split('.')[0]
        torch.set_default_tensor_type('torch.FloatTensor')  # fix bug in pytorch3d loading
        obj_name = video_prefix.split('_')[2]
        mesh_tensors, meshes = Utils.load_smpl_obj_uvmap(video_prefix, use_hy3d='hy3d' in args.fp_root)
        device = 'cuda'
        obj_idx = 1
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
        return mesh_tensors, mesh_tensors_obj, meshes, obj_idx

    def prepare_smpl_verts(self, video_prefix, device='cuda'):
        "load poses from NLF and use avg betas to compute"
        nlf_data = joblib.load(f'{self.args.nlf_path}/{video_prefix}_params.pkl')
        print("NLF data loaded from", f'{self.args.nlf_path}/{video_prefix}_params.pkl')
        thetas, betas, smpl_trans = nlf_data['poses'], nlf_data['betas'], nlf_data['transls']
        gender = 'neutral' if 'gender' not in nlf_data else nlf_data['gender']
        smpl_model = get_smpl(gender, hands=True).to(device) # use gender specific model
        assert 'smplh' in self.args.nlf_path, 'invalid NLF path: {} for SMPLH model!'.format(self.args.nlf_path)
        kids = range(betas.shape[1])  # (T, K, 10)
        betas = nlf_data['betas']
        verts_all = []
        for kid in kids:
            betas_avg = np.mean(betas[:, kid], axis=0)[None].repeat(len(thetas), axis=0)
            verts = smpl_model(torch.from_numpy(thetas[:, kid]).to(device),
                                   torch.from_numpy(betas_avg).to(device),
                                   torch.from_numpy(smpl_trans[:, kid]).to(device)
                                   )[0].cpu().numpy()
            verts_all.append(verts)
        # compute mesh_diameter using T pose SMPL
        betas_avg = np.mean(betas, axis=(0, 1))
        mesh_diameter = self.get_smpl_diameter(betas_avg, smpl_model, smpl_trans)
        verts_all = np.stack(verts_all, axis=0) # (K, T, N, 3)
        print("T pose SMPL mesh diameter based on NLF:", mesh_diameter, verts_all.shape)
        return mesh_diameter, nlf_data, verts_all

    def get_smpl_diameter(self, betas_avg, smpl_model, smpl_trans=None):
        verts_tpose = smpl_model(torch.zeros(1, 156).cuda(),
                                 torch.from_numpy(betas_avg[None]).cuda(),
                                 torch.from_numpy(np.zeros((1, 3))).cuda()
                                 )[0].cpu().numpy()
        np.random.seed(0)
        samples = trimesh.Trimesh(verts_tpose[0], smpl_model.faces).sample(8000)  # where is the uv_ids?
        mesh_diameter = Utils.compute_mesh_diameter(model_pts=samples,
                                                    n_sample=8000)  # this is much smaller than using vertices, and robust to num of samples
        return mesh_diameter



def _child_run(args):
    # run one video 
    renderer = BehaveFPNLFRenderer(args)
    renderer.render_seq(args)


if __name__ == '__main__':

    parser = BehaveFPNLFRenderer.get_parser()

    args = parser.parse_args()

    if osp.isfile(args.video):
        videos = [args.video]
    else:
        videos = sorted(glob.glob(args.video)) # process multiple video
    print(f"In total {len(videos)} video files")
    # cut videos into chunks 
    chunk_size = len(videos) // 6 + 1
    if args.index is not None:
        videos = videos[args.index * chunk_size:(args.index + 1) * chunk_size]
    print(f'processing {len(videos)} videos, first video: {videos[0]}, last video: {videos[-1]}')
    
    for video in tqdm(videos):
        args.video = video
        renderer = BehaveFPNLFRenderer(args)
        seq_name = osp.basename(args.video).split('.')[0]
        fp_file = osp.join(args.fp_root, f'{seq_name}_all.pkl')
        if not osp.isfile(fp_file):
            print(f'{fp_file} not found, skipping...')
            continue
        video_prefix = osp.basename(args.video).split('.')[0]
        args.seq = video_prefix
        print('processing', video_prefix)
        try:
            renderer.render_seq(args)
        except Exception as e:
            import traceback
            traceback.print_exc()