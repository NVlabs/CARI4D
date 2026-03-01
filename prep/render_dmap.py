# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import sys, os
sys.path.append(os.getcwd())
import cv2
import numpy as np
import glob, json
from tqdm import tqdm
import joblib
import trimesh
import os.path as osp
from scipy.spatial.transform import Rotation as R
import imageio
import Utils
from videoio import VideoReader
from videoio import Uint16Writer
import torch
from behave_data.behave_video import BaseBehaveVideoData
from prep.fp_filter_2dir import FPFilterTwoDirProcessor
from lib_smpl import get_smpl
import nvdiffrast.torch as dr
import time

class DMapRenderer(FPFilterTwoDirProcessor):
    def render(self, args):
        "use nvdiff to render depth map"
        compute_dists = True
        dist_key = 'dists_h2o'
        
        # load packed GT data 
        packed_file = f'/home/xianghuix/datasets/behave/behave-packed/{self.video_prefix}_GT-packed.pkl'
        if osp.isfile(packed_file):
            packed_data = joblib.load(packed_file)
        else:
            print(f'{packed_file} does not exist, skipping')
            return
        if dist_key in packed_data and compute_dists:
            print(f'{packed_file} already has HO_dists, skipping')
            return
        # check the length of packed frames, should be the same as the video length 
        video_length = len(self.times)
        if len(packed_data['frames']) != video_length:
            print(f'{packed_file} frames length is not the same as the video length, {len(packed_data["frames"])} != {video_length}')
            # return
        # get GT SMPL verts
        gender = packed_data['gender']
        smpl_model = get_smpl(gender, hands=True).to('cuda')
        gt_smpl_verts, gt_smpl_joints = smpl_model(torch.from_numpy(packed_data['poses']).to('cuda').float(),
                                   torch.from_numpy(packed_data['betas']).to('cuda').float(),
                                   torch.from_numpy(packed_data['trans']).to('cuda').float())[:2]
        gt_smpl_verts = gt_smpl_verts.cpu().numpy() 

        # get obj pose 
        obj_angles = packed_data['obj_angles']
        obj_trans = packed_data['obj_trans']
        obj_pose = np.eye(4)[None].repeat(len(obj_angles), 0)
        obj_pose[:, :3, :3] = R.from_rotvec(obj_angles).as_matrix()
        obj_pose[:, :3, 3] = obj_trans

        device = 'cuda'
        torch.set_default_tensor_type('torch.FloatTensor')  # fix bug in pytorch3d loading
        obj_mesh, template_file = self.load_template_mesh(ret_file=True)
        center = np.mean(trimesh.load(template_file, process=False).vertices, 0)
        obj_samples = obj_mesh.sample(6000)
        if not compute_dists:
            mesh_tensors, meshes = Utils.load_smpl_obj_uvmap(self.video_prefix, use_hy3d=False)
            obj_idx = 1
            tex = meshes[obj_idx].textures
            uv = tex.verts_uvs_padded()[0]
            uv[:, 1] = 1 - uv[:, 1]
            mesh_tensors_obj = {
                "tex": tex.maps_padded().to(device).float(),  # this must be (1, H, W, 3)
                'uv_idx': torch.tensor(tex.faces_uvs_padded()[0], device=device, dtype=torch.int),
                # correct way to get face uvs
                'uv': uv.to(device).float(),
                'pos': meshes[obj_idx].verts_padded()[0].to(device).float() - torch.tensor(center, device=device).float(),
                'faces': torch.tensor(meshes[obj_idx].faces_padded()[0], device=device, dtype=torch.int),
                'vnormals': meshes[obj_idx].verts_normals_padded()[0].to(device).float(),
            }
            # init render
            self.glctx = dr.RasterizeCudaContext()

        video_path = osp.dirname(args.video)
        video_length = VideoReader(args.video).video_params['length']
        depth_files = {k:osp.join(video_path, f'{self.video_prefix}.{k}.depth-reg.mp4') for k in self.kids}

        assert args.data_source in ['imhd', 'hodome', 'behave', 'intercap']
        if args.data_source == 'imhd':
            render_size = (1080, 1920)
        elif args.data_source == 'hodome':
            render_size = (720, 1280)
        elif args.data_source == 'behave':
            render_size = (1536, 2048)
            assert compute_dists, 'can only compute dists for BEHAVE dataset'
        elif args.data_source == 'intercap':
            render_size = (1080, 1920)
            assert compute_dists, 'can only compute dists for InterCap dataset'
        else:
            raise ValueError(f'Invalid data source: {args.data_source}')

        # get w2c transform
        
        w2cs = packed_data['extrinsics']
        dists_all = []
        K_all = packed_data['intrinsics']
        kids = range(len(w2cs)) if not compute_dists else [0] # if compute dists, only compute one view is enough 
        for kid in kids:
            outfile_k = depth_files[kid].replace('.depth-reg.mp4', '_done.txt')
            if osp.isfile(outfile_k) and not compute_dists:
                print(f'{outfile_k} already exists, skipping')
                continue
            depth_writer = None
            if args.viz_path is not None:
                viz_path = f'outputs/viz-render-dmap/{self.video_prefix}_k{kid}.mp4'
                vw = imageio.get_writer(viz_path, 'FFMPEG', fps=6)
            
            w2c = w2cs[kid]
            smpl_verts_k = np.matmul(gt_smpl_verts, w2c[:3, :3].T) + w2c[:3, 3]
            obj_pose_k = np.stack([np.matmul(w2c, p) for p in obj_pose], 0)
            obj_pose_k_th = torch.from_numpy(obj_pose_k).to(device).float()
            obj_samples_k = torch.matmul(torch.from_numpy(obj_samples[None]).to(device).float(), obj_pose_k_th[:, :3, :3].permute(0, 2, 1)) + obj_pose_k_th[:, :3, 3][:, None, :]

            # use pytorch3d knn to compute the nearest distance from human to object samples 
            if compute_dists:
                from pytorch3d.ops import knn_points
                # compute distance from human joints to object samples
                smpl_joints_k = np.matmul(gt_smpl_joints.cpu().numpy(), w2c[:3, :3].T) + w2c[:3, 3]
                knn = knn_points(torch.from_numpy(smpl_joints_k).to(device).float(), obj_samples_k, K=1)
                # get smallest distance from human to object samples 
                dists_h2o = knn.dists.squeeze(-1).cpu().numpy()
                dists_all.append(dists_h2o) # (L, 52)
                continue

            else:
                dists_min = np.zeros(len(smpl_verts_k)) - 1

            chunk_size = 640 if args.data_source == 'hodome' else 196
            for i in tqdm(range(0, video_length, chunk_size)):
                smpl_verts_k_chunk = torch.from_numpy(smpl_verts_k[i:i+chunk_size]).to(device).float()
                obj_pose_k_chunk = torch.from_numpy(obj_pose_k[i:i+chunk_size]).to(device).float()
                obj_verts_k = mesh_tensors_obj['pos'][None].repeat(len(smpl_verts_k_chunk), 1, 1)
                obj_verts_k = torch.matmul(obj_verts_k, obj_pose_k_chunk[:, :3, :3].permute(0, 2, 1)) + obj_pose_k_chunk[:, :3, 3][:, None, :]
                verts_comb = torch.cat([smpl_verts_k_chunk, obj_verts_k], 1)
                K_rois_chunk = [K_all[kid]]*len(smpl_verts_k_chunk)
                depth_only = i > 50

                t0 = time.time()
                ret = Utils.nvdiff_color_depth_render(K_rois_chunk, self.glctx, mesh_tensors, render_size, verts_comb, depth_only=depth_only)
                t1 = time.time()
                print(f'render time: {t1 - t0:.4f} seconds, {len(smpl_verts_k_chunk)} frames')
                if depth_only:
                    depth, xyz_map = ret
                else:
                    color, depth, xyz_map = ret

                # write the depth to file 
                if depth_writer is None:
                    H, W = depth[0].shape[:2]
                    depth_writer = Uint16Writer(depth_files[kid], (W, H), fps=30)
                
                depth = (depth*1000).cpu().numpy().astype(np.uint16)
            
                for d in depth:
                    depth_writer.write(d)
                
                # now visualize
                if args.viz_path is not None and i < 50:
                    # load the RGB 
                    for j in range(i, i+len(smpl_verts_k_chunk)):
                        if j % 5 != 0:
                            continue # only viz 50 frames 
                        frame_time = self.get_time_str(j)
                        # now load RGB 
                        color_orig, _ = self.load_color_depth(kid, self.kids, j)
                        rend_j = (color[j-i][:, :, :3].cpu().numpy() * 255).astype(np.uint8)
                        # visualize depth and color
                        xyz_j = xyz_map[j-i].cpu().numpy()
                        median = np.median(xyz_j[:, :, 2])
                        xyz_j[:, :, 2] = xyz_j[:, :, 2] - median
                        xyz_j_vis = (np.clip(xyz_j+0.5, 0, 1.)* 255).astype(np.uint8)
                        # visualize the depth 
                        dmax = np.max(depth[j-i])
                        dvis = (depth[j-i].astype(np.float32)/dmax*255).astype(np.uint8)[:, :, None].repeat(3, -1)
                        comb = np.concatenate([color_orig, rend_j, xyz_j_vis, dvis, cv2.addWeighted(color_orig, 0.5, rend_j, 0.5, 0)], axis=1)
                        comb = cv2.resize(comb, (comb.shape[1]//4, comb.shape[0]//4))
                        # add text and kid to the image
                        cv2.putText(comb, f'{frame_time} k{kid} dist {dists_min[j]:.3f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        vw.append_data(comb)


            # close the writer and print video path     
            if depth_writer is not None:
                depth_writer.close()
            if args.viz_path is not None:
                vw.close()
                print(f'video saved to {viz_path}, done')
            # indicator that this video is done
            np.savetxt(outfile_k, [1], fmt='%d')
            print(f'{outfile_k} created')
    
        # save the dists to packed file
        if compute_dists:
            packed_data[dist_key] = np.stack(dists_all, 0)
            joblib.dump(packed_data, packed_file)
            print(f'{packed_file} saved')
            

if __name__ == '__main__':
    args = DMapRenderer.get_parser().parse_args()
    args.nodepth = True

    if osp.isfile(args.video):
        videos = [args.video]
    else:
        videos = sorted(glob.glob(args.video)) # process multiple video
    videos_filtered = []
    imhd_seqs = json.load(open('splits/imhd-selected.json', 'r'))['seqs'] # 76 seqs in total.
    for video in videos:
        if '2022' in video:
            if 'monitor' in video:
                continue
            videos_filtered.append(video)
        elif 'ICap' in video:
            if 'sub09' not in video and 'sub10' not in video:
                continue
            videos_filtered.append(video)
        else:
            # load imhd selected videos 
            seq = osp.basename(video).split('.')[0]
            if seq in imhd_seqs:
                videos_filtered.append(video)
    videos = videos_filtered
    if args.index is not None:
        chunk_size = len(videos) // 4 + 1 
        videos = videos[args.index * chunk_size:(args.index + 1) * chunk_size]
    print(f"In total {len(videos)} video files, processing first video: {videos[0]}, last video: {videos[-1]}")
    for video in tqdm(videos):
        args.video = video
        try:
            renderer = DMapRenderer(args)
            renderer.render(args)
        except Exception as e:
            import traceback
            traceback.print_exc()
            continue










