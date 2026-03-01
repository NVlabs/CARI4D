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
        compute_dists = False
        dist_key = 'dists_h2o'
        
        packed_file = f'{args.nlf_path}/{self.video_prefix}_params.pkl'

        packed_data = joblib.load(packed_file)
        # check the length of packed frames, should be the same as the video length 
        video_length = len(self.times)
        if len(packed_data['frames']) != video_length:
            print(f'{packed_file} frames length is not the same as the video length, {len(packed_data["frames"])} != {video_length}')
        # get GT SMPL verts
        gender = packed_data['gender']
        smpl_model = get_smpl(gender, hands=True).to('cuda')
        kid = 0 
        gt_smpl_verts, gt_smpl_joints = smpl_model(torch.from_numpy(packed_data['poses'][:, kid]).to('cuda').float(),
                                   torch.from_numpy(packed_data['betas'][:, kid]).to('cuda').float(),
                                   torch.from_numpy(packed_data['transls'])[:, kid].to('cuda').float())[:2]
        gt_smpl_verts = gt_smpl_verts.cpu().numpy() 

        # no object pose 
        device = 'cuda'
        mesh_tensors, meshes = Utils.load_smpl_obj_uvmap(self.video_prefix, use_hy3d=True, hum_only=True)
        self.glctx = dr.RasterizeCudaContext()

        video_path = osp.dirname(args.video)
        video_length = VideoReader(args.video).video_params['length']
        depth_files = {k:osp.join(video_path, f'{self.video_prefix}.{k}.depth-reg.mp4') for k in self.kids}

        assert args.wild_video, 'can only process wild video'
        img1st = imageio.get_reader(args.video).get_data(0)
        render_size = img1st.shape[:2] # h, w 

        # get w2c transform
        w2cs = np.eye(4)[None]
        dists_all = []
        K_all = self.camera_K[None].copy()
        # scale with the scale ratio 
        K_all[:, :2] *= self.scale_ratio 
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

            # use pytorch3d knn to compute the nearest distance from human to object samples 
            if compute_dists:
                continue

            else:
                dists_min = np.zeros(len(smpl_verts_k)) - 1 

            chunk_size = 640 if args.data_source == 'hodome' else 196 
            for i in tqdm(range(0, video_length, chunk_size)):
                smpl_verts_k_chunk = torch.from_numpy(smpl_verts_k[i:i+chunk_size]).to(device).float()
                verts_comb = smpl_verts_k_chunk
                K_rois_chunk = [K_all[kid]]*len(smpl_verts_k_chunk)
                depth_only = i > 50

                # count the time to finish render
                t0 = time.time()
                ret = Utils.nvdiff_color_depth_render(K_rois_chunk, self.glctx, mesh_tensors, render_size, verts_comb, depth_only=depth_only)
                t1 = time.time() # render time: 0.0240 seconds, 64 frames, super fast! and takes only 8GB memory, render time: 0.2275 seconds, 640 frames, 70 GB memory
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










