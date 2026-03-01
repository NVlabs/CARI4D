# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
align monodepth to human estimation, i.e. use human as an anchor to provide absolute translation
this is usually helpful for in the wild videos where mono depth is not temporally stable
"""
import sys, os

sys.path.append(os.getcwd())
import cv2
import numpy as np
import glob, json
from tqdm import tqdm
import joblib
import os.path as osp
import imageio
import Utils
from videoio import VideoReader
from videoio import Uint16Writer
import torch

from lib_smpl import get_smpl
import nvdiffrast.torch as dr
from prep.align_monodmap import MonodepthAligner


class Monodepth2HumanAligner(MonodepthAligner):
    def get_target_dfile(self, k, method_name, video_root) -> str:
        "point to the temporally rendered depth file"
        dfile_hum = osp.join(video_root.replace(f'/{method_name}', '/videos'), f'{self.video_prefix}.{k}.depth_render.mp4')
        return dfile_hum

    def align(self, args, kid_to_run=None):
        "first render the seq as one video, run alignment, and then clean up"
        self.render_dmap(args)
        super().align(args, kid_to_run)
        # clean up temporally depth
        depth_files = self.get_depth_files(args)
        for file in depth_files:
            os.system(f'rm {file}')
            print(f"File {file} removed.")

    def render_dmap(self, args):
        packed_file = f'{args.nlf_path}/{self.video_prefix}_params.pkl'
        packed_data = joblib.load(packed_file)
        # check the length of packed frames, should be the same as the video length
        video_length = len(self.times)
        if len(packed_data['frames']) != video_length:
            print(f'{packed_file} frames length is not the same as the video length, {len(packed_data["frames"])} != {video_length}')
            return
        depth_files = self.get_depth_files(args)
        for file in depth_files:
            if osp.isfile(file):
                os.remove(file) # clean up first

        gender = packed_data['gender']
        smpl_model = get_smpl(gender, hands=True).to('cuda')
        kid = 0
        gt_smpl_verts, gt_smpl_joints = smpl_model(torch.from_numpy(packed_data['poses'][:, kid]).to('cuda').float(),
                                                   torch.from_numpy(packed_data['betas'][:, kid]).to('cuda').float(),
                                                   torch.from_numpy(packed_data['transls'])[:, kid].to('cuda').float())[:2]
        gt_smpl_verts = gt_smpl_verts.cpu().numpy()

        # Prepare rendering
        # no object pose
        device = 'cuda'
        mesh_tensors, meshes = Utils.load_smpl_obj_uvmap(self.video_prefix, use_hy3d=True, hum_only=True)
        self.glctx = dr.RasterizeCudaContext()

        video_length = VideoReader(args.video).video_params['length']
        assert args.wild_video, 'can only process wild video'
        img1st = imageio.get_reader(args.video).get_data(0)
        render_size = img1st.shape[:2]  # h, w

        w2cs = np.eye(4)[None]
        dists_all = []
        K_all = self.camera_K[None].copy()
        # scale with the scale ratio
        K_all[:, :2] *= self.scale_ratio
        compute_dists, dist_key = False, 'dists_h2o'
        kids = range(len(w2cs)) if not compute_dists else [0]  # if compute dists, only compute one view is enough
        for kid in kids:
            outfile_k = depth_files[kid].replace('.depth_render.mp4', '_done.txt')
            if osp.isfile(outfile_k) and not compute_dists:
                print(f'{outfile_k} already exists, skipping')
                continue
            depth_writer = None
            if args.viz_path is not None:
                viz_path = f'outputs/viz-render-dmap/{self.video_prefix}_k{kid}.mp4'
                vw = imageio.get_writer(viz_path, 'FFMPEG', fps=6)
            w2c = w2cs[kid]
            smpl_verts_k = np.matmul(gt_smpl_verts, w2c[:3, :3].T) + w2c[:3, 3]
            dists_min = np.zeros(len(smpl_verts_k)) - 1  # dummy data

            chunk_size = 640 if args.data_source == 'hodome' else 196
            for i in tqdm(range(0, video_length, chunk_size)):
                smpl_verts_k_chunk = torch.from_numpy(smpl_verts_k[i:i + chunk_size]).to(device).float()
                verts_comb = smpl_verts_k_chunk
                K_rois_chunk = [K_all[kid]] * len(smpl_verts_k_chunk)
                depth_only = i > 50

                ret = Utils.nvdiff_color_depth_render(K_rois_chunk, self.glctx, mesh_tensors, render_size, verts_comb,
                                                      depth_only=depth_only)
                if depth_only:
                    depth, xyz_map = ret
                else:
                    color, depth, xyz_map = ret

                # write the depth to file
                if depth_writer is None:
                    H, W = depth[0].shape[:2]
                    os.makedirs(osp.dirname(depth_files[kid]), exist_ok=True)
                    depth_writer = Uint16Writer(depth_files[kid], (W, H), fps=30)

                depth = (depth * 1000).cpu().numpy().astype(np.uint16)

                for d in depth:
                    depth_writer.write(d)

                # now visualize
                if args.viz_path is not None and i < 50:
                    # load the RGB
                    for j in range(i, i + len(smpl_verts_k_chunk)):
                        if j % 5 != 0:
                            continue  # only viz 50 frames
                        frame_time = self.get_time_str(j)
                        # now load RGB
                        color_orig, _ = self.load_color_depth(kid, self.kids, j)
                        rend_j = (color[j - i][:, :, :3].cpu().numpy() * 255).astype(np.uint8)
                        # visualize depth and color
                        xyz_j = xyz_map[j - i].cpu().numpy()
                        median = np.median(xyz_j[:, :, 2])
                        xyz_j[:, :, 2] = xyz_j[:, :, 2] - median
                        xyz_j_vis = (np.clip(xyz_j + 0.5, 0, 1.) * 255).astype(np.uint8)
                        # visualize the depth
                        dmax = np.max(depth[j - i])
                        dvis = (depth[j - i].astype(np.float32) / dmax * 255).astype(np.uint8)[:, :, None].repeat(3, -1)
                        comb = np.concatenate(
                            [color_orig, rend_j, xyz_j_vis, dvis, cv2.addWeighted(color_orig, 0.5, rend_j, 0.5, 0)],
                            axis=1)
                        comb = cv2.resize(comb, (comb.shape[1] // 4, comb.shape[0] // 4))
                        # add text and kid to the image
                        cv2.putText(comb, f'{frame_time} k{kid} dist {dists_min[j]:.3f}', (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        vw.append_data(comb)

            # close the writer and print video path
            if depth_writer is not None:
                depth_writer.close()
            if args.viz_path is not None:
                vw.close()
                print(f'video saved to {viz_path}, done')
            # save the dists to packed file
        if compute_dists:
            packed_data[dist_key] = np.stack(dists_all, 0)
            joblib.dump(packed_data, packed_file)
            print(f'{packed_file} saved')

    def get_depth_files(self, args) -> list[str]:
        video_root = osp.dirname(args.video)
        method_name = osp.basename(video_root)
        depth_files = [self.get_target_dfile(k, method_name, video_root) for k in self.kids]
        return depth_files

def child_run(args, kid_to_run):
    aligner = Monodepth2HumanAligner(args)
    aligner.align(args, kid_to_run)


if __name__ == '__main__':
    args = MonodepthAligner.get_parser().parse_args()
    import multiprocessing as mp
    import time
    from copy import deepcopy

    mp.set_start_method('spawn', force=True)
    ctx = mp.get_context('spawn')

    if osp.isfile(args.video):
        videos = [args.video]
    else:
        videos = sorted(glob.glob(args.video))

    chunk_size = len(videos) // 8 + 1
    if args.index is not None:
        videos = videos[args.index * chunk_size:(args.index + 1) * chunk_size]
    max_procs_per_gpu = 2
    procs = []
    print(f'running {len(videos)} videos, first video: {videos[0]}, last video: {videos[-1]}')
    args_orig = args
    kids = [0, 1, 2, 3] if args.data_source != 'intercap' else [0, 1, 2, 3, 4, 5]
    if args.wild_video:
        kids = [0]
    for i in range(len(videos)):
        args = deepcopy(args_orig)
        args.video = videos[i]
        for k in kids:
            # check if the
            proc = ctx.Process(target=child_run, args=(args, k))
            proc.start()
            time.sleep(2) # to avoid race condition
            procs.append(proc)
        for p in procs:
            p.join()
    print('all done')