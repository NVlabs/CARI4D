# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
run pose hypothesis selection.

"""
import sys, os
import os.path as osp

import time
from glob import glob
from copy import deepcopy
import os.path as osp

sys.path.append(os.getcwd())
import cv2, json 
import h5py
import torch
import numpy as np
import joblib
from scipy.spatial.transform import Rotation as R
import Utils
from estimater import cluster_poses
from learning.training.predict_score import make_crop_data_batch, vis_batch_data_scores
import imageio

from prep.fp_behave import FPBehaveVideoProcessor, load_masks
from behave_data.utils import get_intrinsics_unified
from behave_data.const import START_END_FRAMES


class FPFilterTwoDirProcessor(FPBehaveVideoProcessor):
    ""
    def get_chunk_num(self):
        return 300 
    
    @torch.no_grad()
    def process_video(self, kid_to_run, refiner=None, mesh=None, glctx=None, est=None):
        ""
        # for this it is much slower
        self.output_path = self.output_path.replace('.pkl', f'_k{kid_to_run}.pkl')
        if osp.isfile(self.output_path):
            print('Already exists {}, all done'.format(self.output_path))
            return
        if est is None:
            est, glctx, mesh, refiner = self.init_pose_estimator()
        else:
            assert refiner is not None
            assert mesh is not None
            assert glctx is not None

        self.crop_ratio_default = refiner.cfg.crop_ratio
        kids = self.kids
        jump_step, occ_frames_allowed, max_attempts = 30, self.args.occ_frames_allowed, self.args.max_attempts  

        print('kids:', kids)
        args = self.args
        self.zfar = 3.8 if self.args.data_source in ['behave', 'intercap'] and not self.args.wild_video else 8.0  # for HODome, allow larger depth range 
        pose_dict, pose_hist_dict = {}, {}
        pose_best_dict = {}
        reliability_dict = {}
        visibility_dict = {}
        vw = imageio.get_writer(self.output_path.replace('.pkl', f'_filter_k{kid_to_run}.mp4'), 'ffmpeg',
                                fps=2) if args.viz_path is not None else None # with viz, ~0.2s more per-frame 
        video_shape = None
        packed_data, pose_gt_all = None, None 
        if self.video_prefix in START_END_FRAMES:
            last_frame = float(START_END_FRAMES[self.video_prefix][1][1:])
            first_frame = float(START_END_FRAMES[self.video_prefix][0][1:])
            times_cut = [t for t in self.times if round(t, 3) <= last_frame and round(t, 3) >= first_frame]
            print(f'Sequence {self.video_prefix} Cutting times to {len(self.times)} frames, last frame: {times_cut[-1]}, original last frame: {self.times[-1]}, packed data last frame: {START_END_FRAMES[self.video_prefix][1]}')
            # print out first frame info
            print(f'Sequence {self.video_prefix} First frame: {first_frame}, original first frame: {self.times[0]}, after cut: {times_cut[0]}, packed data first frame: {START_END_FRAMES[self.video_prefix][0]}')
            self.times = times_cut
            print(f"after cut: {self.times[0]}->{self.times[-1]}, in total {len(self.times)} frames")

        # prepare vlm model
        assert args.vlm_model == 'none', 'does not support vlm model!'

        if not self.args.wild_video:
            K_all = [get_intrinsics_unified(self.args.data_source, self.video_prefix, kid, self.args.wild_video) for kid in self.kids]
            K_all = np.stack(K_all)
            K_all[:, :2] /= self.scale_ratio # make sure the resolution matches 
        else:
            K_all = [self.camera_K] # this already take into account the scale ratio 
        

        for enum_idx, k in enumerate(kids):
            if kid_to_run is not None and k != kid_to_run:
                continue
            print(f'Processing view {k}')

            visibility_last, pose_last, last_vis_index, reliable_last = 0, None, 0, True  # assuming the first frame is visible and FP is reliable
            timestamp_last_visible = self.times[0]
            vis_thres = args.vis_thres  # 0.7 might be too strong
            print('using visibility threshold: {}'.format(vis_thres))
            poses_i = []
            tar_mask = h5py.File(self.tar_path.replace('_masks_k0.h5', f'_masks_k{k}.h5'), 'r')

            index = 0
            t0 = time.time()
            while index < len(self.times): # TODO: some bug here to handle the last frame
                t1 = time.time()
                print(f'k{k} Frame {index} time: {t1 - t0:.2f}s')
                t0 = t1
                t = self.times[index]
                frame_time = 't{:08.3f}'.format(t) if not args.wild_video else f'{t:06d}'
                if args.data_source not in ['behave']:
                    frame_time = f'{t:06d}'

                mask_h, mask_o = load_masks(self.video_prefix, frame_time, k, tar_mask)

                color, depth = self.load_color_depth(enum_idx, kids, t)
                # merge human and object mask
                try:
                    if np.sum(mask_o > 127) < 800:
                        # use H+O mask to crop
                        refiner.cfg.crop_ratio = 1.0
                        # convert back to 0-255
                        mask_o = mask_o.astype(np.uint8) * 255
                        print(f'Using HO mask for frame {frame_time}, kinect {k}')
                    else:
                        refiner.cfg.crop_ratio = self.crop_ratio_default
                except Exception as e:
                    print(f'failed on frame {frame_time}, kinect {k} due to:', e)
                    if index == 0:
                        print(f'no mask found for frame {frame_time}, kinect {k}, exiting...')
                        return
                    index += 1 
                    continue

                h, w = color.shape[:2]
                mask_o = cv2.resize(mask_o, (int(w / self.scale_ratio), int(h / self.scale_ratio))) > 127
                color = cv2.resize(color, (int(w / self.scale_ratio), int(h / self.scale_ratio)))
                depth = cv2.resize(depth, (int(w / self.scale_ratio), int(h / self.scale_ratio)), cv2.INTER_NEAREST) / 1000.
                depth[(depth < 0.001) | (depth >= self.zfar)] = 0
                t2 = time.time()
                # save the color and depth 

                score_vis = osp.join(args.outpath, f'{self.video_prefix}+{frame_time}+k{k}_score.png')
                refine_vis = osp.join(args.outpath, f'{self.video_prefix}+{frame_time}+k{k}_refine.png')
                pose = est.register(K=K_all[enum_idx], rgb=color, depth=depth, ob_mask=mask_o.astype(bool),
                                    iteration=5,
                                    vis_score_path=score_vis, vis_refine_path=refine_vis,
                                    rgb_only=False, both_depth_and_rgb=True)
                t3 = time.time()
                # do some basic filtering
                tf_to_centered = est.get_tf_to_centered_mesh()
                # Now use some heuristics to further filter out poses
                num_IoU_keep = 30

                # check if there is one good candidate
                if est.poses is None:
                    assert index == 0, f'no pose found for frame {frame_time}'
                    est.poses = torch.eye(4)[None].repeat(est.rot_grid.shape[0]*2, 1, 1).cuda() 
                    print(f"no pose found for frame {frame_time}, set to identity")
                    est.poses[:, :3, :3] = torch.cat([est.rot_grid[:, :3, :3].clone(), est.rot_grid[:, :3, :3].clone()], 0)
                    est.poses[:, :3, 3] = torch.tensor([0, 0, 2.2]).cuda() # set tod a dummy pose so that bbox can be non-zero 
                pose_hist_full = torch.stack([p @ tf_to_centered for p in est.poses])  # pose saved to output dict

                pose_cluster = torch.from_numpy(np.stack(cluster_poses(10, 0.1, est.poses.cpu().numpy(), [np.eye(4)]))).cuda().float()

                # send mask to crop
                mask_h_full, mask_o_full = load_masks(self.video_prefix, frame_time, k, tar_mask)

                mask_o_in = mask_o_full & (~mask_h_full)
                mask_o_in = cv2.resize(mask_o_in.astype(np.uint8) * 255,
                                       (int(w / self.scale_ratio), int(h / self.scale_ratio)))
                mask_h_in = cv2.resize(mask_h_full.astype(np.uint8) * 255,
                                       (int(w / self.scale_ratio), int(h / self.scale_ratio)))
                mask_o_in = np.stack((mask_h_in, mask_o_in, np.zeros_like(mask_o_in)), -1)  # (H, W, 3) for crop
                intersect, ious, masks_render, masks_render_full, pose_data, union = self.render_IoUs(depth, est, glctx,
                                                                                                      mask_o_in, mesh,
                                                                                                      pose_cluster, K_full=K_all[enum_idx])
                t4 = time.time()
                iou_thres = torch.sort(ious, descending=True)[0][min(num_IoU_keep, len(ious)-1)]  # only keep the top 30
                if args.iou_thres1 > 0:
                    iou_thres = torch.sort(ious, descending=True)[0][min(num_IoU_keep, len(ious)-1)]  # only keep the top 30
                    mask_iou = ious >= iou_thres  # 0.4 filter lots of the images already
                else:
                    iou_thres = -1
                    mask_iou = torch.ones_like(ious).bool() # do not filter any 

                # Filter based on visibility and temporal smoothness
                obj_visibilities = torch.sum(masks_render, dim=(-1, -2)) / torch.sum(masks_render_full, dim=(-1, -2))
                timestamp_current = self.controllers[enum_idx].get_current_timestamp()  # actual kinect timestamp
                # activate only when both prev frame, and current frame is highly visible
                if pose_last is not None and ((visibility_last > vis_thres) or (index - last_vis_index < occ_frames_allowed)):  # assume last 30 frame is reliable
                    rot_dists = Utils.geodesic_distance_batch(pose_last[None].expand(len(pose_cluster), -1, -1)[:, :3, :3], pose_cluster[:, :3, :3]) * 180 / torch.pi
                    if not args.wild_video and not args.data_source not in ['behave']:
                        # behave: compute time now
                        time_diff = np.abs(timestamp_current - timestamp_last_visible)
                        rot_thres = 13 + time_diff * 30 * args.angular_velo
                        print(f'Index {index}, time_diff: {time_diff:.4f}, last-vis: {timestamp_last_visible:.2f}, {timestamp_current:.2f}, rot_thres: {rot_thres:.4f}')
                    else:
                        rot_thres = 13 + (index - last_vis_index) * args.angular_velo
                    mask_temporal = (rot_dists < rot_thres) & mask_iou  # 60 deg
                    if torch.sum(mask_temporal) == 0:
                        need_to_jump = False 
                        print('could not find a good temporal template')
                        # try to run more attempts 
                        attempt, rgb_only = 0, False
                        while torch.sum(mask_temporal) == 0:
                            if attempt >= max_attempts:
                                print(f"Failed {max_attempts} attempts... no new try on") # for correct vis
                                mask_iou = ious >= -1.
                                mask_temporal = mask_iou
                                need_to_jump = True
                                break
                            rgb_only = not rgb_only
                            attempt += 1
                            pose = est.register(K=K_all[enum_idx], rgb=color, depth=depth, ob_mask=mask_o.astype(bool),
                                    iteration=5, rgb_only=rgb_only, seed=np.random.randint(60000), 
                                    both_depth_and_rgb=True
                                    )
                            pose_cluster = torch.from_numpy(np.stack(cluster_poses(10, 0.1, est.poses.cpu().numpy(), [np.eye(4)]))).cuda().float()
                            intersect, ious, masks_render, masks_render_full, pose_data, union = self.render_IoUs(depth,
                                                                                                                est,
                                                                                                                glctx,
                                                                                                                mask_o_in,
                                                                                                                mesh,
                                                                                                                pose_cluster,
                                                                                                                K_full=K_all[enum_idx])
                            rot_dists = Utils.geodesic_distance_batch(pose_last[None].expand(len(pose_cluster), -1, -1)[:, :3, :3], pose_cluster[:, :3, :3]) * 180 / torch.pi
                            iou_thres = 0.08 + (torch.mean(ious) - 0.08) * (1-attempt/max_attempts)  # if set too high, good pose candidate will be filtered out by mistake
                            if torch.isnan(iou_thres):
                                iou_thres = 0.
                            mask_iou = ious >= iou_thres # the iou_thres is not reliable
                            obj_visibilities = torch.sum(masks_render, dim=(-1, -2)) / torch.sum(masks_render_full, dim=(-1, -2))
                            # Do not update rot_thres, for fp-hy3d-unidepth-vis0.5-jump-0.3-av0.1-attm10-allow30-sameroth
                            mask_temporal = (rot_dists < rot_thres) & mask_iou # also loose the threshold a bit
                            print(f'Index {index}, frame {frame_time}, attempt {attempt}: temporal {torch.sum(mask_temporal)}, iou: {torch.sum(mask_iou)}, no-depth? {rgb_only} th={rot_thres:.3f}, iou-th={iou_thres:.2f}')
                            
                        pose_hist_full = torch.stack([p @ tf_to_centered for p in est.poses]) 
                            
                    else:
                        need_to_jump = False
                else:
                    mask_temporal = mask_iou
                    rot_thres = -1
                    rot_dists = torch.zeros(len(pose_cluster), dtype=torch.float32)
                    need_to_jump = True if index > 0 else False
                if index + jump_step >= len(self.times):
                    need_to_jump = False # no more jump

                # add vis of this frame
                rot_e = rot_dists[mask_temporal][0] if torch.sum(mask_temporal) > 0 else -1
                info = f'Index={index}/{len(self.times)} {frame_time} cluster: {len(pose_cluster)}, iou: {torch.sum(mask_iou)}, temporal: {torch.sum(mask_temporal)} ({rot_thres:.2f}), mean IoU: {torch.mean(ious):.4f}, last_vis_idx: {last_vis_index}, RE: {rot_e:.3f}'
                mask_iou = mask_temporal
                ids = torch.arange(len(mask_iou)).cuda()[mask_iou]
                ids_filtered = torch.arange(len(mask_iou)).cuda()[~mask_iou]
                
                if args.viz_path is not None:
                    # now visualize the top-K
                    K = 10
                    # add input RGB crop
                    pose_data_rgb = make_crop_data_batch(est.scorer.cfg.input_resize,
                                                        pose_cluster,
                                                        mesh,
                                                        color, depth, K_all[enum_idx],
                                                        crop_ratio=est.scorer.cfg['crop_ratio'], glctx=glctx,
                                                        mesh_tensors=est.mesh_tensors, dataset=est.scorer.dataset,
                                                        cfg=est.scorer.cfg,
                                                        mesh_diameter=est.diameter)
                    if packed_data is not None:
                        rot_errs, pose_gt_ik = self.compute_gt_rot_errors(frame_time, k, packed_data, pose_gt_all, torch.stack(
                            [p @ tf_to_centered for p in pose_cluster]), ret_gt_pose=True)
                    else:
                        rot_errs = torch.zeros(len(pose_cluster), dtype=torch.float32)
                    rot_errs = rot_errs * 180 / np.pi

                    info += f' RE-gt:{rot_errs[int(ids[0])]:.2f}' if len(ids) > 0 else ''
                    info += f' vis_curr: {obj_visibilities[ids[0]]:.2f}' if len(ids) > 0 else ''
                    print(info)
                    pose_data.rgbBs = torch.cat([pose_data.rgbBs,
                                                torch.cat([union[:, None].float(),
                                                            intersect[:, None].repeat(1, 2, 1, 1).float()], 1),
                                                pose_data_rgb.rgbBs], -1)  # (B, 3, H, 2W)
                    score_txts = [
                        f'{iou:.2f}({iou_thres:.2f}) vis: {vis:.2f} (lvis {visibility_last:.2f}) RE: {re:.2f}({rot_thres:.2f}) RE-gt:{re_gt:.2f} idx {index} last vis {last_vis_index} {frame_time[2:]}'
                        for iou, vis, re, re_gt in zip(ious, obj_visibilities, rot_dists, rot_errs)]
                    canvas = vis_batch_data_scores(pose_data, ids=torch.cat([ids, ids_filtered])[:K], scores=score_txts,
                                                no_text=False,
                                                vis_size=120)  # vis both filter and non-filtered top K
                    vw.append_data(canvas)

                # TODO: for this, jump to next K frames, find one reliable frame, and run backwards
                if need_to_jump:
                    print("Trying to jump to find a good frame...")
                    assert index > 0
                    # get buffer and jump
                    success = False
                    rgb_all, depth_all, mask_all_o, mask_all_h, timestamps_all, frame_times_all, refiner_crop_ratio = [], [], [], [], [], [], []
                    ind_jump_start, last_before_jump = index, index
                    while not success:
                        # TODO: handle last frame
                        for i in range(ind_jump_start, min(ind_jump_start+jump_step, len(self.times))):
                            timestamp = self.times[i]
                            frame_time = 't{:08.3f}'.format(timestamp) if not args.wild_video else f'{timestamp:06d}'
                            if args.data_source not in ['behave']:
                                frame_time = f'{timestamp:06d}'

                            color, depth, mask_h, mask_o = self.load_rgbmd(enum_idx, kids, tar_mask, timestamp)
                            mask_all_h.append(mask_h)
                            mask_all_o.append(mask_o)
                            rgb_all.append(color)
                            depth_all.append(depth)
                            timestamps_all.append(self.controllers[enum_idx].get_current_timestamp())
                            frame_times_all.append(frame_time)

                        # run one pose estimation and check
                        score_vis = osp.join(args.outpath, f'{self.video_prefix}+{frame_time}+k{k}_score.png')
                        refine_vis = osp.join(args.outpath, f'{self.video_prefix}+{frame_time}+k{k}_refine.png')
                        # check if we need to change the mask input
                        if np.sum(mask_o > 127) < 800:
                            mask_o_in = (mask_o > 127) | (mask_h > 127)
                            est.refiner.cfg.crop_ratio = 1.0
                        else:
                            mask_o_in = mask_o > 127
                            est.refiner.cfg.crop_ratio = self.crop_ratio_default
                        pose = est.register(K=K_all[enum_idx], rgb=color, depth=depth, ob_mask=mask_o_in,
                                            iteration=5,
                                            vis_score_path=score_vis, vis_refine_path=refine_vis,
                                            rgb_only=False, both_depth_and_rgb=True)
                        # now compute IoU to check if it is a success
                        pose_cluster = torch.from_numpy(np.stack(cluster_poses(10, 0.1, est.poses.cpu().numpy(), [np.eye(4)]))).cuda().float()
                        mask_h_iou = mask_h
                        mask_o_iou = ((mask_o > 127) & (~mask_h_in)).astype(np.uint8)*255
                        mask_iou_in = np.stack((mask_h_iou, mask_o_in, np.zeros_like(mask_o_iou)), -1)
                        intersect, ious, masks_render, masks_render_full, pose_data, union = self.render_IoUs(depth,
                                                                                                              est,
                                                                                                              glctx,
                                                                                                              mask_iou_in,
                                                                                                              mesh,
                                                                                                              pose_cluster,
                                                                                                              K_full=K_all[enum_idx])
                        # set a stronger iou threshold
                        iou_thres = args.iou_thres2 # TODO: verify if this is a reasonable one
                        mask_iou = ious >= iou_thres
                        obj_visibilities = torch.sum(masks_render, dim=(-1, -2)) / torch.sum(masks_render_full, dim=(-1, -2))

                        if (torch.sum(mask_iou)==0 or obj_visibilities[0] < args.vis_thres2) and ind_jump_start + jump_step < len(self.times):
                            # continue jump
                            print(f"from {ind_jump_start} jumped to {i} ({frame_time}) (vis {obj_visibilities[0]:.3f}), no pose candidate reliable.")
                            success = False
                            ind_jump_start = ind_jump_start + jump_step
                        else:
                            success = True
                            index_jumped = i 
                            if torch.sum(mask_iou) == 0:
                                # towards the end of this seq, still no reliable, then just keep use which ever is the first
                                # or should we assume the last frame is good as well?
                                iou_thres = 0.
                                mask_iou = ious >= iou_thres
                            ids = torch.arange(len(mask_iou)).cuda()[mask_iou]
                            ids_filtered = torch.arange(len(mask_iou)).cuda()[~mask_iou]
                            if ind_jump_start + jump_step >= len(self.times):
                                # use GT to pick up the last frame
                                pose_hist_full = torch.stack([p @ tf_to_centered for p in est.poses])  # pose saved to output dict
                                if packed_data is not None:
                                    rot_errs = self.compute_gt_rot_errors(frame_time, k, packed_data, pose_gt_all, pose_hist_full)
                                else:
                                    rot_errs = torch.zeros(len(pose_hist_full), dtype=torch.float32)
                                pidx = torch.argmin(rot_errs)
                                pose_last_jumped = pose_hist_full[pidx] @ tf_to_centered.inverse()
                            else:
                                # pick the pose and run backward
                                pose_last_jumped = pose_cluster[ids[0]]
                            visibility_last_jumped = obj_visibilities[ids[0]]

                            # Visualize this frame
                            # add input RGB crop
                            if args.viz_path is not None:
                                pose_data_rgb = make_crop_data_batch(est.scorer.cfg.input_resize,
                                                                    pose_cluster,
                                                                    mesh,
                                                                    color, depth, K_all[enum_idx],
                                                                    crop_ratio=est.scorer.cfg['crop_ratio'],
                                                                    glctx=glctx,
                                                                    mesh_tensors=est.mesh_tensors,
                                                                    dataset=est.scorer.dataset,
                                                                    cfg=est.scorer.cfg,
                                                                    mesh_diameter=est.diameter)
                                if packed_data is not None:
                                    rot_errs, pose_gt_ik = self.compute_gt_rot_errors(frame_time, enum_idx, packed_data, pose_gt_all, torch.stack([p @ tf_to_centered for p in pose_cluster]), ret_gt_pose=True)
                                else:
                                    rot_errs = torch.zeros(len(pose_cluster), dtype=torch.float32)
                                rot_errs = rot_errs * 180 / np.pi
                                info = f'found good pose candidate at {i}, {frame_time}, (vis {obj_visibilities[0]:.3f} ), cluster: {len(pose_cluster)}, iou: {torch.sum(mask_iou)}'
                                info += f' RE-gt:{rot_errs[int(ids[0])]:.3f}'
                                print(info)
                                pose_data.rgbBs = torch.cat([pose_data.rgbBs,
                                                            torch.cat([union[:, None].float(),
                                                                        intersect[:, None].repeat(1, 2, 1, 1).float()], 1),
                                                            pose_data_rgb.rgbBs], -1)
                                rot_dists = torch.zeros_like(rot_errs)
                                score_txts = [
                                    f'{iou:.2f}({iou_thres:.2f}) vis: {vis:.2f} (lvis {visibility_last:.2f}) RE: {re:.2f}(-1) RE-gt:{re_gt:.2f} idx {i} last vis {last_vis_index} {frame_time[2:]}'
                                    for iou, vis, re, re_gt in zip(ious, obj_visibilities, rot_dists, rot_errs)]
                                canvas = vis_batch_data_scores(pose_data, ids=torch.cat([ids, ids_filtered])[:10],
                                                            scores=score_txts, no_text=False,
                                                            vis_size=120)
                                vw.append_data(canvas)

                            # run backward
                            activate_temporal = True
                            timestamp_last_visible = timestamps_all[-1]
                            last_vis_index = len(rgb_all)
                            pose_last = pose_last_jumped
                            for i in range(1, len(rgb_all)):
                                ind_backward = len(rgb_all) - i - 1
                                mask_o, mask_h = mask_all_o[ind_backward], mask_all_h[ind_backward]
                                color, depth = rgb_all[ind_backward], depth_all[ind_backward]
                                timestamp_current = timestamps_all[ind_backward]
                                frame_time = frame_times_all[ind_backward] if not args.wild_video else f'{timestamp_current:06d}'
                                if np.sum(mask_o > 127) < 600:
                                    mask_o_in = (mask_o > 127) | (mask_h > 127)
                                    est.refiner.cfg.crop_ratio = 1.0
                                else:
                                    mask_o_in = mask_o > 127
                                    est.refiner.cfg.crop_ratio = self.crop_ratio_default

                                pose = est.register(K=K_all[enum_idx], rgb=color, depth=depth, ob_mask=mask_o_in,
                                                    iteration=5,
                                                    vis_score_path=None, vis_refine_path=None,
                                                    rgb_only=False, both_depth_and_rgb=True)
                                # use mask iou + pose to filter again
                                pose_cluster = torch.from_numpy(np.stack(cluster_poses(10, 0.1, est.poses.cpu().numpy(), [np.eye(4)]))).cuda().float()
                                mask_h_iou = mask_h
                                mask_o_iou = ((mask_o > 127) & (~mask_h_in)).astype(np.uint8) * 255
                                mask_iou_in = np.stack((mask_h_iou, mask_o_in, np.zeros_like(mask_o_iou)), -1)
                                intersect, ious, masks_render, masks_render_full, pose_data, union = self.render_IoUs(
                                    depth,
                                    est,
                                    glctx,
                                    mask_iou_in,
                                    mesh,
                                    pose_cluster,
                                    K_full=K_all[enum_idx])
                                # set a stronger iou threshold
                                iou_thres = torch.sort(ious, descending=True)[0][num_IoU_keep]  # only keep the top 30
                                mask_iou = ious >= iou_thres
                                if torch.sum(mask_iou) == 0:
                                    # set it to all true
                                    mask_iou[:] = True
                                    print(f"NO valid iou_thres, set it to all true!, frame index {ind_backward}")
                                obj_visibilities = torch.sum(masks_render, dim=(-1, -2)) / torch.sum(masks_render_full, dim=(-1, -2))
                                if activate_temporal:
                                    rot_dists = Utils.geodesic_distance_batch(pose_last[None].expand(len(pose_cluster), -1, -1)[:, :3, :3],pose_cluster[:, :3, :3]) * 180 / torch.pi
                                    if not args.wild_video:
                                        # behave: compute time now
                                        time_diff = np.abs(timestamp_current - timestamp_last_visible)
                                        rot_thres = 13 + time_diff * 30 * args.angular_velo # this threshold seems to be too large?? 
                                        print(f'Index {ind_backward}, time_diff: {time_diff:.4f}, last: {timestamp_last_visible:.4f}, {timestamp_current:.4f}, rot_thres: {rot_thres:.4f}')
                                    else:
                                        rot_thres = 13 + (np.abs(index - last_vis_index) - 1) * args.angular_velo  # the longer, the higher the thres is
                                    mask_temporal = (rot_dists < rot_thres) & mask_iou  # 60 deg
                                    if torch.sum(mask_temporal)==0:
                                        # TODO: should we try more attempts? 
                                        activate_temporal = False # do not use temporal to masking anymore, just let it do what ever it wants, this is heavily occluded frames
                                        mask_temporal = mask_iou
                                        print(f'disabling temporal filtering at {timestamp_current}, index {ind_backward}')
                                else:
                                    rot_thres = -1
                                    mask_temporal = mask_iou
                                    rot_dists = torch.zeros(len(pose_cluster), dtype=torch.float32)
                                

                                info = f'Backward: index={ind_backward + last_before_jump} {frame_time} cluster: {len(pose_cluster)}, iou: {torch.sum(mask_iou)}, temporal: {torch.sum(mask_temporal)} ({rot_thres:.2f}), mean IoU: {torch.mean(ious):.4f}, last_vis: {last_vis_index}, RE: {rot_dists[mask_temporal][0] if torch.sum(mask_temporal) > 0 else -10:.3f}'
                                # now pick the first pose candidate
                                mask_iou = mask_temporal
                                ids = torch.arange(len(mask_iou)).cuda()[mask_iou]
                                ids_filtered = torch.arange(len(mask_iou)).cuda()[~mask_iou]
                                visibility_last = obj_visibilities[ids[0]]

                                pose_last = pose_cluster[ids[0]] # update pose
                                if frame_time not in pose_hist_dict:
                                    pose_hist_dict[frame_time] = []
                                if frame_time not in pose_dict:
                                    pose_dict[frame_time] = []
                                if frame_time not in pose_best_dict:
                                    pose_best_dict[frame_time] = []
                                if frame_time not in reliability_dict:
                                    reliability_dict[frame_time] = []
                                if frame_time not in visibility_dict:
                                    visibility_dict[frame_time] = []
                                pose_hist_full = torch.stack([p @ tf_to_centered for p in est.poses])
                                pose_best_dict[frame_time].append((pose_last @ tf_to_centered).cpu().numpy())
                                pose_dict[frame_time].append((pose_last @ tf_to_centered).cpu().numpy())
                                pose_hist_dict[frame_time].append(pose_hist_full.cpu().numpy())
                                visibility_dict[frame_time].append(visibility_last.cpu().numpy())

                                if visibility_last > vis_thres:
                                    last_vis_index = ind_backward  # indicate when was the time that the object is visible
                                    timestamp_last_visible = timestamp_current  # update this as well
                                if args.viz_path is not None:
                                    K = 10
                                    # add input RGB crop
                                    pose_data_rgb = make_crop_data_batch(est.scorer.cfg.input_resize,
                                                                        pose_cluster,
                                                                        mesh,
                                                                        color, depth, K_all[enum_idx],
                                                                        crop_ratio=est.scorer.cfg['crop_ratio'],
                                                                        glctx=glctx,
                                                                        mesh_tensors=est.mesh_tensors,
                                                                        dataset=est.scorer.dataset,
                                                                        cfg=est.scorer.cfg,
                                                                        mesh_diameter=est.diameter)
                                    if packed_data is not None:
                                        rot_errs, pose_gt_ik = self.compute_gt_rot_errors(frame_time, enum_idx, packed_data,
                                                                                    pose_gt_all, torch.stack([p @ tf_to_centered for p in pose_cluster]), ret_gt_pose=True)
                                    else:
                                        rot_errs = torch.zeros(len(pose_cluster), dtype=torch.float32)
                                    rot_errs = rot_errs * 180 / np.pi

                                    info += f' RE-gt:{rot_errs[int(ids[0])]:.3f}'
                                    if args.viz_path is not None:
                                        print(info)
                                    pose_data.rgbBs = torch.cat([pose_data.rgbBs,
                                                                torch.cat([union[:, None].float(),
                                                                            intersect[:, None].repeat(1, 2, 1, 1).float()],1),
                                                                pose_data_rgb.rgbBs], -1)  # (B, 3, H, 2W)

                                    score_txts = [f'{iou:.2f}({iou_thres:.2f}) vis: {vis:.2f} (lvis {visibility_last:.2f}) RE: {re:.2f}({rot_thres:.2f}) RE-gt:{re_gt:.2f} idx {ind_backward + last_before_jump} last vis {last_vis_index+last_before_jump} {frame_time[2:]}'
                                        for iou, vis, re, re_gt in zip(ious, obj_visibilities, rot_dists, rot_errs)]
                                    canvas = vis_batch_data_scores(pose_data, ids=torch.cat([ids, ids_filtered])[:K],
                                                                scores=score_txts, no_text=False,
                                                                vis_size=120)
                                    vw.append_data(canvas)

                            # once done, rewind back to the jumped frame
                            timestamp_last_visible = timestamps_all[-1]
                            last_vis_index = index_jumped
                            index = index_jumped
                            visibility_last = visibility_last_jumped
                            pose_last = pose_last_jumped
                            print('backward done, continuing forward!')

                else:
                    # # just keep current
                    # now update pose and others 
                    if index == 0 and not args.wild_video and packed_data is not None:
                        # for the first frame, use GT to pick the best, assuming FP always work for the 1st frame
                        rot_errs = self.compute_gt_rot_errors(frame_time, enum_idx, packed_data, pose_gt_all, pose_hist_full)
                        pidx = torch.argmin(rot_errs)
                        pose_last = pose_hist_full[pidx] @ tf_to_centered.inverse()
                    else:
                        # use all the tricks to check
                        pose_last = pose_cluster[ids[0]] if len(ids) > 0 else pose_cluster[0]

                    visibility_last = obj_visibilities[ids[0]] if len(ids) > 0 else obj_visibilities[0]
                    if visibility_last > vis_thres:
                        last_vis_index = index  # indicate when was the time that the object is visible
                        timestamp_last_visible = timestamp_current  # update this as well
                    if video_shape is None and args.viz_path is not None:
                        video_shape = canvas.shape  # fix the video shape

                    if frame_time not in pose_hist_dict:
                        pose_hist_dict[frame_time] = []
                    if frame_time not in pose_dict:
                        pose_dict[frame_time] = []
                    if frame_time not in pose_best_dict:
                        pose_best_dict[frame_time] = []
                    if frame_time not in reliability_dict:
                        reliability_dict[frame_time] = []
                    if frame_time not in visibility_dict:
                        visibility_dict[frame_time] = []
                    pose_best_dict[frame_time].append((pose_last @ tf_to_centered).cpu().numpy())
                    pose_dict[frame_time].append((pose_last @ tf_to_centered).cpu().numpy())
                    pose_hist_dict[frame_time].append(pose_hist_full.cpu().numpy())
                    reliability_dict[frame_time].append(reliable_last)
                    visibility_dict[frame_time].append(visibility_last.cpu().numpy())
                    index += 1
                torch.cuda.empty_cache()

        # save results
        # rewrite this chunk to sort the results by keys
        kids = [kid_to_run]
        poses_all = [np.stack(v) for k, v in sorted(pose_dict.items()) if len(v) == len(kids)]
        poses_all_hist = [np.stack(v) for k, v in sorted(pose_hist_dict.items()) if len(v) == len(kids)]
        pose_best_dict = [np.stack(v) for k, v in sorted(pose_best_dict.items()) if len(v) == len(kids)]
        reliability_dict = [np.stack(v) for k, v in sorted(reliability_dict.items()) if len(v) == len(kids)]
        frames = [k for k, v in sorted(pose_dict.items()) if len(v) == len(kids)]
        visibility = [np.stack(v) for k, v in sorted(visibility_dict.items()) if len(v) == len(kids)]

        pose_all = np.stack(poses_all)  # T, K, 4, 4
        pose_all_hist = np.stack(poses_all_hist)  # (T, K, N, 4, 4)
        pose_pred = np.stack(pose_best_dict)
        reliability = np.stack(reliability_dict)
        visibility = np.stack(visibility)  # T, K
        out_dict = {"fp_poses": pose_all, "frames": frames, 'fp_poses_all': pose_all_hist, 'fp_best': pose_pred,
                    'visibility': visibility}
        for k in out_dict.keys():
            print(k, len(out_dict[k]))
        out_dict['vis_thres'] = args.vis_thres
        joblib.dump(out_dict, self.output_path)
        print('all done, saved to', self.output_path, 'pose_all:', pose_all.shape)

    def render_IoUs(self, depth, est, glctx, mask_o_in, mesh, pose_cluster, K_full=None):
        pose_data = make_crop_data_batch(est.scorer.cfg.input_resize,
                                         pose_cluster,
                                         mesh,
                                         mask_o_in, depth, K_full if K_full is not None else self.camera_K,
                                         crop_ratio=est.scorer.cfg['crop_ratio'], glctx=glctx,
                                         mesh_tensors=est.mesh_tensors, dataset=est.scorer.dataset,
                                         cfg=est.scorer.cfg,
                                         mesh_diameter=est.diameter)
        # now compute mask IoU
        masks_render_full = pose_data.depthAs[:, 0] > 0  # xyz is normalized, so z is not always positive, but depth is correct
        masks_h = pose_data.rgbBs[:, 0] > 0
        masks_o = pose_data.rgbBs[:, 1] > 0
        masks_render = masks_render_full & (~masks_h)  # (B, H, W), human mask can occlude the full object mask!
        union = masks_o | masks_render
        intersect = masks_render & masks_o  # TODO: improve the mask, this can help to obtain a more meaningful IoUs.
        ious = torch.sum(intersect, dim=(-1, -2)) / torch.sum(union, dim=(-1, -2))
        return intersect, ious, masks_render, masks_render_full, pose_data, union

    def load_gt_data(self, kids):
        if not self.args.wild_video:
            packed_data = joblib.load(
                f'/home/xianghuix/datasets/behave/behave-packed/{self.video_prefix}_GT-packed.pkl')
            frames_packed = packed_data['frames']
            gt_poses = np.eye(4)[None].repeat(len(frames_packed), 0)
            gt_poses[:, :3, :3] = R.from_rotvec(packed_data['obj_angles']).as_matrix()
            gt_poses[:, :3, 3] = packed_data['obj_trans']
            if self.args.data_source in ['behave', 'procigen']:
                from behave_data.utils import load_kinect_poses_back
                w2c_rots, w2c_trans = load_kinect_poses_back(
                    osp.join('/home/xianghuix/datasets/behave', 'calibs', self.video_prefix.split('_')[0], 'config'),
                    kids)
            elif self.args.data_source in ['hodome', 'intercap', 'imhd']:
                extrinsics = packed_data['extrinsics']  # (K, 4, 4)
                w2c_rots, w2c_trans = extrinsics[:, :3, :3], extrinsics[:, :3, 3]

            pose_gt_all = []
            for k in range(len(kids)):
                w2c = np.eye(4)
                w2c[:3, :3] = w2c_rots[k]
                w2c[:3, 3] = w2c_trans[k]
                pose_gt_i = np.matmul(w2c[None], gt_poses)
                pose_gt_all.append(pose_gt_i)
            pose_gt_all = np.stack(pose_gt_all, 1)
            return packed_data, pose_gt_all
        return None, None

    def compute_gt_rot_errors(self, frame_time, k, packed_data, pose_gt_all, pose_hist, ret_gt_pose=False):
        gt_idx = packed_data['frames'].index(frame_time)
        pose_gt_ik = pose_gt_all[gt_idx, k]
        rot_errs = Utils.geodesic_distance_batch(
            torch.from_numpy(pose_gt_ik)[None].repeat(len(pose_hist), 1, 1).float().cuda()[:, :3, :3],
            pose_hist[:, :3, :3])
        if ret_gt_pose:
            return rot_errs, pose_gt_ik
        return rot_errs


    def load_rgbmd(self, enum_idx, kids, tar_mask, timestamp):
        color, depth = self.load_color_depth(enum_idx, kids, timestamp)

        mask_h, mask_o = load_masks(self.video_prefix, self.get_time_str(timestamp), kids[enum_idx], tar_mask)

        # resize
        h, w = color.shape[:2]
        color = cv2.resize(color, (int(w / self.scale_ratio), int(h / self.scale_ratio)))
        depth = cv2.resize(depth, (int(w / self.scale_ratio), int(h / self.scale_ratio)), cv2.INTER_NEAREST) / 1000.
        depth[(depth < 0.001) | (depth >= self.zfar)] = 0
        mask_o = cv2.resize(mask_o, (int(w / self.scale_ratio), int(h / self.scale_ratio)))
        mask_h = cv2.resize(mask_h, (int(w / self.scale_ratio), int(h / self.scale_ratio)))
        return color, depth, mask_h, mask_o

    @staticmethod
    def get_parser():
        parser = FPBehaveVideoProcessor.get_parser()
        parser.add_argument('--iou_thres2', default=0.5, type=float)
        parser.add_argument('--iou_thres1', default=0.5, type=float)
        parser.add_argument('--vlm_model', default='none', type=str, choices=['none', 'gpt-4o', 'gpt-5'])
        parser.add_argument('--vis_thres2', default=0.7, type=float)
        parser.add_argument('--angular_velo', default=2.5, type=float)
        parser.add_argument('--occ_frames_allowed', default=15, type=int)
        # add attemps
        parser.add_argument('--max_attempts', default=5, type=int)
        

        return parser

def _child_run_kid(kid, args):
    """Run one view in an isolated child process.
    Creates CUDA/context-heavy objects inside the child to avoid pickling issues.
    """
    import torch

    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(0)
        except Exception:
            pass

    # check if the video is a file or a pattern
    if osp.isfile(args.video):
        videos = [args.video]
    else:
        videos = sorted(glob(args.video))
    print(f"In total {len(videos)} video files")
    args = deepcopy(args)
    chunk_size = len(videos) // 320
    if args.index is not None:
        videos = videos[args.index * chunk_size:(args.index + 1) * chunk_size]
    print(f"Processing {len(videos)} video files, first video: {videos[0]}, last video: {videos[-1]}")

    for video in videos:
        args.video = video
        processor_child = FPFilterTwoDirProcessor(args) # this holds some thread lock 
        processor_child.process_video(kid)
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

def process_video(args):
    # use multiple processes to process different views of this video 
    import multiprocessing as mp

    mp.set_start_method('spawn', force=True)
    ctx = mp.get_context('spawn')

    procs = []  # run 4 processes in parallel, around 7s/frame 
    kids = args.cameras # run 2 processes in parallel, 4.8s/frame, reasonable speed up 

    # Oct30: ablation, run on the selected view
    if 'gtdmap' in args.outpath:
        selected_views = json.load(open('splits/selected-views-map.json'))
        video_prefix = osp.basename(args.video).split('.')[0]
        kids = [int(selected_views[video_prefix][1])] if video_prefix in selected_views else args.cameras
        args.cameras = kids 
        print(f"running kids {kids} for seq {video_prefix}")

    for k in kids:
        p = ctx.Process(target=_child_run_kid, args=(k, args))
        p.start()
        time.sleep(5) # to avoid race condition
        procs.append(p)
    for p in procs:
        p.join()

if __name__ == '__main__':
    parser = FPFilterTwoDirProcessor.get_parser()
    args = parser.parse_args()
    import traceback
    from glob import glob

    try:
        process_video(args)
    except Exception as e:
        print(args.video, 'failed')
        traceback.print_exc()