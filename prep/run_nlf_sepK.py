# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os, sys

import h5py

sys.path.append(os.getcwd())
import cv2
import torch
import torchvision  # <--- This registers the 'nms' operator, required for loading NLF model.
from glob import glob
from tqdm import tqdm
import numpy as np
from smplfitter.pt import BodyModel, BodyFitter
import os.path as osp
import joblib
from behave_data.behave_video import BaseBehaveVideoData
from tools import img_utils
from lib_smpl import get_smpl, SMPL_MODEL_ROOT
from behave_data.const import _sub_gender, EXCLUDE_OBJECTS
from behave_data.utils import get_intrinsics_unified

NLF_MODEL_PATH = 'weights/nlf_l_multi_0.3.2.torchscript'

class ViewSpecificNLFRunner(BaseBehaveVideoData):
    @torch.no_grad()
    def run(self, args, model=None):
        ""
        outfile = osp.join(args.outpath, f'{self.video_prefix}_params.pkl')
        if osp.isfile(outfile) and not args.redo:
            print(f'{outfile} already exists, skipping...all done')
            return
        model = torch.jit.load(NLF_MODEL_PATH).cuda().eval() if model is None else model # works better for torch>=2.4
        device = 'cuda'
        gender = _sub_gender[self.video_prefix.split('_')[1]]
        fitter_smplh = BodyFitter(BodyModel('smplh', gender, model_root=SMPL_MODEL_ROOT).to('cuda')).to(device)
        
        if args.wild_video:
            K_all = np.array([self.camera_K])
        else:
            K_all = [get_intrinsics_unified(self.args.data_source, self.video_prefix, kid, self.args.wild_video) for kid in self.kids]
            K_all = np.array(K_all) 

        nlf_data = {}
        kids = self.kids
        tars = [h5py.File(self.tar_path.replace('_masks_k0.h5', f'_masks_k{k}.h5'), 'r') for k in kids]
        for kid_idx, kid in enumerate(self.kids):
            nlf_data[kid] = {
                'poses': [],
                'betas': [],
                'transls': [],
                'frames': [],
            }
            chunk_size = 120 
            for i in tqdm(range(0, len(self.times), chunk_size), desc=f"processing {self.video_prefix}/{kid}"):
                t = self.times[i]
                frame_time = self.get_time_str(t)
                
                images_all, masks_all, frames_chunk = [], [], []
                for j in range(i, min(i + chunk_size, len(self.times))):
                    t = self.times[j]
                    frame_time = 't{:08.3f}'.format(t) if not args.wild_video else f'{t:06d}'
                    if args.data_source not in ['behave']:
                        frame_time = f'{t:06d}'
                    mask_name = f'{self.video_prefix}/{frame_time}-k{kid}.person_mask.png'
                    try:
                        masks = tars[kid_idx][mask_name][:].astype(np.uint8)*255
                        masks_all.append(masks)
                        actual_times = np.array([self.controllers[ii].get_closest_time(t) for ii, x in enumerate(kids)])
                        best_kid = np.argmin(np.abs(actual_times - t))
                        actual_time = actual_times[best_kid]
                        images_all.append(self.controllers[kid_idx].get_closest_frame(actual_time))
                        frames_chunk.append(frame_time)
                    except Exception as e:
                        print(f'error loading mask for frame {frame_time}, k{kid}: {e}')
                        continue

                if len(images_all) == 0:
                    continue
                # now compute bboxes 
                bboxes = []
                for iii, (img, mask) in enumerate(zip(images_all, masks_all)):
                    i_curr = iii 
                    while np.sum(mask>127) < 20:
                        i_curr -= 1
                        if i_curr < 0:
                            break # if it failed, it can be due to no human found in the frame, this seq is not usable. 
                        mask = masks_all[i_curr] # revert back to previous frame mask 
                        print(f"warning: no human found in this frame, revert back to previous frame mask, index {iii}->{i_curr}")
                    bmin, bmax = img_utils.masks2bbox([mask])
                    bbox = np.concatenate([bmin, bmax - bmin]) # xywh
                    bbox = np.append(bbox, [1.01]) # add confidence score
                    bboxes.append(torch.from_numpy(bbox)[None].cuda().float())
                image_tensor = torch.from_numpy(np.stack(images_all)).permute(0, 3, 1, 2).cuda()
                pred = model.detect_smpl_batched(image_tensor, extra_boxes=bboxes, detector_threshold=1.0, suppress_implausible_poses=False, 
                        intrinsic_matrix=torch.from_numpy(K_all[kid_idx]).cuda().float()[None].to(device))  # higher threshold to prevent using any detector out
                
                verts = torch.cat(pred['vertices3d'], dim=0)/1000. # (K, N, 3)
                cent = torch.mean(verts, dim=1, keepdim=True)
                # do filtering
                mask = cent[:, 0, 2] < 8.0  # z should be smaller than 8.0
                verts = verts[mask]
                if len(verts) != len(frames_chunk):
                    print(f'invalid prediction found on {frame_time}!')
                    # don't do any filtering 
                    verts = torch.cat(pred['vertices3d'], dim=0)/1000.
                    assert len(verts) == len(frames_chunk), f'something wrong in this frame {frame_time}!'

                verts_new = verts 
                fit_res_smplh = fitter_smplh.fit(verts_new, num_iter=3,
                                                beta_regularizer=1,
                                                requested_keys=['shape_betas', 'trans', 'vertices', 'pose_rotvecs'])
                fit_res = fit_res_smplh

                nlf_data[kid]['poses'].append(fit_res['pose_rotvecs'].cpu().numpy()) 
                nlf_data[kid]['betas'].append(fit_res['shape_betas'].cpu().numpy())
                nlf_data[kid]['transls'].append(fit_res['trans'].cpu().numpy())
                nlf_data[kid]['frames'].extend(frames_chunk)
                assert len(fit_res['pose_rotvecs']) == len(frames_chunk), f'incomplete prediction {fit_res["pose_rotvecs"].shape} on frame {frame_time}!'
                assert len(fit_res['shape_betas']) == len(frames_chunk), f'incomplete prediction {fit_res["shape_betas"].shape} on frame {frame_time}!'
                assert len(fit_res['trans']) == len(frames_chunk), f'incomplete prediction {fit_res["trans"].shape} on frame {frame_time}!'

                if i == 0:
                    print("Translation:", fit_res['trans'][0], 'please check if this is reasonable!')

            nlf_data[kid]['poses'] = np.concatenate(nlf_data[kid]['poses'], 0)
            nlf_data[kid]['betas'] = np.concatenate(nlf_data[kid]['betas'], 0)
            nlf_data[kid]['transls'] = np.concatenate(nlf_data[kid]['transls'], 0)

        # stack all views into shape (N, K, D)
        nlf_data_all = {}
        for k in nlf_data[kids[0]].keys():
            if k == 'frames':
                continue
            nlf_data_all[k] = np.stack([nlf_data[kid][k] for kid in kids], 1)
        nlf_data_all['frames'] = nlf_data[kids[0]]['frames'] 
        nlf_data_all['gender'] = gender
        nlf_data_all['kids'] = kids
        joblib.dump(nlf_data_all, outfile)
        print('all done, saved to', outfile)

    
def main():
    parser = BaseBehaveVideoData.get_parser()
    args = parser.parse_args()
    args.nodepth = True
    # process multiple videos if the given video path is a pattern, use glob to glob all videos 
    model = torch.jit.load(NLF_MODEL_PATH).cuda().eval() # preload the model to avoid loading it multiple times

    if osp.isfile(args.video):
        videos = [args.video]
    else:
        videos = sorted(glob(args.video))
    print(f'in total {len(videos)} videos')
    videos_filter = []
    for video in videos:
        obj_name = osp.basename(video).split('.')[0].split('_')[2]
        if obj_name not in EXCLUDE_OBJECTS:
            videos_filter.append(video)
        else:
            print(f'filtering out video {video} with obj name {obj_name}')
    videos = videos_filter

    if args.index is not None:
        chunk_size = len(videos) // 100 + 1 
        videos = videos[args.index * chunk_size:(args.index + 1) * chunk_size]
        print(f'processing {len(videos)} videos, first video: {videos[0]}, last video: {videos[-1]}') # around 14s/video of 64 frames 
    print(f"In total, processing {len(videos)} videos")
    for video in tqdm(videos):
        args.video = video
        video_prefix = osp.basename(video).split('.')[0]
        outfile = osp.join(args.outpath, f'{video_prefix}_params.pkl')
        if osp.isfile(outfile) and not args.redo:
            print(f'{outfile} already exists, skipping...')
            continue
        runner = ViewSpecificNLFRunner(args) # need to reinitialize the video loader 
        runner.run(args, model)
    print('all done')

if __name__ == '__main__':
    main()

