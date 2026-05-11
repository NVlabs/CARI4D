# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Replace NLF with SAM3D-body for human pose estimation.
Drop-in replacement for prep/run_nlf_sepK.py with identical output format.

Usage:
    python prep/run_sam3d_sepK.py -o data/cari4d-demo/videogen/nlf \
        --masks_root data/cari4d-demo/videogen/masks/ \
        --video <video_path> --wild_video
"""

import os, sys
import warnings

# IMPORTANT: Add cari4d project root BEFORE sam-3d-body to avoid `tools` module conflict
sys.path.insert(0, os.getcwd())

import h5py
import cv2
import torch
import numpy as np
from glob import glob
from tqdm import tqdm
from torch.utils.data import default_collate
import os.path as osp
import joblib

from behave_data.behave_video import BaseBehaveVideoData
from behave_data.const import _sub_gender, EXCLUDE_OBJECTS
from behave_data.utils import get_intrinsics_unified
from tools import img_utils
from lib_smpl import SMPL_MODEL_ROOT

# MHR→SMPLH conversion (import before sam3d to avoid conflicts)
from prep.mhr2smplh import transfer_via_barycentric_gpu
from smplfitter.pt import BodyModel, BodyFitter

# Now add sam-3d-body to path (after cari4d imports are resolved)
# Clone or symlink sam-3d-body into the project root:
#   git clone https://github.com/facebookresearch/sam-3d-body.git sam-3d-body
SAM3D_ROOT = osp.join(os.getcwd(), 'sam-3d-body')
sys.path.append(SAM3D_ROOT)

# SAM3D-body imports
from sam_3d_body import load_sam_3d_body
from sam_3d_body.data.transforms import (
    Compose,
    GetBBoxCenterScale,
    TopdownAffine,
    VisionTransformWrapper,
)
from sam_3d_body.data.utils.prepare_batch import NoCollate
from sam_3d_body.utils import recursive_to
from torchvision.transforms import ToTensor

# Defaults
SAM3D_CKPT = osp.join(SAM3D_ROOT, 'checkpoints/sam-3d-body-dinov3/model.ckpt')
MHR_PATH = osp.join(SAM3D_ROOT, 'checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt')
MAPPING_DIR = osp.join(os.getcwd(), 'data/assets')


def prepare_batch_multi_frame(frames, bboxes, transform, cam_int):
    """Batch multiple frames as 'persons' in SAM3D's person dimension.

    Args:
        frames: list of RGB numpy arrays (H, W, 3)
        bboxes: list of numpy arrays (4,) in xyxy format
        transform: SAM3D transform pipeline
        cam_int: torch.Tensor (1, 3, 3) camera intrinsics

    Returns:
        batch dict ready for model inference
    """
    data_list = []
    for img, bbox in zip(frames, bboxes):
        h, w = img.shape[:2]
        data_info = dict(img=img)
        data_info["bbox"] = bbox
        data_info["bbox_format"] = "xyxy"
        data_info["mask"] = np.zeros((h, w, 1), dtype=np.uint8)
        data_info["mask_score"] = np.array(0.0, dtype=np.float32)
        data_list.append(transform(data_info))

    batch = default_collate(data_list)
    for key in [
        "img", "img_size", "ori_img_size", "bbox_center",
        "bbox_scale", "bbox", "affine_trans", "mask", "mask_score",
    ]:
        if key in batch:
            batch[key] = batch[key].unsqueeze(0).float()
    if "mask" in batch:
        batch["mask"] = batch["mask"].unsqueeze(2)
    batch["person_valid"] = torch.ones((1, len(frames)))
    batch["cam_int"] = cam_int.to(batch["img"])
    batch["img_ori"] = [NoCollate(frames[0])]  # dummy, unused for body-only inference
    return batch


def load_mhr2smplh_mapping(mapping_dir, device='cuda'):
    """Load pre-built MHR→SMPLH barycentric mapping."""
    mapping = np.load(osp.join(mapping_dir, 'mhr2smplh_mapping.npz'))

    tri_ids = torch.tensor(mapping['triangle_ids'], dtype=torch.long, device=device)
    bary_coords = torch.tensor(mapping['baryc_coords'], dtype=torch.float32, device=device)
    mhr_faces = torch.tensor(mapping['mhr_faces'], dtype=torch.long, device=device)

    return tri_ids, bary_coords, mhr_faces


def mhr_verts_to_smplh_params(all_verts, tri_ids, bary_coords, mhr_faces,
                               fitter, batch_size=64):
    """Convert MHR vertices in camera space to SMPLH parameters.

    Args:
        all_verts: torch.Tensor (N, 18439, 3) MHR vertices in camera space (meters)
        tri_ids, bary_coords, mhr_faces: barycentric mapping tensors (on GPU)
        fitter: smplfitter BodyFitter
        batch_size: batch size for GPU processing

    Returns:
        poses (N, 156), betas (N, 10), transls (N, 3) as numpy arrays
    """
    device = tri_ids.device
    N = all_verts.shape[0]

    # Step 1: Barycentric transfer (MHR → SMPLH topology)
    target_smplh = []
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        batch = all_verts[s:e].to(device)
        transferred = transfer_via_barycentric_gpu(batch, mhr_faces, tri_ids, bary_coords)
        target_smplh.append(transferred.cpu())
    target_smplh = torch.cat(target_smplh)  # (N, 6890, 3)

    # Step 2: Fit SMPLH parameters (directly in camera space, no centering needed)
    fit_batch = min(batch_size * 4, 4096)
    all_rotvecs, all_betas, all_trans = [], [], []
    with torch.no_grad():
        for s in range(0, N, fit_batch):
            e = min(s + fit_batch, N)
            fit_res = fitter.fit(
                target_smplh[s:e].to(device),
                num_iter=5,
                beta_regularizer=0.001,
                share_beta=True,
                requested_keys=['pose_rotvecs', 'shape_betas', 'trans'],
            )
            all_rotvecs.append(fit_res['pose_rotvecs'].cpu())
            all_betas.append(fit_res['shape_betas'].cpu())
            all_trans.append(fit_res['trans'].cpu())

    poses = torch.cat(all_rotvecs).numpy()     # (N, 156)
    betas = torch.cat(all_betas).numpy()       # (N, 10)
    transls = torch.cat(all_trans).numpy()     # (N, 3)

    return poses, betas, transls


class ViewSpecificSAM3DRunner(BaseBehaveVideoData):
    @torch.no_grad()
    def run(self, args, model=None, model_cfg=None, transform=None,
            mapping_data=None, fitter=None):
        outfile = osp.join(args.outpath, f'{self.video_prefix}_params.pkl')
        if osp.isfile(outfile) and not args.redo:
            print(f'{outfile} already exists, skipping...all done')
            return

        device = 'cuda'
        gender = _sub_gender[self.video_prefix.split('_')[1]]

        # Camera intrinsics (per view)
        if args.wild_video:
            K_all = np.array([self.camera_K])
        else:
            K_all = [get_intrinsics_unified(self.args.data_source, self.video_prefix, kid, self.args.wild_video)
                     for kid in self.kids]
            K_all = np.array(K_all)

        # Unpack mapping data
        tri_ids, bary_coords, mhr_faces = mapping_data

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

            # Camera intrinsics for this view
            cam_int = torch.from_numpy(K_all[kid_idx]).float()[None].to(device)  # (1, 3, 3)

            chunk_size = args.chunk_size
            for i in tqdm(range(0, len(self.times), chunk_size),
                          desc=f"SAM3D {self.video_prefix}/k{kid}"):
                images_all, masks_all, frames_chunk = [], [], []
                for j in range(i, min(i + chunk_size, len(self.times))):
                    t = self.times[j]
                    frame_time = 't{:08.3f}'.format(t) if not args.wild_video else f'{t:06d}'
                    if args.data_source not in ['behave']:
                        frame_time = f'{t:06d}'
                    mask_name = f'{self.video_prefix}/{frame_time}-k{kid}.person_mask.png'
                    try:
                        masks = tars[kid_idx][mask_name][:].astype(np.uint8) * 255
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

                # Extract bboxes from masks (xyxy format for SAM3D)
                bboxes = []
                for iii, (img, mask) in enumerate(zip(images_all, masks_all)):
                    i_curr = iii
                    while np.sum(mask > 127) < 20:
                        i_curr -= 1
                        if i_curr < 0:
                            break
                        mask = masks_all[i_curr]
                        print(f"warning: no human found in this frame, revert to previous mask, index {iii}->{i_curr}")
                    bmin, bmax = img_utils.masks2bbox([mask])
                    # xyxy format for SAM3D
                    bbox_xyxy = np.array([bmin[0], bmin[1], bmax[0], bmax[1]], dtype=np.float32)
                    bboxes.append(bbox_xyxy)

                # Batch inference: treat N frames as N "persons"
                batch = prepare_batch_multi_frame(images_all, bboxes, transform, cam_int)
                batch = recursive_to(batch, device)
                model._initialize_batch(batch)
                pose_output = model.run_inference(
                    images_all[0], batch,
                    inference_type="body",
                    transform_hand=None,
                    thresh_wrist_angle=1.4,
                )

                # Extract pred_vertices for all frames in chunk
                out = pose_output["mhr"]
                out = recursive_to(out, "cpu")
                out = recursive_to(out, "numpy")

                # Move pred_vertices to camera space by adding pred_cam_t
                # pred_vertices are in body-local space; pred_cam_t provides the camera translation
                n_chunk = len(frames_chunk)
                all_pred_verts = []
                for idx in range(n_chunk):
                    v = out["pred_vertices"][idx]       # (18439, 3) body-local
                    cam_t = out["pred_cam_t"][idx]      # (3,)
                    all_pred_verts.append(v + cam_t)    # camera space

                # Check validity (z < 8m), fallback to neighbors
                valid = [v[:, 2].mean() < 8.0 for v in all_pred_verts]

                ordered_verts = []
                for idx in range(n_chunk):
                    if valid[idx]:
                        ordered_verts.append(torch.tensor(all_pred_verts[idx], dtype=torch.float32))
                    else:
                        # Search ±10 neighbors for a valid prediction
                        found = False
                        for offset in range(1, 11):
                            for neighbor in [idx - offset, idx + offset]:
                                if 0 <= neighbor < n_chunk and valid[neighbor]:
                                    ordered_verts.append(torch.tensor(all_pred_verts[neighbor], dtype=torch.float32))
                                    print(f'  fallback: {frames_chunk[idx]} -> {frames_chunk[neighbor]}')
                                    found = True
                                    break
                            if found:
                                break
                        if not found:
                            ordered_verts.append(torch.tensor(all_pred_verts[idx], dtype=torch.float32))
                            print(f'  no valid neighbor for {frames_chunk[idx]}, using original')

                all_verts_chunk = torch.stack(ordered_verts)  # (chunk, 18439, 3)

                # MHR→SMPLH conversion
                poses, betas, transls = mhr_verts_to_smplh_params(
                    all_verts_chunk, tri_ids, bary_coords, mhr_faces,
                    fitter, batch_size=64,
                )

                nlf_data[kid]['poses'].append(poses)
                nlf_data[kid]['betas'].append(betas)
                nlf_data[kid]['transls'].append(transls)
                nlf_data[kid]['frames'].extend(frames_chunk)

                assert len(poses) == len(frames_chunk), \
                    f'incomplete prediction {poses.shape} on frame {frames_chunk[-1]}!'

                if i == 0:
                    print("Translation:", transls[0], 'please check if this is reasonable!')

            nlf_data[kid]['poses'] = np.concatenate(nlf_data[kid]['poses'], 0)
            nlf_data[kid]['betas'] = np.concatenate(nlf_data[kid]['betas'], 0)
            nlf_data[kid]['transls'] = np.concatenate(nlf_data[kid]['transls'], 0)

        for tar in tars:
            tar.close()

        # Stack all views into shape (N, K, D)
        nlf_data_all = {}
        for k in nlf_data[kids[0]].keys():
            if k == 'frames':
                continue
            nlf_data_all[k] = np.stack([nlf_data[kid][k] for kid in kids], 1)
        nlf_data_all['frames'] = nlf_data[kids[0]]['frames']
        nlf_data_all['gender'] = gender
        nlf_data_all['kids'] = kids
        os.makedirs(osp.dirname(outfile), exist_ok=True)
        joblib.dump(nlf_data_all, outfile)
        print('all done, saved to', outfile)


def main():
    parser = BaseBehaveVideoData.get_parser()
    parser.add_argument('--sam3d_ckpt', default=SAM3D_CKPT,
                        help='Path to SAM3D-body model checkpoint')
    parser.add_argument('--mhr_path', default=MHR_PATH,
                        help='Path to MHR model asset')
    parser.add_argument('--mapping_dir', default=MAPPING_DIR,
                        help='Directory with pre-built mhr2smplh mapping files')
    parser.add_argument('--chunk_size', type=int, default=16,
                        help='Number of frames per batch (adjust for GPU memory)')
    args = parser.parse_args()
    args.nodepth = True

    # Load SAM3D-body model (once)
    print("Loading SAM3D-body model...")
    device = torch.device('cuda')
    model, model_cfg = load_sam_3d_body(args.sam3d_ckpt, device=device, mhr_path=args.mhr_path)

    # Build transform (same as SAM3DBodyEstimator.__init__)
    transform = Compose([
        GetBBoxCenterScale(),
        TopdownAffine(input_size=model_cfg.MODEL.IMAGE_SIZE, use_udp=False),
        VisionTransformWrapper(ToTensor()),
    ])

    # Load MHR→SMPLH mapping (once)
    print(f"Loading MHR→SMPLH mapping from {args.mapping_dir}...")
    tri_ids, bary_coords, mhr_faces = load_mhr2smplh_mapping(args.mapping_dir)
    mapping_data = (tri_ids, bary_coords, mhr_faces)

    # Collect videos
    if osp.isfile(args.video):
        videos = [args.video]
    else:
        videos = sorted(glob(args.video))
    print(f'In total {len(videos)} videos')

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
        print(f'processing {len(videos)} videos, first: {videos[0]}, last: {videos[-1]}')

    print(f"Processing {len(videos)} videos")
    for video in tqdm(videos):
        args.video = video
        video_prefix = osp.basename(video).split('.')[0]
        outfile = osp.join(args.outpath, f'{video_prefix}_params.pkl')
        if osp.isfile(outfile) and not args.redo:
            print(f'{outfile} already exists, skipping...')
            continue

        # Create fitter with correct gender for this video
        gender = _sub_gender[video_prefix.split('_')[1]]
        fitter = BodyFitter(BodyModel('smplh', gender, model_root=SMPL_MODEL_ROOT).to('cuda')).to('cuda')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitter = torch.jit.script(fitter)

        runner = ViewSpecificSAM3DRunner(args)
        runner.run(args, model=model, model_cfg=model_cfg, transform=transform,
                   mapping_data=mapping_data, fitter=fitter)

    print('all done')


if __name__ == '__main__':
    main()
