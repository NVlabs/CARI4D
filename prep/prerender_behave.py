# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
script to pre-render behave objects for training
for every frame, render RGBA image,

"""
import glob
import pickle
import sys, os

import imageio
import h5py
import joblib
import os.path as osp

sys.path.append(os.getcwd())
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import h5py, hdf5plugin
import kornia

sys.path.append(os.getcwd())
import json, torch, trimesh
from scipy.spatial.transform import Rotation as R
from behave_data.utils import load_template, load_template_orig, init_video_controllers, load_kinect_poses_back
from tools.offscreen_renderer import ModelRendererOffscreen
import Utils


def random_direction():
  '''https://stackoverflow.com/questions/33976911/generate-a-random-sample-of-points-distributed-on-the-surface-of-a-unit-sphere
  '''
  vec = np.random.randn(3).reshape(3)
  vec /= np.linalg.norm(vec)
  return vec

class BehaveRenderer:
    def __init__(self, args):
        self.args = args
        self.rend_size = args.rend_size # for the crop
        if not args.wild_video:
            if args.data_source == 'behave':
                fx, fy = 979.7844, 979.840  # for original kinect coordinate system
                cx, cy = 1018.952, 779.486
            elif args.data_source == 'intercap':
                from behave_data.const import ICAP_CENTERs
                from behave_data.const import ICAP_FOCALs
                fx, fy = ICAP_FOCALs[0, 0], ICAP_FOCALs[0, 1]
                cx, cy = ICAP_CENTERs[0, 0], ICAP_CENTERs[0, 1]
            elif args.data_source == 'hodome':
                from behave_data.const import get_camera_K_hodome, HODOME_VIEW_IDS
                K = get_camera_K_hodome(osp.basename(self.args.video), HODOME_VIEW_IDS[1]) # simply take one KID
                fx, fy = K[0,0], K[1,1]
                cx, cy = K[0,2], K[1,2]
            elif args.data_source == 'imhd':
                from behave_data.const import IMHD_VIEW_IDS, get_IMHD_camera_K
                K = get_IMHD_camera_K(osp.basename(self.args.video), IMHD_VIEW_IDS[0]) # simply take one KID, to be consistent with NLF, and FP 
                fx, fy = K[0,0], K[1,1]
                cx, cy = K[0,2], K[1,2]
            elif args.data_source == 'procigen':
                K = np.array([[979.784, 0, 1018.952],
                          [0, 979.840, 779.486],
                          [0, 0, 1]])
                K[:2] /= 2. 
                fx, fy = K[0,0], K[1,1]
                cx, cy = K[0,2], K[1,2]
            else:
                raise ValueError(f'Invalid data source: {args.data_source}')
        else:
            pkl_file = self.args.video.replace('.mp4', '.pkl')
            d = joblib.load(pkl_file)
            fx, fy = d['fx'], d['fy']
            cx, cy = d['cx'], d['cy']
        self.K_full = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1.]])
        print('Using Camera K:', self.K_full)
        self.focal = np.array([fx, fy])
        self.principal_point = np.array([cx, cy])

    @staticmethod
    def perturb_pose(theta_range, trans_range, ob_in_cam):
        '''
        @theta_range: radian
        '''
        trial = 0
        while 1:
            trial += 1
            axis = random_direction()
            theta = np.random.uniform(theta_range[0], theta_range[1])
            rot_noise = np.eye(4)
            rot_noise[:3, :3] = cv2.Rodrigues(axis * theta)[0]
            pose_perturbed = ob_in_cam @ rot_noise
            axis = random_direction()
            mag = np.random.uniform(0, trans_range)
            trans_noise = np.eye(4)
            trans_noise[:3, 3] = axis * mag
            pose_perturbed = trans_noise @ pose_perturbed
            return pose_perturbed

    def crop_color_dmap(self, bbox, color, depth, render_size, K_full=None):
        # compute depth xyz
        crop_ratio = 1.0  # this bbox is already scaled up, no need to scale a gain!
        bmin, bmax = bbox[:2], bbox[2:]
        crop_size = np.max(bmax - bmin) * crop_ratio
        crop_center = (bmin + bmax) / 2
        rend_size = render_size[0]
        scale = rend_size / crop_size
        top_left = crop_center - crop_size / 2
        bottom_right = crop_center + crop_size / 2
        left, right, top, bottom = torch.tensor([top_left[0], ]), torch.tensor(
            [bottom_right[0], ]), torch.tensor([top_left[1], ]), torch.tensor([bottom_right[1], ])
        tf_full = Utils.compute_tf_batch(left=left, right=right, top=top, bottom=bottom,
                                         out_size=render_size).cpu()
        dmap_xyz = Utils.depth2xyzmap(depth / 1000., self.K_full if K_full is None else K_full)
        valid = depth > 0
        dmap_xyz[~valid] = 0
        dmap_xyz = kornia.geometry.transform.warp_perspective(
            torch.as_tensor(dmap_xyz[None], device='cpu', dtype=torch.float).permute(0, 3, 1, 2),
            tf_full, dsize=render_size, mode='nearest', align_corners=False)[0].permute(1, 2, 0)
        rgbm = kornia.geometry.transform.warp_perspective(
            torch.as_tensor(color[None], device='cpu', dtype=torch.float).permute(0, 3, 1, 2),
            tf_full, dsize=render_size, mode='nearest', align_corners=False)[0].permute(1, 2, 0)  # (H, W, 5)
        return dmap_xyz, rgbm

    def prepare_video_mask_loader(self, args, kids, video_prefix):
        input_color = args.video
        args.nodepth = False
        controllers, _ = init_video_controllers(args, args.video, kids)

        # read h5 file
        h5_path = f'/home/xianghuix/datasets/behave/masks-h5-my/{video_prefix}_masks_k1.h5'
        print(f'loading masks from {h5_path}')
        h5_paths = [h5_path.replace('_k1.h5', f'_k{x}.h5') for x in range(len(controllers))]
        tar_mask = [h5py.File(h5_path, 'r') for h5_path in h5_paths]
        return controllers, tar_mask

    def render_seq(self, args):
        ""
        # Full size camera intrinsics
        H, W = 1536, 2048
        # load object template
        seq_name = args.seq
        mesh_diameter, temp_orig = self.load_obj_template(args, seq_name)

        # now load the packed object pose file
        packed_data = joblib.load(osp.join(args.packed_path, f'{seq_name}_GT-packed.pkl'))
        angles = packed_data['obj_angles']
        transl = packed_data['obj_trans']

        total_count = 0
        
        outfile = osp.join(args.output_dir, f'{seq_name}_render.h5')
        if osp.isfile(outfile):
            print(f'{outfile} already done, exiting...all done')
            return
        os.makedirs(osp.dirname(outfile), exist_ok=True)
        h5_writer = h5py.File(outfile, 'w')

        # some hyperparameters
        n_perturb = 1 # only one perturb possibility

        # prepare video data
        kids = [x for x in range(4)]

        trans_normalizer = np.array(args.trans_normalizer)
        rot_normalizer = args.rot_normalizer
        print(f"Normalization parameters: {trans_normalizer}, rot_normalizer: {rot_normalizer}")

        w2c_rots, w2c_trans = load_kinect_poses_back(osp.join(args.dataset_path, 'calibs', video_prefix.split('_')[0], 'config'), kids)

        # first delete the data, and then update
        h5_writer.create_dataset(seq_name + '_w2c', data=np.void(pickle.dumps({"rot": w2c_rots, "trans": w2c_trans,
                                                                               'mesh_diameter': mesh_diameter,
                                                                               'trans_normalizer': trans_normalizer,
                                                                               'rot_normalizer': rot_normalizer}, 0)))


        for i in tqdm(range(0, len(angles), args.skip)):
            frame_time = packed_data['frames'][i]
            rot = R.from_rotvec(angles[i]).as_matrix()
            mat = np.eye(4)
            mat[:3,:3] = rot
            mat[:3,3] = transl[i]

            frame_key = f'{seq_name}+{frame_time}'

            # add input RGB
            for k in range(4):
                # add world to camera transform
                w2c = np.eye(4)
                w2c[:3,:3] = w2c_rots[k]
                w2c[:3,3] = w2c_trans[k]
                pose_k = np.matmul(w2c, mat)

                for pi in range(n_perturb):
                    pose_perturbed = self.perturb_pose(theta_range=[0, np.deg2rad(rot_normalizer)],
                                                       trans_range=trans_normalizer, ob_in_cam=pose_k) # TODO: use an aabox pose!
                    data_key = frame_key + f'_k{k}_perturb_{pi}'
                    self.render_and_save(data_key, h5_writer, mesh_diameter, pose_perturbed, temp_orig)

        h5_writer.close()
        print('all done, file saved to', outfile)

    def render_and_save(self, data_key, h5_writer, mesh_diameter, pose_perturbed, temp_orig,
                        gt_to_perturb_pose=np.eye(4)):
        K_roi, bottom_right, top_left = self.compute_Kroi(mesh_diameter, pose_perturbed)
        print('bbox:', top_left, bottom_right, 'diameter:', mesh_diameter)
        # this is CPU only
        renderer = ModelRendererOffscreen(H=self.rend_size, W=self.rend_size, cam_K=K_roi)
        rgbA, depthA = renderer.render(temp_orig, pose_perturbed)

        h5_writer.create_dataset(data_key,
                                 data=np.void(pickle.dumps({
                                     'rgba': rgbA,
                                     'depth': depthA.astype(np.float16),
                                     'pose': self.process_perturb_pose(pose_perturbed, gt_to_perturb_pose),
                                     'K_roi': K_roi,
                                     'bbox': np.concatenate([top_left, bottom_right]),
                                 }, 0)))

    def load_obj_template(self, args, seq_name, ret_center=False):
        obj_name = seq_name.split('_')[2]
        if args.wild_video:
            # load from hy3d 
            files = sorted(glob.glob(f'/home/xianghuix/datasets/behave/selected-views/hy3d-aligned/{seq_name}*/*{obj_name}*_align.obj'))
            if len(files) == 0:
                raise ValueError(f'no aligned hy3d template found for {seq_name}')
            temp_orig = trimesh.load(files[0], process=False)
            temp_simplified = temp_orig.copy() # same as temp_orig 
        else:
            if args.data_source == 'behave':
                temp_orig = trimesh.load(osp.join(args.dataset_path, 'objects', f'{obj_name}/{obj_name}.obj'),
                                        process=False, maintain_order=True) # maintain_order is important to be compatible with pyt3d
                temp_simplified = load_template(obj_name, False, args.dataset_path)
            elif args.data_source == 'intercap':
                # for intercap, the center is still from the original dataset, to be consistent with FP! 
                # load from HY3D 
                file = f'/home/xianghuix/datasets/InterCap/objects/{obj_name[3:]}.ply'
                temp_orig = trimesh.load(file, process=False)
                temp_simplified = temp_orig.copy()
            elif args.data_source == 'hodome':
                # use the one that is used to compute center 
                obj_name = seq_name.split('_')[2]
                temp_orig = trimesh.load(f'/home/xianghuix/datasets/HODome/obj-newtex/{obj_name}/{obj_name}.obj', process=False)
                temp_simplified = temp_orig.copy()
            elif args.data_source == 'imhd':
                # load from simplified 
                obj_name = seq_name.split('_')[2]
                mesh_file = f'/home/xianghuix/datasets/IMHD2/hy3d-texgen-simp/{obj_name}/{obj_name}_simplified_transformed.obj'
                temp_simplified = trimesh.load(mesh_file, process=False)
                temp_orig = temp_simplified.copy()
            elif args.data_source == 'procigen':
                # create dummy mesh 
                temp = trimesh.Trimesh(vertices=np.zeros((10, 3)), faces=np.zeros((1, 3)).astype(np.int32))
                temp_orig = temp.copy()
                temp_simplified = temp.copy()
                return -1, temp_orig, np.zeros(3)
            else:
                raise ValueError(f'Invalid data source: {args.data_source}')

        center = np.mean(temp_simplified.vertices, 0)
        print("Subtracting orig_template with center:", center)
        temp_orig.vertices = temp_orig.vertices - center
        temp_simplified.vertices = temp_simplified.vertices - center
        pose = np.eye(4)
        # compute mesh diameter, why use this here???
        mesh_diameter = Utils.compute_mesh_diameter(mesh=temp_simplified)
        if ret_center:
            return mesh_diameter, temp_orig, center
        return mesh_diameter, temp_orig

    def compute_Kroi(self, mesh_diameter, pose_perturbed):
        bottom_right, top_left = self.compute_bbox_2d(self.K_full, pose_perturbed, mesh_diameter)
        # compute K_roi and render
        K_roi = self.Kroi_from_corners(bottom_right, top_left)
        return K_roi, bottom_right, top_left

    def Kroi_from_corners(self, bottom_right, top_left, focal=None, principal_point=None):
        crop_size = np.mean(bottom_right - top_left)  # this is a square anyways
        scale = self.rend_size / crop_size
        focal = self.focal if focal is None else focal
        principal_point = self.principal_point if principal_point is None else principal_point
        focal_roi = focal * scale
        principal_roi = (principal_point - top_left) * scale
        K_roi = np.array([[focal_roi[0], 0, principal_roi[0]],
                          [0, focal_roi[1], principal_roi[1]],
                          [0, 0, 1.]])
        return K_roi

    def process_perturb_pose(self, pose_perturb, gt_to_perturb_pose):
        return np.matmul(pose_perturb, gt_to_perturb_pose) # first convert from GT template, then to the perturb space


    def compute_bbox_2d(self, K, pose, mesh_diameter, crop_ratio = 1.2):
        """
        compute 2D square bbox to crop around the object
        """
        radius = mesh_diameter * crop_ratio / 2
        offsets = np.array([0, 0, 0,
                            radius, 0, 0,
                            -radius, 0, 0,
                            0, radius, 0,
                            0, -radius, 0]).reshape(-1, 3)  # this is an xy plane square
        pts = pose[:3, 3].reshape(1, 3) + offsets
        projected = (K @ pts.T).T
        uvs = projected[:, :2] / projected[:, 2:3]
        center = uvs[0]
        radius = np.abs(uvs - center.reshape(1, 2)).max()
        top_left = center - radius
        bottom_right = center + radius
        return bottom_right, top_left

    @staticmethod
    def get_parser():
        from argparse import ArgumentParser
        parser = ArgumentParser()
        parser.add_argument('-v', '--video', help='path to a video file')
        parser.add_argument('--packed_path', type=str, default='/home/xianghuix/datasets/behave/behave-packed/')
        parser.add_argument('-d', '--dataset_path', type=str, default='/home/xianghuix/datasets/behave/')
        parser.add_argument('-o', '--output_dir', type=str, default='/home/xianghuix/data/foundpose_train/behave')
        parser.add_argument('--h5_path', type=str, default='/home/xianghuix/data/behave_release/30fps-h5')
        parser.add_argument('-sn', '--shard_num', type=int, default=5, help='shard how many examples into one file')
        parser.add_argument('-nodepth', default=False, action='store_true',
                            help='save depth images or not, if not, will not load depth video')
        parser.add_argument('-tn', '--trans_normalizer', default=[0.02, 0.02, 0.05], nargs='+', type=float)
        parser.add_argument('-rn', '--rot_normalizer', default=20, type=float)
        parser.add_argument('-rs', '--rend_size', default=224, type=int)
        parser.add_argument('-s', '--skip', default=1, type=int)
        parser.add_argument('--redo', default=False, action='store_true')
        parser.add_argument('--camera_ids', default=[0, 1, 2, 3], nargs='+', type=int)
        parser.add_argument('--data_source', default='behave', choices=['behave', 'intercap', 'hodome', 'imhd', 'procigen'])

        parser.add_argument('-k', '--kid', default=1, type=int)

        parser.add_argument('-fs', '--start', default=0, type=int)
        parser.add_argument('-fe', '--end', default=None, type=int)

        parser.add_argument('--add_rgb', default=False, action='store_true')

        parser.add_argument('--nlf_path', default='/home/xianghuix/datasets/behave/nlf/', )
        parser.add_argument('--fp_root', default='/home/xianghuix/datasets/behave/fp')

        parser.add_argument('--debug', default=False, action='store_true')
        parser.add_argument('--wild_video', default=False, action='store_true')

        parser.add_argument('-i', '--index', default=None, type=int)

        parser.add_argument('--use_sel_view', default=False, action='store_true')
        return parser

if __name__ == '__main__':
    parser = BehaveRenderer.get_parser()

    args = parser.parse_args()

    video_prefix = osp.basename(args.video).split('.')[0]
    args.seq = video_prefix

    renderer = BehaveRenderer(args)

    try:
        renderer.render_seq(args)
    except Exception as e:
        import traceback
        traceback.print_exc()

