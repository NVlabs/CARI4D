# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""
Given a reconstructed textured mesh, estimate the scale using foundation pose
"""
import sys, os
sys.path.append(os.getcwd())
import torch, os, json 
import trimesh
import numpy as np
from tqdm import tqdm
import cv2 
import os.path as osp
from pytorch3d.io import load_objs_as_meshes, load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesUV

from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
import nvdiffrast.torch as dr
import Utils
from tools.chamfer_dist_np import chamfer_distance


def get_specific_frame(video_prefix, frame_time, kid=1):
    from behave_data.video_reader import ColorDepthController
    ctrl = ColorDepthController(video_prefix, kid)
    color, depth = ctrl.get_closest_frame(float(frame_time[1:]))
    return color, depth

def fp_scale_estimator(args):
    "use FP to estimate the rough scale "
    # init FP
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()

    outdir = args.outdir
    hy_file = args.mesh_file
    
    # Load rgb, depth and mask    
    mask_file = args.mask_file
    color = cv2.imread(args.rgb_file)[:, :, ::-1].copy()
    depth = cv2.imread(args.depth_file, cv2.IMREAD_ANYDEPTH)
    scale_ratio = 1  # do not do any resizing

    mask_o = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    mask_o = cv2.resize(mask_o, (mask_o.shape[1] // scale_ratio, mask_o.shape[0] // scale_ratio), cv2.INTER_NEAREST)

    # load camera_K
    camera_K = np.loadtxt(args.cam_file).reshape((3, 3))
    camera_K[:2] /= scale_ratio

    # load raw hy3d mesh
    estimate_metric_scale(scorer, refiner, glctx, outdir, hy_file, color, depth, mask_o, camera_K)

def estimate_metric_scale(scorer, refiner, glctx, outdir, hy_file, color, depth, mask_o, camera_K):
    """
    estimate metric scale of the object
    input params:
    scorer: the score predictor for foundation pose
    refiner: the pose refiner for foundation pose
    glctx: the nvdiff rasterization context
    outdir: the output directory to save results
    hy_file: the input hy3d file path, raw mesh in normalized scale. 
    
    """
    fname = osp.basename(hy_file)

    # save color and depth to output dir
    name = osp.splitext(osp.basename(hy_file))[0]
    outdir_i = f'{outdir}/{name}'
    os.makedirs(outdir_i, exist_ok=True)
    mesh_out = osp.join(outdir_i, osp.basename(hy_file).replace('_rgba.obj', '_align.obj'))
    if osp.isfile(mesh_out):
        print(mesh_out, 'already exists, exiting...')
        return
    cv2.imwrite(f'{outdir_i}/rgb.png', color[:, :, ::-1])
    cv2.imwrite(f'{outdir_i}/depth.png', depth)

    # perform depth filtering
    depth = torch.as_tensor(depth / 1000., device='cuda', dtype=torch.float)
    depth = Utils.erode_depth(depth, radius=2, device='cuda')
    depth = Utils.bilateral_filter_depth(depth, radius=2, device='cuda')
    depth = depth.cpu().numpy()

    mesh_raw = trimesh.load_mesh(hy_file, process=False)
    xyz_map = Utils.depth2xyzmap(depth, camera_K)

    debug_dir = f'output/debug-scale/'
    os.makedirs(debug_dir, exist_ok=True)
    # dilate the mask
    obj_pts = xyz_map[mask_o > 127].reshape((-1, 3))
    obj_colors = color[mask_o > 127].reshape((-1, 3))
    trimesh.PointCloud(obj_pts, obj_colors / 255.).export(f'{debug_dir}/{fname.replace("_rgba.obj", f"_obj_pts.ply")}')

    for i in range(3): # coarse to fine and then save, in total 3 iterations.
        outfile = f'{outdir_i}/{fname.replace("_rgba.obj", "_fp-res.json")}'
        if osp.isfile(outfile):
            # load results and init the range from best K
            res = json.load(open(outfile))
            top3_scales = np.array(res['chamfs'])[:, 0][np.argsort(np.array(res['chamfs'])[:, 1])[:3]]
            scale_candidates = np.linspace(np.min(top3_scales)*0.8, np.max(top3_scales)*1.2, 8)
            # add top3 back
            scale_candidates = np.concatenate([scale_candidates, top3_scales])
            outfile = outfile.replace('_fp-res.json', '_fp-res-refine.json')
            if osp.isfile(outfile):
                # now load the result and save rescaled mesh
                res = json.load(open(outfile))
                best_scale = res['best_scale']

                os.makedirs(osp.dirname(mesh_out), exist_ok=True)
                assert mesh_out != hy_file
                # use pytorch3d to load and save
                torch.set_default_tensor_type('torch.FloatTensor') # avoid loading error
                mesh_hy_orig: Meshes = load_objs_as_meshes([hy_file], device='cpu')[0]
                tex: TexturesUV = mesh_hy_orig.textures
                save_obj(mesh_out, mesh_hy_orig.verts_packed()*best_scale,
                        mesh_hy_orig.faces_packed(),
                        normals=mesh_hy_orig.verts_normals_packed(),
                        faces_normals_idx=mesh_hy_orig.faces_normals_packed(),
                        verts_uvs=tex.verts_uvs_padded()[0],
                        faces_uvs=tex.faces_uvs_padded()[0],
                        texture_map=tex.maps_padded()[0])
                print(mesh_out, 'saved with scale', best_scale)
                break
        else:
            scale_candidates = np.linspace(0.03, 3.0, 30)
        print(f'{osp.basename(hy_file)} scale candidates: {scale_candidates}')

        # give a range of scales and compute the best scale
        samples = mesh_raw.sample(10000)
        chamfs = []
        pose_results = []
        for scale in tqdm(scale_candidates):
            mesh_t = mesh_raw.copy()
            mesh_t.vertices = mesh_t.vertices * scale
            est = FoundationPose(model_pts=mesh_t.vertices, model_normals=mesh_t.vertex_normals, mesh=mesh_t, scorer=scorer,
                             refiner=refiner, debug_dir=debug_dir, debug=0, glctx=glctx)
            score_vis = f'{debug_dir}/scale_{scale:.3f}_score.png'
            refine_vis = f'{debug_dir}/scale_{scale:.3f}_refine.png'
            pose = est.register(K=camera_K, rgb=color, depth=depth, ob_mask=mask_o.astype(bool),
                                iteration=5,
                                vis_score_path=score_vis, vis_refine_path=refine_vis
                                ) # this is the pose that is w.r.t to mesh_t
            samples_t = samples * scale
            samples_cam = np.matmul(samples_t, pose[:3, :3].T) + pose[:3, 3]
            mesh_verts = np.matmul(mesh_t.vertices, pose[:3, :3].T) + pose[:3, 3] 
            depth_rend, _ = Utils.nvdiff_color_depth_render([camera_K], est.glctx, est.mesh_tensors, mask_o.shape[:2], torch.from_numpy(mesh_verts).to('cuda').float()[None],
                                                      depth_only=True)
            mask_o_rend = depth_rend.cpu().numpy()[0] > 0 
            mask_o_input = mask_o > 127 # for computing iou 
            iou = (np.sum(mask_o_rend & mask_o_input) + 1e-6) / (np.sum(mask_o_rend | mask_o_input) + 1e-6)
            # save the visualizations 
            comb = np.concatenate([mask_o_input, mask_o_rend], axis=1)

            pose_results.append(pose)
            cd = chamfer_distance(obj_pts, samples_cam)
            print(f'scale: {scale}, chamfer: {cd:.4f}, scaled chamfer: {cd/iou:.4f}')
            cd = cd/iou # scale the chamfer distance by the iou 
            chamfs.append([scale, cd])
            trimesh.PointCloud(samples_cam).export(f'{debug_dir}/scale_{scale:.3f}_obj.ply')
            # also save the original mesh
            mesh_copy = mesh_raw.copy()
            mesh_copy.vertices = mesh_copy.vertices * scale
            mesh_copy.vertices = np.matmul(mesh_copy.vertices, pose[:3, :3].T) + pose[:3, 3]
            mesh_copy.export(f'{debug_dir}/scale_{scale:.3f}_orig.obj')
        # pick the top2 scales
        best_idx = np.argmin(np.array(chamfs)[:, 1])
        best_scale = scale_candidates[best_idx]
        print(f'best scale: {best_scale}')
        mesh_best = mesh_raw.copy()
        mesh_best.vertices = mesh_best.vertices * best_scale
        # apply the best pose as well
        pose_best = pose_results[best_idx]
        mesh_best.vertices = np.matmul(mesh_best.vertices, pose_best[:3, :3].T) + pose_best[:3, 3]
        # get fname of this hy3d file
        trimesh.PointCloud(mesh_best.vertices).export(f'{outdir}/{fname.replace("_rgba.obj", f"_best_{best_scale:.3f}_obj.ply")}')
        # save scales and chamf scores as json file
        # pick the top 2 scales with lowest chamfer distance
        top2_scales = scale_candidates[np.argsort(chamfs)[:2]]
        res = {
            "scales": scale_candidates.tolist(),
            "chamfs": chamfs,
            "best_scale": best_scale,
            "top2_scales": top2_scales.tolist(),
            "poses": [x.tolist() for x in pose_results],
        }
        json.dump(res, open(outfile, 'w'), indent=2)



if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--mesh_file', default='data/cari4d-demo/meshes-raw/gas/gas_rgba.obj')
    parser.add_argument('--rgb_file', default='data/cari4d-demo/meshes-raw/gas/rgb.png')
    parser.add_argument('--depth_file', default='data/cari4d-demo/meshes-raw/gas/depth.png')
    parser.add_argument('--mask_file', default='data/cari4d-demo/meshes-raw/gas/mask.png')
    parser.add_argument('--cam_file', default='data/cari4d-demo/meshes-raw/gas/K.txt')
    parser.add_argument('-o', '--outdir', default='data/cari4d-demo/meshes')

    args = parser.parse_args()

    fp_scale_estimator(args)


