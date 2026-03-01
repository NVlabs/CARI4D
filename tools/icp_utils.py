# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation


def reg_icp(source, target, voxel_size, init=np.eye(4), method='point2point', with_scale=False):
    """
    iterative icp in 3 steps
    """
    assert method in ['point2plane', 'point2point']
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane(with_scaling=with_scale) if method == 'point2plane' else o3d.pipelines.registration.TransformationEstimationPointToPoint()
    # 3 step icp: from coarse to fine
    voxel_radius = [voxel_size*8, voxel_size*4, voxel_size]
    max_iter = [50, 30, 20]
    current_trans = init.copy()
    for it, radius in zip(max_iter, voxel_radius):
        target_down = target.voxel_down_sample(voxel_size=radius)
        source_down = source.voxel_down_sample(voxel_size=radius)
        if method == 'point2plane':
            target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
            source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        # increase correspondence threshold can improve!
        result_icp = o3d.pipelines.registration.registration_icp(
            source_down, target_down, radius*4, current_trans,
            estimation,
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                              relative_rmse=1e-6,
                                                              max_iteration=it)) # segmentation fault!
        if result_icp.fitness > 0.1:
            current_trans = result_icp.transformation
    return result_icp


def translation_only_icp(src, tgt, R_fixed=np.eye(3), voxel_size=0.005, 
                         max_iter=30, tol=0.001, max_iters=[15, 15, 15]):
    # Work on copies
    src_work = o3d.geometry.PointCloud(src)
    tgt_work = o3d.geometry.PointCloud(tgt)

    # Apply (fixed) rotation once
    src_work.rotate(R_fixed, center=(0,0,0))

    total_t = np.zeros(3)

    voxel_radius = [voxel_size*8, voxel_size*4, voxel_size]

    for it, radius in zip(max_iters, voxel_radius):
        src_down = src_work.voxel_down_sample(voxel_size=voxel_size)
        tgt_down = tgt_work.voxel_down_sample(voxel_size=voxel_size)

        dev = o3d.core.Device("CUDA:0") # cuda is much faster than cpu 
        dtype = o3d.core.Dtype.Float32
        src_pts = o3d.core.Tensor(np.asarray(src_down.points), dtype=dtype, device=dev)          # (N,3)
        tgt_pts = o3d.core.Tensor(np.asarray(tgt_down.points), dtype=dtype, device=dev)          # (M,3)

        # build NNS on target (once)
        nns = o3d.core.nns.NearestNeighborSearch(tgt_pts)
        nns.knn_index()
        thr2 = (radius * 2) ** 2

        total_t_it = np.zeros(3)

        for i in range(it):
            # batched 1-NN: returns (indices[N,1], dists[N,1])
            idx, d2 = nns.knn_search(src_pts, 1)

            # inlier mask by squared distance
            mask = d2.reshape((-1,)) < thr2
            if mask.cpu().numpy().sum() == 0:
                break

            # gather matched target points
            idx = idx.reshape((-1,))
            tgt_matched = tgt_pts[idx]                          # (N,3) gathered
            # residuals (tgt - src) for inliers only
            res = tgt_matched[mask] - src_pts[mask]            # (K,3)

            # mean translation update
            delta_t = res.mean(0)
            delta_t[:2] = 0 # optimize only z 
            if float(np.linalg.norm(delta_t.cpu().numpy())) < tol:
                break

            # apply update
            src_pts = src_pts + delta_t
            total_t = total_t + delta_t.cpu().numpy()
            total_t_it += delta_t.cpu().numpy()
        src_work.translate(total_t_it) # add translation from this iteration 
        

    # Build final 4x4
    T = np.eye(4)
    T[:3,:3] = R_fixed
    T[:3,3]  = total_t
    return T

def translation_only_icp_torch(src, tgt, R_fixed=np.eye(3), voxel_size=0.005, 
                         max_iter=30, tol=0.001, max_iters=[15, 15, 15]):
    """
    Same behavior as original, but replaces Open3D NNS with PyTorch3D knn_points.
    Keeps Open3D voxel downsampling.
    """
    import torch
    from pytorch3d.ops import knn_points

    # Work on copies
    src_work = o3d.geometry.PointCloud(src)
    tgt_work = o3d.geometry.PointCloud(tgt)

    # Apply (fixed) rotation once
    src_work.rotate(R_fixed, center=(0, 0, 0))

    total_t = np.zeros(3)
    voxel_radius = [voxel_size * 8, voxel_size * 4, voxel_size]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for it, radius in zip(max_iters, voxel_radius):
        # Keep Open3D voxel downsampling
        src_down = src_work.voxel_down_sample(voxel_size=voxel_size)
        tgt_down = tgt_work.voxel_down_sample(voxel_size=voxel_size)

        # Convert to torch tensors
        src_pts = torch.from_numpy(np.asarray(src_down.points)).float().to(device)  # (N,3)
        tgt_pts = torch.from_numpy(np.asarray(tgt_down.points)).float().to(device)  # (M,3)
        if src_pts.numel() == 0 or tgt_pts.numel() == 0:
            break

        thr2 = float((radius * 2) ** 2)
        total_t_it = np.zeros(3)

        for _ in range(it):
            # PyTorch3D batched knn (k=1)
            # shapes: dists (1,N,1), idx (1,N,1)
            dists, idx, _ = knn_points(src_pts.unsqueeze(0), tgt_pts.unsqueeze(0), K=1)
            d2 = dists[0, :, 0]  # (N,)
            nn_idx = idx[0, :, 0].long()  # (N,)

            # inlier mask by squared distance
            mask = d2 < thr2
            if mask.sum().item() == 0:
                break

            # gather matched target points
            tgt_matched = tgt_pts[nn_idx]  # (N,3)

            # residuals (tgt - src) for inliers only
            res = tgt_matched[mask] - src_pts[mask]  # (K,3)

            # mean translation update; constrain to z only
            delta_t = res.mean(dim=0)
            delta_t[:2] = 0.0

            if torch.norm(delta_t).item() < tol:
                break

            # apply update on the working src points tensor
            src_pts = src_pts + delta_t

            # accumulate totals for outputs and to advance full-res cloud
            delta_np = delta_t.detach().cpu().numpy()
            total_t = total_t + delta_np
            total_t_it += delta_np

        # advance the high-res source cloud for the next pyramid level
        src_work.translate(total_t_it)

    # Build final 4x4
    T = np.eye(4)
    T[:3, :3] = R_fixed
    T[:3, 3] = total_t
    return T

        

def pose2mat(angle, trans):
    """
    axis angle and trans to 4x4 transformation matrix
    """
    rot = Rotation.from_rotvec(angle)
    mat = np.eye(4)
    mat[:3, :3] = rot.as_matrix()
    mat[:3, 3] = trans
    return mat


def mat2pose(mat):
    """

    """
    angle = Rotation.from_matrix(mat[:3, :3]).as_rotvec()
    trans = mat[:3, 3]
    return angle, trans


def get_bbox_mask(bmax, bmin, points):
    """
    find the mask for the given points with bbox mask
    """
    assert np.all(bmax>=bmin), 'given bbox invalid: {}->{}'.format(bmin, bmax)
    bmask = (points[:, 0] <= bmax[0]) & (points[:, 0] >= bmin[0]) & \
            (points[:, 1] <= bmax[1]) & (points[:, 1] >= bmin[1]) & \
            (points[:, 2] <= bmax[2]) & (points[:, 2] >= bmin[2])
    return bmask


def get_full_pc(idx, kin_transform, reader, color=False, mask_target=None, filter3d=False):
    """
    get multiview pc of one frame
    """
    from prep_data.pc_filter import PCloudsFilter
    depths = reader.get_depth_images(idx, reader.kids)
    points = []
    for kid, dmap in enumerate(depths):
        if mask_target is not None:
            mask = reader.get_mask(idx, kid, mask_target)
            if mask is not None:
                dmap[~mask] = 0
        p = kin_transform.intrinsics[kid].dmap2pc(dmap)
        if len(p)>10:
            pl = kin_transform.local2world(p, kid)
            points.append(pl)
    if len(points) == 0:
        # rerun without mask
        print("Rerun get full pc for frame {} without any mask".format(reader.get_frame_folder(idx)))
        return get_full_pc(idx, kin_transform, reader)
    points = np.concatenate(points, 0)
    if filter3d and mask_target is not None: # apply 3d bbox filter
        points = PCloudsFilter.filter_pc_only(points, reader.seq_info.get_obj_name())
    return points

def is_symmetry(obj_name):
    """
    whether the given object is symmetric or not
    """
    symm_objects = ['basketball', 'boxlarge', 'boxlong', 'boxmedium',
           'boxsmall', 'boxtiny', 'stool', 'suitcase',
                    'toolbox', 'trashbin', 'yogaball', 'yogamat']
    return obj_name in symm_objects


def is_flipped(verts1, verts2, thres=0.5):
    """
    given two set of vertices, verify the mesh is flipped or not
    if one axis is rotated by more than 60 degree (dot product smaller than 0.5), then it is recognized as flipped
    """
    from sklearn.decomposition import PCA
    pca1 = PCA(n_components=3)
    pca1.fit(verts1)
    pca2 = PCA(n_components=3)
    pca2.fit(verts2)
    axis1 = pca1.components_
    axis2 = pca2.components_

    flipped = False
    for idx in range(3):
        vec1 = axis1[:, idx]
        vec2 = axis2[:, idx]
        if np.dot(vec2, vec1) < thres:
            flipped = True
            break
    return flipped


def test_flip():
    """

    """
    from psbody.mesh import Mesh

    seq = "/BS/xxie-4/work/kindata/Oct02_xiaomei_monitor_move"

    # two axis are flipped, dot products: -0.37510005329622115
    # 0.9564728587541083
    # -0.3333265648462075

    # rotated by ~45 degree: 0.40258066383463875
    # 0.9723546722251455
    # 0.4302157842497676
    frame1 = "t0003.000"
    frame2 = "t0009.000"

    # flipped: -0.9424428073050919
    # 0.93135520461439
    # -0.943012307103883
    frame1 = "t0003.000"
    frame2 = "t0015.000"

    m1, m2 = Mesh(), Mesh()
    m1.load_from_file(seq + "/" + frame1 + "/monitor/fit01/monitor_fit.ply")
    m2.load_from_file(seq + "/" + frame2 + "/monitor/fit01/monitor_fit.ply")

    is_flipped(m1.v, m2.v)


if __name__ == '__main__':
    from argparse import ArgumentParser
    test_flip()
