# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
a simple wrapper for pytorch3d rendering
Cite: BEHAVE: Dataset and Method for Tracking Human Object Interaction
"""
import numpy as np
import torch
from copy import deepcopy
# Data structures and functions for rendering
from pytorch3d.renderer import (
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
    PerspectiveCameras,
    PointsRasterizer,
    AlphaCompositor,
    PointsRasterizationSettings,
)
from pytorch3d.structures import Meshes, join_meshes_as_scene, Pointclouds

SMPL_OBJ_COLOR_LIST = [
        [0.65098039, 0.74117647, 0.85882353],  # SMPL
        [251 / 255.0, 128 / 255.0, 114 / 255.0],  # object
    ]


class MeshRendererWrapper:
    "a simple wrapper for the pytorch3d mesh renderer"
    def __init__(self, image_size=1200,
                 faces_per_pixel=1,
                 device='cuda:0',
                 blur_radius=0, lights=None,
                 materials=None, max_faces_per_bin=50000):
        self.image_size = image_size
        self.faces_per_pixel=faces_per_pixel
        self.max_faces_per_bin=max_faces_per_bin # prevent overflow, see https://github.com/facebookresearch/pytorch3d/issues/348
        self.blur_radius = blur_radius
        self.device = device
        self.lights=lights if lights is not None else PointLights(
            ((0.5, 0.5, 0.5),), ((0.5, 0.5, 0.5),), ((0.05, 0.05, 0.05),), ((0, -2, 0),), device
        )
        self.materials = materials
        self.renderer = self.setup_renderer()

    def setup_renderer(self):
        # for sillhouette rendering
        sigma = 1e-4
        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=self.blur_radius,
            faces_per_pixel=self.faces_per_pixel,
            clip_barycentric_coords=False,
            max_faces_per_bin=self.max_faces_per_bin
        )
        shader = SoftPhongShader(
            device=self.device,
            lights=self.lights,
            materials=self.materials)
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=raster_settings),
                shader=shader
        )
        return renderer

    def render(self, meshes, cameras, ret_mask=False, mode='rgb'):
        assert len(meshes.faces_list()) == 1, 'currently only support batch size =1 rendering!'
        images = self.renderer(meshes, cameras=cameras)
        if ret_mask or mode=='mask':
            mask = images[0, ..., 3].cpu().detach().numpy()
            return images[0, ..., :3].cpu().detach().numpy(), mask > 0
        return images[0, ..., :3].cpu().detach().numpy()


def get_kinect_camera(device='cuda:0', kid=1):
    R, T = torch.eye(3), torch.zeros(3)
    R[0, 0] = R[1, 1] = -1  # pytorch3d y-axis up, need to rotate to kinect coordinate
    R = R.unsqueeze(0)
    T = T.unsqueeze(0)
    assert kid in [0, 1, 2, 3], f'invalid kinect index {kid}!'
    if kid == 0:
        fx, fy = 976.212, 976.047
        cx, cy = 1017.958, 787.313
    elif kid == 1:
        fx, fy = 979.784, 979.840  # for original kinect coordinate system
        cx, cy = 1018.952, 779.486
    elif kid == 2:
        fx, fy = 974.899, 974.337
        cx, cy = 1018.747, 786.176
    else:
        fx, fy = 972.873, 972.790
        cx, cy = 1022.0565, 770.397
    color_w, color_h = 2048, 1536  # kinect color image size
    cam_center = torch.tensor((cx, cy), dtype=torch.float32).unsqueeze(0)
    focal_length = torch.tensor((fx, fy), dtype=torch.float32).unsqueeze(0)
    cam = PerspectiveCameras(focal_length=focal_length, principal_point=cam_center,
                             image_size=((color_w, color_h),),
                             device=device,
                             R=R, T=T,
                             in_ndc=False)
    return cam


class PcloudRenderer:
    "a simple wrapper for pytorch3d point cloud renderer"
    def __init__(self, image_size=1024, radius=0.005, points_per_pixel=10,
                 device='cuda:0', bin_size=128, batch_size=1, ret_depth=False,
                 bkg_color=None):
        camera_centers = []
        focal_lengths = []
        if not isinstance(image_size, int):
            self.image_size = image_size
            image_size = image_size[1] # assume image_size is (height, width)
        for i in range(batch_size):
            camera_centers.append(torch.Tensor([image_size / 2., image_size / 2.]).to(device))
            focal_lengths.append(torch.Tensor([image_size / 2., image_size / 2.]).to(device))
        self.image_size = image_size
        self.device = device
        self.camera_center = torch.stack(camera_centers)
        self.focal_length = torch.stack(focal_lengths)
        self.ret_depth = ret_depth # return depth  map or not
        self.bkg_color = bkg_color
        self.renderer = self.setup_renderer(radius, points_per_pixel, bin_size)

    def render(self, pc, cameras, mode='image'):
        # TODO: support batch rendering
        """
        render the point cloud, compute the world coordinate of each pixel based on zbuf
        image: (H, W, 3)
        xyz_world: (H, W, 3), the third dimension is the xyz coordinate in world space
        """
        images, fragments = self.renderer(pc, cameras=cameras)
        if mode=='image':
            if images.shape[0] == 1:
                img = images[0, ..., :3].cpu().numpy().copy()
            else:
                img = images[..., :3].cpu().numpy().copy()
            return img
        elif mode=='mask':
            zbuf = torch.mean(fragments.zbuf, -1)  # (B, H, W)
            masks = zbuf >= 0
            if images.shape[0] == 1:
                img = images[0, ..., :3].cpu().numpy()
                masks = masks[0].cpu().numpy().astype(bool)
            else:
                img = images[..., :3].cpu().numpy()
                masks = masks.cpu().numpy().astype(bool)

            return img, masks

    def get_xy_ndc(self):
        """
        return (H, W, 2), each pixel is the x,y coordinate in NDC space
        """
        py, px = torch.meshgrid(torch.linspace(0, self.image_size-1, self.image_size),
                                torch.linspace(0, self.image_size-1, self.image_size))
        x_ndc = 1 - 2*px/(self.image_size - 1)
        y_ndc = 1 - 2*py/(self.image_size - 1)
        xy_ndc = torch.stack([x_ndc, y_ndc], axis=-1).to(self.device)
        return xy_ndc.squeeze(0).unsqueeze(0)

    def setup_renderer(self, radius, points_per_pixel, bin_size):
        raster_settings = PointsRasterizationSettings(
            image_size=self.image_size,
            radius=radius,
            points_per_pixel=points_per_pixel,
            bin_size=bin_size,
            max_points_per_bin=500000
        )
        # Create a points renderer by compositing points using an alpha compositor (nearer points
        # are weighted more heavily). See [1] for an explanation.
        rasterizer = PointsRasterizer(raster_settings=raster_settings)
        renderer = PointsRendererWithFragments(
            rasterizer=rasterizer,
            compositor=AlphaCompositor(background_color=self.bkg_color) # use white bkg, otherwise will be black
        )
        return renderer


class PointsRendererWithFragments(torch.nn.Module):
    def __init__(self, rasterizer, compositor):
        super().__init__()
        self.rasterizer = rasterizer
        self.compositor = compositor

    def forward(self, point_clouds, **kwargs) -> (torch.Tensor, torch.Tensor):
        fragments = self.rasterizer(point_clouds, **kwargs)
        # Construct weights based on the distance of a point to the true point.
        # However, this could be done differently: e.g. predicted as opposed
        # to a function of the weights.
        r = self.rasterizer.raster_settings.radius

        dists2 = fragments.dists.permute(0, 3, 1, 2)
        weights = 1 - dists2 / (r * r)
        images = self.compositor(
            fragments.idx.long().permute(0, 3, 1, 2),
            weights,
            point_clouds.features_packed().permute(1, 0),
            **kwargs,
        )

        # permute so image comes at the end
        images = images.permute(0, 2, 3, 1)

        return images, fragments

# class PcloudsRenderer


class DepthRasterizer(torch.nn.Module):
    """
    simply rasterize a mesh or point cloud to depth image
    """
    def __init__(self, image_size, dtype='pc',
                 radius=0.005, points_per_pixel=1,
                 bin_size=128,
                 blur_radius=0,
                 max_faces_per_bin=50000,
                 faces_per_pixel=1,):
        """
        image_size: (height, width)
        """
        super(DepthRasterizer, self).__init__()
        if dtype == 'pc':
            raster_settings = PointsRasterizationSettings(
                image_size=image_size,
                radius=radius,
                points_per_pixel=points_per_pixel,
                bin_size=bin_size
            )
            self.rasterizer = PointsRasterizer(raster_settings=raster_settings)
        elif dtype == 'mesh':
            raster_settings = RasterizationSettings(
                image_size=image_size,
                blur_radius=blur_radius,
                faces_per_pixel=faces_per_pixel,
                clip_barycentric_coords=False,
                max_faces_per_bin=max_faces_per_bin
            )
            self.rasterizer=MeshRasterizer(raster_settings=raster_settings)
        else:
            raise NotImplemented

    def forward(self, data, to_np=True, **kwargs):
        fragments = self.rasterizer(data, **kwargs)
        if to_np:
            zbuf = fragments.zbuf # (B, H, W, points_per_pixel)
            return zbuf[0, ..., 0].cpu().numpy()
        return fragments.zbuf


def test_depth_rasterizer():
    from psbody.mesh import Mesh
    import cv2
    m = Mesh()
    m.load_from_file("/BS/xxie-4/work/kindata/Sep29_shuo_chairwood_hand/t0003.000/person/person.ply")
    device = 'cuda:0'
    pc = Pointclouds([torch.from_numpy(m.v).float().to(device)],
                     features=[torch.from_numpy(m.vc).float().to(device)])
    rasterizer = DepthRasterizer(image_size=(480, 640))
    camera = get_kinect_camera(device)

    depth = rasterizer(pc, cameras=camera)
    std = torch.std(depth, -1)
    print('max std', torch.max(std)) # maximum std is up to 1.7m, too much!
    print('min std', torch.min(std))

    print(depth.shape)
    dmap = depth[0, ..., 0].cpu().numpy()
    dmap[dmap<0] = 0
    cv2.imwrite('debug/depth.png', (dmap*1000).astype(np.uint16))

def test_mesh_rasterizer():
    from psbody.mesh import Mesh
    import cv2
    m = Mesh()
    m.load_from_file("/BS/xxie-4/work/kindata/Sep29_shuo_chairwood_hand/t0003.000/person/fit02/person_fit.ply")
    device = 'cuda:0'
    mesh = Meshes([torch.from_numpy(m.v).float().to(device)],
                  [torch.from_numpy(m.f.astype(int)).to(device)])
    rasterizer = DepthRasterizer(image_size=(480, 640), dtype='mesh')
    camera = get_kinect_camera(device)

    depth = rasterizer(mesh, to_np=False, cameras=camera)

    print(depth.shape)
    dmap = depth[0, ..., 0].cpu().numpy()
    dmap[dmap < 0] = 0
    cv2.imwrite('debug/depth_mesh.png', (dmap * 1000).astype(np.uint16))



if __name__ == '__main__':
    test_mesh_rasterizer()






