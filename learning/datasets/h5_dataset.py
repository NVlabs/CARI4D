# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.



import os,sys,h5py,bisect,io,json
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../../../../')
from learning.datasets.pose_dataset import *
from learning.datasets.augmentations import *

BACKGROUND_ROOT = f'VOC2012/JPEGImages/'



class PairH5Dataset(torch.utils.data.Dataset):
  def __init__(self, cfg, h5_file, mode='train', max_num_key=None, cache_data=None):
    self.cfg = cfg
    self.h5_file = h5_file
    self.mode = mode

    logging.info(f"self.h5_file:{self.h5_file}")
    self.n_perturb = None
    self.H_ori = None
    self.W_ori = None
    self.cache_data = cache_data

    if self.mode=='test':
      pass
    else:
      self.object_keys = []
      key_file = h5_file.replace('.h5','_keys.pkl')
      if os.path.exists(key_file):
        with open(key_file, 'rb') as ff:
          self.object_keys = pickle.load(ff)
        logging.info(f'object_keys loaded#:{len(self.object_keys)} from {key_file}')
        if max_num_key is not None:
          self.object_keys = self.object_keys[:max_num_key]
      else:
        with h5py.File(h5_file, 'r', libver='latest') as hf:
          for k in hf:
            self.object_keys.append(k)
            if max_num_key is not None and len(self.object_keys)>=max_num_key:
              logging.info("break due to max_num_key")
              break

      logging.info(f'self.object_keys#:{len(self.object_keys)}, max_num_key:{max_num_key}')

      with h5py.File(h5_file, 'r', libver='latest') as hf:
        group = hf[self.object_keys[0]]
        cnt = 0
        for k_perturb in group:
          if 'i_perturb' in k_perturb:
            cnt += 1
          if 'crop_ratio' in group[k_perturb]:
            self.cfg['crop_ratio'] = float(group[k_perturb]['crop_ratio'][()])
          if self.H_ori is None:
            if 'H_ori' in group[k_perturb]:
              self.H_ori = int(group[k_perturb]['H_ori'][()])
              self.W_ori = int(group[k_perturb]['W_ori'][()])
            else:
              self.H_ori = 540
              self.W_ori = 720
        self.n_perturb = cnt
        logging.info(f'self.n_perturb:{self.n_perturb}')


  def __len__(self):
    if self.mode=='test':
      return 1
    return len(self.object_keys)



  def transform_depth_to_xyzmap(self, batch:BatchPoseData, H_ori, W_ori, bound=1, subtract_trans=True, crop_xyz_3d=False):
    bs = len(batch.rgbAs)
    H,W = batch.rgbAs.shape[-2:]
    mesh_radius = batch.mesh_diameters.cuda()/2
    tf_to_crops = batch.tf_to_crops.cuda()
    crop_to_oris = batch.tf_to_crops.inverse().cuda()  #(B,3,3)
    batch.poseA = batch.poseA.cuda()
    batch.Ks = batch.Ks.cuda()
    if batch.xyz_mapAs is None:
      depthAs_ori = kornia.geometry.transform.warp_perspective(batch.depthAs.cuda().expand(bs,-1,-1,-1), crop_to_oris, dsize=(H_ori, W_ori), mode='nearest', align_corners=False)
      batch.xyz_mapAs = depth2xyzmap_batch(depthAs_ori[:,0], batch.Ks, zfar=np.inf).permute(0,3,1,2)  #(B,3,H,W)
      batch.xyz_mapAs = kornia.geometry.transform.warp_perspective(batch.xyz_mapAs, tf_to_crops, dsize=(H,W), mode='nearest', align_corners=False)
    batch.xyz_mapAs = batch.xyz_mapAs.cuda()
    if self.cfg['normalize_xyz']:
      invalid = batch.xyz_mapAs[:,2:3]<0.001
    if batch.xyz_mapAs.shape[1]==3:
      batch.xyz_mapAs = batch.xyz_mapAs-batch.poseA[:,:3,3].reshape(bs,3,1,1) if subtract_trans else batch.xyz_mapAs
      if self.cfg['normalize_xyz']:
        batch.xyz_mapAs *= 1/mesh_radius.reshape(bs,1,1,1)
        invalid = invalid.expand(bs,3,-1,-1) | (torch.abs(batch.xyz_mapAs)>=2)
        batch.xyz_mapAs[invalid.expand(bs,3,-1,-1)] = 0
    else:
      assert batch.xyz_mapAs.shape[1] == 5, 'invalid xyz_mapAs shape {}'.format(batch.xyz_mapAs.shape)
      mask_ho = batch.xyz_mapAs[:, 3:]
      batch.xyz_mapAs = batch.xyz_mapAs[:, :3] - batch.poseA[:, :3, 3].reshape(bs, 3, 1, 1) if subtract_trans else batch.xyz_mapAs[:, :3]
      if self.cfg['normalize_xyz']:
        batch.xyz_mapAs *= 1 / mesh_radius.reshape(bs, 1, 1, 1)
        invalid = invalid.expand(bs, 3, -1, -1) | (torch.abs(batch.xyz_mapAs) >= 2)
        batch.xyz_mapAs[invalid.expand(bs, 3, -1, -1)] = 0
      batch.xyz_mapAs = torch.cat([batch.xyz_mapAs, mask_ho], dim=1)

    if batch.xyz_mapBs is None:
      depthBs_ori = kornia.geometry.transform.warp_perspective(batch.depthBs.cuda().expand(bs,-1,-1,-1), crop_to_oris, dsize=(H_ori, W_ori), mode='nearest', align_corners=False)
      batch.xyz_mapBs = depth2xyzmap_batch(depthBs_ori[:,0], batch.Ks, zfar=np.inf).permute(0,3,1,2)  #(B,3,H,W)
      batch.xyz_mapBs = kornia.geometry.transform.warp_perspective(batch.xyz_mapBs, tf_to_crops, dsize=(H,W), mode='nearest', align_corners=False)
    batch.xyz_mapBs = batch.xyz_mapBs.cuda()
    if self.cfg['normalize_xyz']:
      invalid = batch.xyz_mapBs[:,2:3]<0.001
    # TODO: understand why here
    if batch.xyz_mapBs.shape[1] == 3:
      batch.xyz_mapBs = batch.xyz_mapBs - batch.poseA[:,:3,3].reshape(bs,3,1,1) if subtract_trans else batch.xyz_mapBs
      if self.cfg['normalize_xyz']:
        batch.xyz_mapBs *= 1/mesh_radius.reshape(bs,1,1,1)
        invalid = invalid.expand(bs,3,-1,-1) | (torch.abs(batch.xyz_mapBs)>=2)
        batch.xyz_mapBs[invalid.expand(bs,3,-1,-1)] = 0
    else:
      assert batch.xyz_mapBs.shape[1] == 5, 'invalid xyz_mapBs shape {}'.format(batch.xyz_mapBs.shape)
      mask_ho = batch.xyz_mapBs[:, 3:] # TODO: understand why poseA is subtracted?
      batch.xyz_mapBs = batch.xyz_mapBs[:, :3] - batch.poseA[:, :3, 3].reshape(bs, 3, 1, 1) if subtract_trans else batch.xyz_mapBs[:, :3]
      if self.cfg['normalize_xyz']:
        batch.xyz_mapBs *= 1 / mesh_radius.reshape(bs, 1, 1, 1)
        invalid = invalid.expand(bs, 3, -1, -1) | (torch.abs(batch.xyz_mapBs) >= 2) # XH: here it is cropped!!
        batch.xyz_mapBs[invalid.expand(bs, 3, -1, -1)] = 0
      batch.xyz_mapBs = torch.cat([batch.xyz_mapBs, mask_ho], dim=1)
    if crop_xyz_3d:
      assert subtract_trans
      print('cropping xyz_mapBs in 3D bbox')
      dmap_xyz = batch.xyz_mapBs[:, :3]
      bound_min, bound_max = np.array([-1, -1, -1.]), np.array([1, 1, 1.])
      m = ((dmap_xyz[:, 0] < bound_max[0]) & (dmap_xyz[:, 0] > bound_min[0])
           & (dmap_xyz[:, 1] < bound_max[1]) & (dmap_xyz[:, 1] > bound_min[1])
           & (dmap_xyz[:, 1] < bound_max[2]) & (dmap_xyz[:, 1] > bound_min[2]))  # (B, H, W)
      dmap_xyz[~m[:, None].repeat(1, 3, 1, 1)] = 0.
      batch.xyz_mapBs[:, :3] = dmap_xyz

    return batch


  def transform_batch(self, batch:BatchPoseData, H_ori, W_ori, bound=1):
    bs = len(batch.rgbAs)
    batch.rgbAs = batch.rgbAs.cuda().float()/255.0
    batch.rgbBs = batch.rgbBs.cuda().float()/255.0

    batch = self.transform_depth_to_xyzmap(batch, H_ori, W_ori, bound=bound)
    return batch




class TripletH5Dataset(PairH5Dataset):
  def __init__(self, cfg, h5_file, mode, max_num_key=None, cache_data=None):
    super().__init__(cfg, h5_file, mode, max_num_key, cache_data=cache_data)


  def make_aug(self):
    if self.mode in ['train', 'val']:
      self.aug = ComposedAugmenter([
        ReplaceBackground(prob=0.2, img_root=BACKGROUND_ROOT),
        RGBGaussianNoise(max_noise=15, prob=0.3),
        ChangeBright(mag=[0.5, 1.5], prob=0.5, augment_imgA=True),
        ChangeContrast(mag=[0.8, 1.2], prob=0.5),
        GaussianBlur(max_kernel_size=7, min_kernel_size=3, sigma_range=(0, 3.0), prob=0.3),
        JpegAug(prob=0.5, compression_range=[0,20]),
        DepthCorrelatedGaussianNoise(prob=1, H_ori=self.H_ori, W_ori=self.W_ori, noise_range=[0, 0.01], rescale_factor_min=2, rescale_factor_max=10),
        DepthMissing(prob=1, H_ori=self.H_ori, W_ori=self.W_ori, max_missing_percent=0.5, down_scale=1),
        DepthRoiMissing(prob=0.5, H_ori=self.H_ori, W_ori=self.W_ori, max_missing_ratio=0.5, downscale_range=[0.1, 1]),
        DepthEllipseMissing(prob=0.5, H_ori=self.H_ori, W_ori=self.W_ori, max_num_ellipse=20, max_radius=30),
      ])
    else:
     self.aug = ComposedAugmenter([
      ])



  def transform_depth_to_xyzmap(self, batch:BatchPoseData, H_ori, W_ori, bound=1):
    bs = len(batch.rgbAs)
    H,W = batch.rgbAs.shape[-2:]
    mesh_radius = batch.mesh_diameters.cuda()/2
    tf_to_crops = batch.tf_to_crops.cuda()
    crop_to_oris = batch.tf_to_crops.inverse().cuda()  #(B,3,3)
    batch.poseA = batch.poseA.cuda()
    batch.Ks = batch.Ks.cuda()

    if batch.xyz_mapAs is None:
      depthAs_ori = kornia.geometry.transform.warp_perspective(batch.depthAs.cuda().expand(bs,-1,-1,-1), crop_to_oris, dsize=(H_ori, W_ori), mode='nearest', align_corners=False)
      batch.xyz_mapAs = depth2xyzmap_batch(depthAs_ori[:,0], batch.Ks, zfar=np.inf).permute(0,3,1,2)  #(B,3,H,W)
      batch.xyz_mapAs = kornia.geometry.transform.warp_perspective(batch.xyz_mapAs, tf_to_crops, dsize=(H,W), mode='nearest', align_corners=False)
    batch.xyz_mapAs = batch.xyz_mapAs.cuda()
    invalid = batch.xyz_mapAs[:,2:3]<0.1
    batch.xyz_mapAs = (batch.xyz_mapAs-batch.poseA[:,:3,3].reshape(bs,3,1,1))
    if self.cfg['normalize_xyz']:
      batch.xyz_mapAs *= 1/mesh_radius.reshape(bs,1,1,1)
      invalid = invalid.expand(bs,3,-1,-1) | (torch.abs(batch.xyz_mapAs)>=2)
      batch.xyz_mapAs[invalid.expand(bs,3,-1,-1)] = 0

    if batch.xyz_mapBs is None:
      # make mini batch to avoid OOM issue
      chunk_size, xyz_mapBs_list = 128, []
      for i in range(0, bs, chunk_size):
        depthBs_ori = kornia.geometry.transform.warp_perspective(batch.depthBs.expand(bs, -1, -1, -1)[i:i+chunk_size],
                                                                 crop_to_oris[i:i+chunk_size], dsize=(H_ori, W_ori),
                                                                 mode='nearest', align_corners=False)
        xyz_mapBs = depth2xyzmap_batch(depthBs_ori[:, 0], batch.Ks[i:i+chunk_size], zfar=np.inf).permute(0, 3, 1, 2)  # (B,3,H,W)
        xyz_mapBs = kornia.geometry.transform.warp_perspective(xyz_mapBs, tf_to_crops[i:i+chunk_size], dsize=(H, W),
                                                                     mode='nearest', align_corners=False)
        xyz_mapBs_list.append(xyz_mapBs)
      batch.xyz_mapBs = torch.cat(xyz_mapBs_list, 0)
    batch.xyz_mapBs = batch.xyz_mapBs.cuda()
    invalid = batch.xyz_mapBs[:,2:3]<0.1
    batch.xyz_mapBs = (batch.xyz_mapBs-batch.poseA[:,:3,3].reshape(bs,3,1,1))
    if self.cfg['normalize_xyz']:
      batch.xyz_mapBs *= 1/mesh_radius.reshape(bs,1,1,1)
      invalid = invalid.expand(bs,3,-1,-1) | (torch.abs(batch.xyz_mapBs)>=2)
      batch.xyz_mapBs[invalid.expand(bs,3,-1,-1)] = 0

    return batch


  def transform_batch(self, batch:BatchPoseData, H_ori, W_ori, bound=1):
    bs = len(batch.rgbAs)
    batch.rgbAs = batch.rgbAs.cuda().float()/255.0
    batch.rgbBs = batch.rgbBs.cuda().float()/255.0

    batch = self.transform_depth_to_xyzmap(batch, H_ori, W_ori, bound=bound)
    return batch



class ScoreMultiPairH5Dataset(TripletH5Dataset):
  def __init__(self, cfg, h5_file, mode, max_num_key=None, cache_data=None):
    super().__init__(cfg, h5_file, mode, max_num_key, cache_data=cache_data)
    if mode in ['train', 'val']:
      self.cfg['train_num_pair'] = self.n_perturb


class PoseRefinePairH5Dataset(PairH5Dataset):
  def __init__(self, cfg, h5_file, mode='train', max_num_key=None, cache_data=None):
    super().__init__(cfg=cfg, h5_file=h5_file, mode=mode, max_num_key=max_num_key, cache_data=cache_data)

    if mode!='test':
      with h5py.File(h5_file, 'r', libver='latest') as hf:
        group = hf[self.object_keys[0]]
        for key_perturb in group:
          depthA = imageio.imread(group[key_perturb]['depthA'][()])
          depthB = imageio.imread(group[key_perturb]['depthB'][()])
          self.cfg['n_view'] = min(self.cfg['n_view'], depthA.shape[1]//depthB.shape[1])
          logging.info(f'n_view:{self.cfg["n_view"]}')
          self.trans_normalizer = group[key_perturb]['trans_normalizer'][()]
          if isinstance(self.trans_normalizer, np.ndarray):
            self.trans_normalizer = self.trans_normalizer.tolist()
          self.rot_normalizer = group[key_perturb]['rot_normalizer'][()]/180.0*np.pi
          logging.info(f'self.trans_normalizer:{self.trans_normalizer}, self.rot_normalizer:{self.rot_normalizer}')
          break

  def make_aug(self):
    if self.mode=='train' or self.mode=='val':
      self.aug = ComposedAugmenter([
        ReplaceBackground(prob=0.2, img_root=BACKGROUND_ROOT),
        RGBGaussianNoise(max_noise=15, prob=0.3),
        ChangeBright(mag=[0.5, 1.5], prob=0.5, augment_imgA=True),
        ChangeContrast(mag=[0.8, 1.2], prob=0.5),
        GaussianBlur(max_kernel_size=7, min_kernel_size=3, sigma_range=(0, 3.0), prob=0.3),
        JpegAug(prob=0.5, compression_range=[0,20]),
        DepthCorrelatedGaussianNoise(prob=1, H_ori=self.H_ori, W_ori=self.W_ori, noise_range=[0, 0.01], rescale_factor_min=2, rescale_factor_max=10),
        DepthMissing(prob=1, H_ori=self.H_ori, W_ori=self.W_ori, max_missing_percent=0.5, down_scale=1),
        DepthRoiMissing(prob=0.5, H_ori=self.H_ori, W_ori=self.W_ori, max_missing_ratio=0.5, downscale_range=[0.1, 1]),
        DepthEllipseMissing(prob=0.5, H_ori=self.H_ori, W_ori=self.W_ori, max_num_ellipse=20, max_radius=30),
      ])
    else:
     self.aug = ComposedAugmenter([
      ])


  def transform_batch(self, batch:BatchPoseData, H_ori, W_ori, bound=1, subtract_trans=True, crop_xyz_3d=False):
    bs = len(batch.rgbAs)
    batch.rgbAs = batch.rgbAs.cuda().float()/255.0
    batch.rgbBs = batch.rgbBs.cuda().float()/255.0

    batch = self.transform_depth_to_xyzmap(batch, H_ori, W_ori, bound=bound, subtract_trans=subtract_trans, crop_xyz_3d=crop_xyz_3d)
    return batch

