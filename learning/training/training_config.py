# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os,sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple,Union, Dict, Any
import numpy as np
import omegaconf
import torch


@dataclass
class TrainingConfig(omegaconf.dictconfig.DictConfig):
    input_resize: tuple = (160, 160)
    normalize_xyz:Optional[bool] = True
    use_mask:Optional[bool] = False
    crop_ratio:Optional[float] = None
    split_objects_across_gpus: bool = True
    max_num_key: Optional[int] = None
    use_normal:bool = False
    n_view:int = 1
    zfar:float = np.inf
    c_in:int = 6
    train_num_pair:Optional[int] = None
    make_pair_online:Optional[bool] = False
    render_backend:Optional[str] = 'nvdiffrast'

    # Run management
    run_id: Optional[str] = None
    exp_name:Optional[str] = None
    resume_run_id: Optional[str] = None
    save_dir: Optional[str] = None
    batch_size: int = 64
    epoch_size: int = 115200
    val_size: int = 1280
    n_epochs: int = 25
    save_epoch_interval: int = 100
    n_dataloader_workers: int = 20
    n_rendering_workers: int = 1
    gradient_max_norm:float = np.inf
    max_step_per_epoch: Optional[int] = 25000

    # Network
    use_BN:bool = True
    loss_type:Optional[str] = 'pairwise_valid'

    # Optimizer
    optimizer: str = "adam"
    weight_decay: float = 0.0
    clip_grad_norm: float = np.inf
    lr: float = 0.0001
    warmup_step: int = -1   # -1 means disable
    n_epochs_warmup: int = 1

    # Visualization
    vis_interval: Optional[int] = 1000

    debug: Optional[bool] = None



@dataclass
class TrainRefinerConfig:
    # Datasets
    input_resize: tuple = (160, 160)  #(W,H)
    crop_ratio:Optional[float] = None
    max_num_key: Optional[int] = None
    use_normal:bool = False
    use_mask:Optional[bool] = False
    normal_uint8:bool = False
    normalize_xyz:Optional[bool] = True
    trans_normalizer:Optional[list] = None
    rot_normalizer:Optional[float] = None
    c_in:int = 6
    n_view:int = 1
    zfar:float = np.inf
    trans_rep:str = 'tracknet'
    rot_rep:Optional[str] = 'axis_angle'  # 6d/axis_angle
    save_dir: Optional[str] = None

    # Run management
    run_id: Optional[str] = None
    exp_name:Optional[str] = None
    batch_size: int = 64
    use_BN:bool = True
    optimizer: str = "adam"
    weight_decay: float = 0.0
    clip_grad_norm: float = np.inf
    lr: float = 0.0001
    warmup_step: int = -1
    loss_type:str = 'l2'   # l1/l2/add

    vis_interval: Optional[int] = 1000
    debug: Optional[bool] = None


@dataclass
class LRSchedulerConfig:
    type: str = 'none'
    kwargs: Dict = field(default_factory=lambda: dict())


@dataclass
class LinearSchedulerConfig(LRSchedulerConfig):
    type: str = 'transformers'

    kwargs: Dict = field(default_factory=lambda: dict(
        name='linear',
        num_warmup_steps=0,
        num_training_steps="${max_steps}",
    ))


@dataclass
class DenoiserConfig:
    out_dim_hum: int = 157 # smpl pose 6d, transl, betas
    out_dim_obj: int = 9
    out_dim_contact: int = 0
    avgbeta: bool = True 
    latent_dim_hum: int = 512
    latent_dim_obj: int = 512   
    latent_dim_xt: int = 256
    latent_dim_contact: int = 0
    latent_dim: int = 1024
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    num_layers: int = 12
    num_heads: int = 8
    max_len: int = 180 # set this automatically to clip_len 
    layer_norm: bool = False
    bps_repr: str = 'none' # the bps input representation, none for no bps encoding. 
    bps_dim: int = 0 # the dimension of the bps encoding 
    pred_contact_points: bool = False # predict contact points 
    num_contact_points: int = 8 # number of contact points 

@dataclass
class DiffSchedulerInitArgsConfig:
    beta_start: float = 1e-5  # 0.00085
    beta_end: float = 8e-3  # 0.012
    beta_schedule: str = "linear"
    prediction_type: str = "sample"  # predict x0

@dataclass
class DiffSchedulerFullConfig:
    init_args: DiffSchedulerInitArgsConfig = field(default_factory=DiffSchedulerInitArgsConfig)
    num_inference_steps: int = 5
    eta: float = 1.0 # with step 5 is the best
    disable_tqdm: bool = False
    contact_guidance: float = 0. # default none 
    cfg_prob: float = 0. # probability to use cfg training: zero out the cond probability
    cfg_weight: float = 0 # weight for the cfg training, zero means only cond, see https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#classifier-free-guidance 
    input_predx0: bool = False # input the predicted x0 as the cond, not the noisy x0 

@dataclass
class ContactOptimConfig:
    # optimization config for contact optimization
    contact_dist_thres: float = 0.2 # only compute loses smaller than this 
    opt_rot: bool = False # optimize the rotation of the object
    opt_trans: bool = True # optimize the translation of the object
    opt_betas: bool = False # optimize the betas of the object
    opt_smpl_pose: bool = True # optimize the smpl pose 
    opt_smpl_trans: bool = False # optimize the smpl translation

    # contact prediction config 
    contact_dim: int = 24 # contact dimension
    contact_pred_type: str = 'binary' # binary or distance 
    contact_mask_thres: float = 0.015 # contact mask threshold. 

    # loss weights
    w_contact: float = 100.0 # weight for contact loss
    w_acc_r: float = 100.0 # weight for temporal smoothness loss
    w_acc_t: float = 100.0 # weight for temporal smoothness loss
    w_acc_v: float = 100.0 # weight for temporal smoothness loss of object points 
    w_orig_t: float = 100.0 # weight for original translation loss
    w_cdir: float = 0.0 # weight for contact direction loss

    # input file
    pth_file: str = 'xxx' # path to the pth file, contains input, pr and GT 
    wild_video: bool = False # whether the video is in the wild
    data_source: str = 'behave' # behave or hodome 
    
    # opt configs
    lr: float = 0.001
    num_steps: int = 300
    batch_size: int = 384

    # logging
    no_wandb: bool = True 
    viz_steps: int = 500 
    save_every_n_steps: int = 500 
    save_name: str = 'contact'
    debug: bool = False
    use_gt: bool = False # use GT contacts or not 

    # data paths
    video_root: str = '/home/xianghuix/datasets/behave/videos-demo'
    masks_root: str = '/home/xianghuix/datasets/behave/masks-h5-my'
    packed_root: str = '/home/xianghuix/datasets/behave/behave-packed'
    hy3d_meshes_root: str = '/home/xianghuix/datasets/behave/selected-views/hy3d-aligned-center'
    index: Optional[int] = None # index of the video to use

@dataclass
class RefineOutOptimConfig(ContactOptimConfig):
    save_name: str = 'refineout'
    w_sil: float = 1e-3 # weight for silhouette loss 
    w_pen: float = 10.0 # weight for penetration loss 
    w_j2d: float = 0.02 # weight for 2D joint loss 
    w_contact: float = 0.1 # weight for contact loss 
    w_temp: float = 30.0 # weight for temporal smoothness loss 
    # w_velo: float = 30.0 # weight for velocity loss: object mainly 
    w_velo: float = 0.0 # weight for velocity loss: object mainly 
    w_init_ot: float = 100.0 # weight for initial object translation 
    w_init_ht: float = 8000.0 # weight for initial human translation, do not allow large human translation changes 
    w_pinit: float = 0 # weight for init pose 
    pen_loss_start: float = 0.6 # start of the pen loss, after this step, the pen loss is disabled 
    batch_size: int = 192

    opt_rot: bool = True # optimize the rotation of the object
    opt_trans: bool = True # optimize the translation of the object
    opt_betas: bool = False # optimize the betas of the object
    opt_smpl_pose: bool = True # optimize the smpl pose 
    opt_smpl_trans: bool = False # optimize the smpl translation
    op_thres: float = 0.3 # joint confidence threshold, 0.83 for v2.7 
    # apply oneeuro filter to results first
    oneeuro_type: str = 'none' # none, xyz, or all  

    use_input: bool = False # use the input 2D joints as the initial pose and translation
    outpath: str = 'output/opt'

@dataclass
class TrainTemporalRefinerConfig:
    # Datasets
    rgb_files: Optional[str] = 'xxx' # give the pattern to tar files
    render_files: Optional[str] = 'xxx'  # path to pre-rendered images

    split_file: Optional[str] = 'splits/behave-chairs.json'  # contains train, and val list
    render_root: Optional[str] = '/home/xianghuix/datasets/foundpose_train/behave-h5' # root to all renderings
    rgb_root: Optional[str] = '/home/xianghuix/datasets/behave/30fps/h5-resized' # root to all rgb files
    packed_root: Optional[str] = '/home/xianghuix/datasets/behave/behave-packed'  # root to packed files
    fp_root: Optional[str] = '/home/xianghuix/datasets/behave/fp' # root to fp files
    contacts_root: Optional[str] = '/home/xianghuix/datasets/behave/contact-jts' # root to contact files
    clip_len: int = 96 # temporal length of one clip, 96 & bs=8 maximizes the GPU memory usage
    window: int = 10 # distance between two clip start point
    data_name: str = 'video-data'

    # Dataloaders
    batch_size: int = 8
    num_workers: int = 32

    # Dataset config
    input_resize: tuple = (160, 160)  #(W,H)
    crop_ratio:Optional[float] = None
    max_num_key: Optional[int] = None
    use_normal:bool = False
    use_mask:Optional[bool] = False
    normal_uint8:bool = False
    normalize_xyz:Optional[bool] = True
    trans_normalizer:Optional[list] = None
    rot_normalizer:Optional[float] = None
    pose_init_type:Optional[str] = 'copy-first' # copy first frame for all this window
    add_ho_mask: bool = False
    crop_xyz_3d: bool = False
    subtract_transl: bool = False # subtract xyz map by translation of A
    mask_encode_type: str = 'stack' # stack human + object
    mask_rgb_bkg: bool = False # mask out input rgb image bkg
    occ_drop_thres: float = 2. # when a window contains a frame with occlusion more than this, then drop it
    data_filter_type: str = 'none' # no filter, or drop-occ-<thres>-<probability>, or mixed-<thres> (half heavy occlusion, half simple)
    mask_out_thres_1st: float = 50000 # when the error to the 2nd transformer input is larger than this, zero it out
    input_gtpose_2nd: bool = False # input GT pose to the 2nd transformer
    input_fp_2nd: bool = False  # input directly FP predictions to 2nd transformer
    pose_pred_dir: Optional[str] = None # input pose dir, if given then use this as the predicted pose
    nlf_root: Optional[str] = None
    exclude_frames: Optional[str] = None # file to frames that should not be used for training
    align_poses: bool = False # align the poses to the canonical shape
    trans_ref_type: str = 'frame' # which translation should be used to normalize the input xyz map 

    # motion diffusion
    random_flip: bool = False
    fp_error_thres: float = 500  # mask out object if error is larger than this
    mask_out_contact_in: bool = False # mask out contact input as well 
    fp_vis_thres: float = 2. # threshold for visibility from FP rendering
    body_vis_thres: float = 2.0 # threshold for visibility from NLF rendering

    # Model architecture
    model_name: str = 'tempnet' # decide which model to use
    c_in:int = 6
    n_view:int = 1
    posi_len: int = 400 # positional encoding length, related to input resolution
    time_posi_len: int = 400
    zfar:float = np.inf
    trans_rep:str = 'tracknet'
    rot_rep:Optional[str] = 'axis_angle'  # 6d/axis_angle
    rot_rep_hum: str = '6d'
    use_fp_pretrained:bool = True
    use_fp_head: bool = False
    dino_model:Optional[str] = 'dinov2_vitb14' # DINO base
    dino_bnorm: bool = False
    encoder_xyzm: str = 'dinov2_vits14' # DINO small
    embed_dim_spatial_temporal: int = 768 # feature dim for performing spatial temporal attention
    mixed_spatial_temporal: bool = False # mix spatial temporal in one transformer layer
    num_attn_layers: int = 1 # spatial attention layers
    feat_dim_6dpose: int = 128 # object pose feature dimension
    obj_pose_dim_input: int = 12 # object pose feature input dim
    pred_head_attn: bool = True # use attention in prediction head
    hum_cond_dim: int = 75 # human information dimension
    abspose_layers: int = 3 # number of attention layers for abs pose prediction
    hum_cond_embed_dim: int = 128
    abspose_posi_encode: bool = False
    feat_dim_mask: int = 192 # feature dimension of the masks sent to 2nd transformer
    bnorm_mask: bool = False  # batch norm for mask feature
    mask_dino_type: str = 'dinov2_vitb14'
    no_hum_mask: bool = False # for 2nd transformer, do not input human mask
    dropout: float = 0.1
    dropout_1st: float = 0.1
    attn_activation: str = 'relu'
    pred_head_dims: tuple = (128, 64)
    pred_head_hum_pose: tuple = (512, 256, 256)
    pred_head_hum_trans: tuple = (512, 256, 128)
    pred_head_contact: tuple = (512, 256, 128)
    cont_out_dim: int = -1 # prediction contacts or not 
    cont_out_type: str = 'binary' # binary or distance 
    cont_mask_thres: float = 0.015 # contact mask threshold. 
    merge_factor: float = 0.5
    merge_strategy: str = 'fixed' # alpha blending strategy
    d_ff_2nd: int = 512 # feedforward model feat dim of 2nd transformer
    nhead_2nd: int = 4
    fp_err_dim: int = -1 # add fp err as feature or not
    visibility_dim: int = -1 # add object visibility value
    pred_uncertainty: bool = False # add uncertainty in output prediction 
    pred_shape: bool = False # add human shape prediction
    beta_nll: float = 1.0 # for uncertainty NLL loss
    var_epsilon: float = 0.001 # to avoid negative values, not really helpful
    ## siMLPe additional
    simlpe_hidden_dim: int = 128
    simlpe_num_layers: int = 48 # use default
    ## TCN network
    tcn_hidden_dim: int = 512
    tcn_num_layers: int = 3
    tcn_kernel_size: int = 11
    tcn_dropout: float = 0.1

    # for D-Linear
    individual_chs: bool = False

    # For Decoder like SMPL head
    transformer_decoder_cfg: Dict = field(default_factory=lambda: dict(
        depth=6, # number of layers
        heads=8,
        mlp_dim=1024, # feedforward MLP dim
        dim_head= 64,
        dropout= 0.0,
        emb_dropout= 0.0,
        norm='layer',
        context_dim='${embed_dim_spatial_temporal}'  # the feat dim after spatial temporal transformer, should be the same as embed_dim_spatial_temporal
    ))
    smpl_head_input: str = 'zero'
    num_body_joints: int = 23
    # End of decoder like SMPL head

    # for HOI diffusion
    denoiser_cfg: DenoiserConfig = field(default_factory=DenoiserConfig)
    diff_scheduler_cfg: DiffSchedulerFullConfig = field(default_factory=DiffSchedulerFullConfig)
    contact_dist_thres: float = 10. # only compute loses smaller than this 

    # Run management
    config: Optional[str] = 'learning/configs/cari4d-release.yml'
    wandb_project: Optional[str] = 'e2etracker'
    no_wandb:bool = False
    job: Optional[str] = 'train'
    run_id: Optional[str] = None
    exp_name:Optional[str] = None
    save_dir: Optional[str] = 'experiments'
    vis_every_n_steps: int = 100 # visualize input every n steps
    log_errors: bool = False # add errors to the visualization images

    max_step_val: int = 20
    num_epochs: int = 2000
    max_steps: int = 200000
    val_epoch_interval: int = 10
    val_step_interval: int = 200
    ckpt_interval: int = 1000 # save ckpt after this steps
    val_at_start: bool = True # do one evaluation at the start
    use_BN:bool = True
    BN_momentum: float = 0.1
    optimizer: str = "adam"
    weight_decay: float = 0.0
    clip_grad_norm: float = np.inf
    lr: float = 0.0001
    warmup_step: int = -1

    # Losses
    w_rot: float = 0.1
    w_transl: float = 1.0
    w_abs_rot: float = 0.1
    w_abs_trans: float = 0.1
    w_hum_rot: float = 0.1
    w_hum_t: float = 0.1
    w_hum_j: float = 0 # joint locations
    w_hum_b: float = 0 # body shape loss
    w_velo: float = 0.0 # velocity loss for combined motion 
    w_velo_obj: float = 0.0 # velocity loss for object motion  
    w_diff_l2: float = 10.0 
    w_contact: float = 0. # explicit contacts after joints are computed 
    w_heatmap: float = 0.0 # heatmap loss for contact points 
    loss_type:str = 'l2'   # l1/l2/add
    enable_amp: bool = False
    lw_acc: float = 0.0 # acceleration loss
    vis_interval: Optional[int] = 1000
    loss_abs_trans_rela: bool = False # when compute abs trans loss, use relative to first frame or not
    train_stage: str = 'train'
    train_smpl_only: bool = False # do not train other branches, but just SMPL
    symm_loss: bool = False # compute loss with all symmetries

    # scheduler
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)

    # Inference config
    mesh_file: Optional[str] = None
    test_scene_dir: Optional[str] = None
    track_refine_iter: int = 1
    save_name: Optional[str] = 'test'
    video_only: Optional[bool] = False
    ckpt: Optional[str] = None
    fp_skip: int = 10
    video_out: str = 'output/viz' # path to save visualizations
    video: str = '' # the path to the video file to be processed 
    debug: Optional[int] = 2
    debug_dir: str = ''
    skip: int = 30 # skip how many frames in sliding window
    redo: bool = False
    test_gt_pose: bool = False
    ckpt_file: Optional[str] = None
    use_intermediate: bool = False # for abs + delta prediction
    eval_normalize: bool = False # normalize the error
    eval_input: bool = False # evaluate the input pose
    identifier: str = '' # for eval output file
    run_smooth: bool = False
    smooth_smplt: bool = False
    render_video: bool = False
    cam_id: int = 1
    refine_iters: int = 1
    wild_video: bool = False
    masks_root: str = '/home/xianghuix/datasets/behave/masks-h5-my'
    hy3d_meshes_root: str = '/home/xianghuix/datasets/cari4d-demo/behave/meshes'
    fp_root: str = '/home/xianghuix/datasets/behave/fp'
    outpath: str = '/home/xianghuix/datasets/behave/foundpose-input/e2etracker/results'

    prev_pred_root: Optional[str] = None # path to previous predicted results, for motion diffusion 
    viz_input: bool = False
    use_sel_view: bool = True # use the selected view for testing
    full_seq: bool = False # use the full sequence for testing, i.e. run sliding window and averaging 
    inf_only: bool = False # only inference, no metrics computation
    opt_name: Optional[str] = None # name of the optimization method

    # evaluation config
    align2gt: bool = True  # align the predicted pose to the GT pose
    result_dir: str = 'none' # path to pth files
    use_hy3d: bool = True # the reconstruction was done using HY3D reconstructed mesh

