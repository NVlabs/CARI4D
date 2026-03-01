# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Local paths to SMPL models, modify before you run.

SMPL_MODEL_ROOT = 'data/smpl'
SMPL_ASSETS_ROOT = f'data/assets'

# related to smpl and smplh parameter count
SMPL_POSE_PRAMS_NUM = 72
SMPLH_POSE_PRAMS_NUM = 156
SMPLH_HANDPOSE_START = 66 # hand pose start index for smplh
NUM_BETAS = 10

# split smplh
GLOBAL_POSE_NUM = 3
BODY_POSE_NUM = 63
HAND_POSE_NUM = 90
TOP_BETA_NUM = 2

# split smpl
SMPL_HAND_POSE_NUM=6

SMPL_PARTS_NUM = 14

# SMPLH->SMPL: keep the first 23 joints (first 66 parameters + 66:69 one hand)
# then pick the 38-th joint parameter (156-15*3 = 111:114)

# 24 SMPL joints 
# add index for each name
JOINT_NAMES = [
    'pelvis', # 0
    'left_hip', # 1
    'right_hip', # 2
    'spine1', # 3
    'left_knee', # 4
    'right_knee', # 5
    'spine2', # 6
    'left_ankle', # 7
    'right_ankle', # 8
    'spine3', # 9
    'left_foot', # 10
    'right_foot', # 11
    'neck', # 12
    'left_collar', # 13
    'right_collar', # 14
    'head', # 15
    'left_shoulder', # 16
    'right_shoulder', # 17
    'left_elbow', # 18          
    'right_elbow', # 19
    'left_wrist', # 20
    'right_wrist', # 21
    'left_hand', # 22
    'right_hand', # 23 
]