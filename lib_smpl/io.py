# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import pickle

import chumpy as ch
import numpy as np
import cv2
from chumpy.ch import MatVecMult
from .geometry import verts_core, Rodrigues


def compute_local_rotation_offsets(pose):
    """Compute the local rotation offsets (R - I) for each joint from pose parameters.
    Args:
        pose: pose parameters, either numpy array or chumpy array
    Returns:
        Flattened array of (rotation_matrix - identity) for each joint (excluding root)
    """
    if isinstance(pose, np.ndarray):
        pose = pose.ravel()[3:]
        return np.concatenate([
            (cv2.Rodrigues(np.array(joint_pose))[0] - np.eye(3)).ravel()
            for joint_pose in pose.reshape((-1, 3))
        ]).ravel()
    if pose.ndim != 2 or pose.shape[1] != 3:
        pose = pose.reshape((-1, 3))
    pose = pose[1:]
    return ch.concatenate([
        (Rodrigues(joint_pose) - ch.eye(3)).ravel()
        for joint_pose in pose
    ]).ravel()


def get_pose_mapping(blend_shape_type):
    """Return the pose mapping function for the given blend shape type.
    Args:
        blend_shape_type: string identifier, currently only 'lrotmin' is supported
    Returns:
        A function that maps pose parameters to blend shape coefficients
    """
    if blend_shape_type == 'lrotmin':
        return compute_local_rotation_offsets
    else:
        raise Exception('Unknown pose mapping: %s' % (str(blend_shape_type),))


def apply_backwards_compatibility(model_dict):
    """Apply backwards compatibility replacements to SMPL model dictionary keys.
    Args:
        model_dict: dictionary loaded from SMPL pkl file
    """
    # Key renames for older model formats
    if 'default_v' in model_dict:
        model_dict['v_template'] = model_dict['default_v']
        del model_dict['default_v']
    if 'template_v' in model_dict:
        model_dict['v_template'] = model_dict['template_v']
        del model_dict['template_v']
    if 'joint_regressor' in model_dict:
        model_dict['J_regressor'] = model_dict['joint_regressor']
        del model_dict['joint_regressor']
    if 'blendshapes' in model_dict:
        model_dict['posedirs'] = model_dict['blendshapes']
        del model_dict['blendshapes']
    if 'J' not in model_dict:
        model_dict['J'] = model_dict['joints']
        del model_dict['joints']

    # Set defaults
    if 'bs_style' not in model_dict:
        model_dict['bs_style'] = 'lbs'


def load_smpl_model_data(fname_or_dict):
    """Load and prepare SMPL model data from a pkl file path or a dictionary.
    Args:
        fname_or_dict: either a file path string to a pkl file, or a pre-loaded dictionary
    Returns:
        Dictionary with all model data ready for use (v_template, shapedirs, posedirs, etc.)
    """
    if not isinstance(fname_or_dict, dict):
        model_dict = pickle.load(open(fname_or_dict, 'rb'), encoding='latin-1')
    else:
        model_dict = fname_or_dict

    apply_backwards_compatibility(model_dict)

    has_shape_model = 'shapedirs' in model_dict
    num_pose_params = model_dict['kintree_table'].shape[1] * 3

    if 'trans' not in model_dict:
        model_dict['trans'] = np.zeros(3)
    if 'pose' not in model_dict:
        model_dict['pose'] = np.zeros(num_pose_params)
    if 'shapedirs' in model_dict and 'betas' not in model_dict:
        model_dict['betas'] = np.zeros(model_dict['shapedirs'].shape[-1])

    for key in ['v_template', 'weights', 'posedirs', 'pose', 'trans', 'shapedirs', 'betas', 'J']:
        if (key in model_dict) and not hasattr(model_dict[key], 'dterms'):
            model_dict[key] = ch.array(model_dict[key])

    if has_shape_model:
        model_dict['v_shaped'] = model_dict['shapedirs'].dot(model_dict['betas']) + model_dict['v_template']
        shaped_vertices = model_dict['v_shaped']
        joint_x = MatVecMult(model_dict['J_regressor'], shaped_vertices[:, 0])
        joint_y = MatVecMult(model_dict['J_regressor'], shaped_vertices[:, 1])
        joint_z = MatVecMult(model_dict['J_regressor'], shaped_vertices[:, 2])
        model_dict['J'] = ch.vstack((joint_x, joint_y, joint_z)).T
        model_dict['v_posed'] = shaped_vertices + model_dict['posedirs'].dot(
            get_pose_mapping(model_dict['bs_type'])(model_dict['pose'])
        )
    else:
        model_dict['v_posed'] = model_dict['v_template'] + model_dict['posedirs'].dot(
            get_pose_mapping(model_dict['bs_type'])(model_dict['pose'])
        )

    return model_dict


def load_model(fname_or_dict):
    """Load a full SMPL model and compute initial vertices and joint positions.
    Args:
        fname_or_dict: either a file path string to a pkl file, or a pre-loaded dictionary
    Returns:
        Chumpy object with vertices, joint transforms, and all model attributes
    """
    model_dict = load_smpl_model_data(fname_or_dict)

    args = {
        'pose': model_dict['pose'],
        'v': model_dict['v_posed'],
        'J': model_dict['J'],
        'weights': model_dict['weights'],
        'kintree_table': model_dict['kintree_table'],
        'xp': ch,
        'want_Jtr': True,
        'bs_style': model_dict['bs_style']
    }

    result, joint_transforms = verts_core(**args)
    result = result + model_dict['trans'].reshape((1, 3))
    result.J_transformed = joint_transforms + model_dict['trans'].reshape((1, 3))

    for key, value in model_dict.items():
        setattr(result, key, value)

    return result
