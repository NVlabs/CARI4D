# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from pytorch3d.datasets import collate_batched_meshes
import json
from torch.utils.data import DataLoader
from learning.training.training_config import TrainTemporalRefinerConfig


def get_dataset(cfg: TrainTemporalRefinerConfig):
    ""
    from learning.datasets.video_data import VideoDataset, VideoDataProcessor
    from learning.datasets.behave_fullseq import BehaveFullSeqTestDataset
    d = json.load(open(cfg.split_file))
    seqs_train, seqs_test = d['train'], d['test']

    if cfg.data_name == 'video-data':
        data_class = VideoDataset
    elif cfg.data_name == 'behave-fullseq':
        data_class = BehaveFullSeqTestDataset # this is for testing 
    elif cfg.data_name == 'test-only':
        data_class = VideoDataProcessor
    else:
        raise ValueError('Unknown dataset: {}'.format(cfg.data_name))
    dataset_train, dataset_test = data_class(cfg, seqs_train, 'train'), data_class(cfg, seqs_test, 'val')

    shuffle = cfg.job == 'train'
    print(f"In total {len(dataset_train)} training and {len(dataset_test)} test samples, shuffle? {shuffle}")
    dataloader_train = DataLoader(dataset_train, batch_size=cfg.batch_size,
                                num_workers=cfg.num_workers, shuffle=shuffle)
    dataloader_test = DataLoader(dataset_test, batch_size=cfg.batch_size,
                                num_workers=cfg.num_workers//2, shuffle=shuffle)

    return dataloader_train, dataloader_test, dataset_test, dataset_train


