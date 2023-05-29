from os import path as osp
from typing import Dict
from unicodedata import name

import numpy as np
import torch
import torch.utils as utils
from numpy.linalg import inv
from src.utils.dataset import (
    read_scannet_gray,
    read_scannet_depth,
    read_scannet_pose,
    read_scannet_intrinsic,
    read_c3vd_pose
)


class C3VDDataset(utils.data.Dataset):
    def __init__(self,
                 root_dir,
                 npz_path,
                 intrinsic_path,
                 mode='train',
                 min_overlap_score=0.4,
                 augment_fn=None,
                 pose_dir=None,
                 **kwargs):
        """Manage one scene of ScanNet Dataset.
        Args:
            root_dir (str): ScanNet root directory that contains scene folders.
            npz_path (str): {scene_id}.npz path. This contains image pair information of a scene.
            intrinsic_path (str): path to depth-camera intrinsic file.
            mode (str): options are ['train', 'val', 'test'].
            augment_fn (callable, optional): augments images with pre-defined visual effects.
            pose_dir (str): ScanNet root directory that contains all poses.
                (we use a separate (optional) pose_dir since we store images and poses separately.)
        """
        super().__init__()
        self.root_dir = root_dir
        self.pose_dir = pose_dir if pose_dir is not None else root_dir
        self.mode = mode

        # prepare data_names, intrinsics and extrinsics(T)
        with np.load(npz_path) as data:
            self.data_names = data['name']
            if 'score' in data.keys() and mode not in ['val' or 'test']:
                kept_mask = data['score'] > min_overlap_score
                self.data_names = self.data_names[kept_mask]
        self.intrinsics = np.array([[935.97, 0, 679.54],
                                    [0, 935.14, 543.98],
                                    [0, 0, 1]])
        
        # for training LoFTR
        self.augment_fn = augment_fn if mode == 'train' else None
        print(self.data_names)
    def __len__(self):
        return len(self.data_names)

    def _read_abs_pose(self, scene_name, name):

        pth = osp.join(self.pose_dir,
                       scene_name,
                       'pose.txt')

        return read_c3vd_pose(pth, name)

    def _compute_rel_pose(self, scene_name, name0, name1):
        pose0 = self._read_abs_pose(scene_name, name0)
        pose1 = self._read_abs_pose(scene_name, name1)
        
        return np.matmul(pose1, inv(pose0))  # (4, 4)

    def __getitem__(self, idx):
        data_name = self.data_names[idx]
        scene_name, stem_name_0, stem_name_1 = data_name

        # read the grayscale image which will be resized to (1, 480, 640)
        img_name0 = osp.join(self.root_dir, scene_name, f'{stem_name_0}_color.png')
        img_name1 = osp.join(self.root_dir, scene_name, f'{stem_name_1}_color.png')
        
        # TODO: Support augmentation & handle seeds for each worker correctly.
        image0 = read_scannet_gray(img_name0, resize=(640, 480), augment_fn=None)
                                #    augment_fn=np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))
        image1 = read_scannet_gray(img_name1, resize=(640, 480), augment_fn=None)
                                #    augment_fn=np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))

        # read the depthmap which is stored as (480, 640)
        if self.mode in ['train', 'val']:
            depth0 = read_scannet_depth(osp.join(self.root_dir, scene_name, f'{stem_name_0}_depth.tiff'))
            depth1 = read_scannet_depth(osp.join(self.root_dir, scene_name, f'{stem_name_1}_depth.tiff'))
        else:
            depth0 = depth1 = torch.tensor([])

        # read the intrinsic of depthmap
        K_0 = K_1 = torch.tensor(self.intrinsics.copy(), dtype=torch.float).reshape(3, 3)

        # read and compute relative poses
        T_0to1 = torch.tensor(self._compute_rel_pose(scene_name, stem_name_0, stem_name_1),
                              dtype=torch.float32)
        T_1to0 = T_0to1.inverse()

        data = {
            'image0': image0,   # (1, h, w)
            'depth0': depth0,   # (h, w)
            'image1': image1,
            'depth1': depth1,
            'T_0to1': T_0to1,   # (4, 4)
            'T_1to0': T_1to0,
            'K0': K_0,  # (3, 3)
            'K1': K_1,
            'dataset_name': 'C3VD',
            'scene_id': scene_name,
            'pair_id': idx,
            'pair_names': (osp.join(scene_name, f'{stem_name_0}_color.png'),
                           osp.join(scene_name, f'{stem_name_1}_color.png'))
        }

        return data
