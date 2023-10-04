from os import path as osp
from typing import Dict
from unicodedata import name

import numpy as np
import torch
import torch.utils as utils
from numpy.linalg import inv
import torch.nn.functional as F
from src.utils.dataset import (
    read_scannet_gray,
    read_scannet_depth,
    read_c3vd_gray,
    read_scannet_pose,
    read_scannet_intrinsic,
    read_c3vd_pose,
    read_c3vd_depth,
    read_c3vd_depth_mask,
    sample_homography_np
)
import cv2
import os


class C3VD_UNDISTORT_Dataset(utils.data.Dataset):
    def __init__(self,
                 root_dir,
                 npz_path,
                 intrinsic_path,
                 mode='train',
                 cross_modal=False,
                 min_overlap_score=0.4,
                 augment_fn=None,
                 pose_dir=None,
                 homo = False,
                 homo_param = None,
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
        self.cross_modal = cross_modal
        self.homo = homo
        self.homo_param = homo_param

        # prepare data_names, intrinsics and extrinsics(T)
        if not self.homo:
            with np.load(npz_path) as data:
                self.data_names = data['name']
                if 'score' in data.keys() and mode not in ['val' or 'test']:
                    kept_mask = data['score'] > min_overlap_score
                    self.data_names = self.data_names[kept_mask]
        else:
            scene = npz_path.split('/')[-1].split('.')[0]
            self.data_names = os.listdir(osp.join(self.root_dir,scene))
            self.data_names = [[scene, file.split('_')[0], file.split('_')[0]] for file in self.data_names if file.endswith(("_color.png", "_color.jpg"))]
        
        with open(intrinsic_path, 'r') as f:
            lines = f.readlines()

        self.intrinsics = np.asarray(lines[0].split(), dtype=np.float32)
        self.coarse_scale = getattr(kwargs, 'coarse_scale', 0.125)
        # for training LoFTR
        self.augment_fn = augment_fn if mode == 'train' else None

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
        
        if not self.homo:
            # read the grayscale image which will be resized to (1, 480, 640)
            img_name0 = osp.join(self.root_dir, scene_name, f'{stem_name_0}_color.png')
            img_name1 = osp.join(self.root_dir, scene_name, f'{stem_name_1}_color.png')
            
            # TODO: Support augmentation & handle seeds for each worker correctly.
            image0, _ = read_c3vd_gray(img_name0, resize=(640, 480), augment_fn=None, with_mask=False)
                                    #    augment_fn=np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))
            image1, _ = read_c3vd_gray(img_name1, resize=(640, 480), augment_fn=None, with_mask=False)
                                    #    augment_fn=np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))

            # read the intrinsic of depthmap
            K_0 = K_1 = torch.tensor(self.intrinsics, dtype=torch.float).reshape(3, 3)

            # read and compute relative poses
            T_0to1 = torch.tensor(self._compute_rel_pose(scene_name, stem_name_0, stem_name_1),
                                dtype=torch.float32)
            T_1to0 = T_0to1.inverse()

            data = {
                'image0': image0,   # (1, h, w)
                'image1': image1,
                'T_0to1': T_0to1,   # (4, 4)
                'T_1to0': T_1to0,
                'K0': K_0,  # (3, 3)
                'K1': K_1,
                'dataset_name': 'C3VD_undistort',
                'scene_id': scene_name,
                'pair_id': idx,
                'pair_names': (f'{scene_name}_{stem_name_0}',
                            f'{scene_name}_{stem_name_1}'),
            }
            return data
        else:
            np.random.seed(torch.randint(100,(1,))+idx)
            # read the grayscale image which will be resized to (1, 480, 640)
            img_name0 = osp.join(self.root_dir, scene_name, f'{stem_name_0}_color.png')
            
            image0, _ = read_c3vd_gray(img_name0, resize=(640, 480), augment_fn=None, with_mask=False, return_np = True)
            depth0 = read_c3vd_depth_mask(osp.join(self.root_dir, scene_name, f'{stem_name_0}_depth.tiff'))
            mask0 = depth0.astype(np.uint8) * 255

            mat = sample_homography_np(np.array(image0.shape),
                                    shift=0, perspective=True, scaling=True, rotation=True, translation=True,
                                    n_scales=5, n_angles=25, scaling_amplitude=self.homo_param["scale"],
                                    perspective_amplitude_x=self.homo_param["perspective"],
                                    perspective_amplitude_y=self.homo_param["perspective"],
                                    patch_ratio=self.homo_param["patch_ratio"],
                                    max_angle=self.homo_param["rotation"],
                                    allow_artifacts=False,
                                    translation_overflow=0.
                                    )
            mat = np.linalg.inv(mat)
            image1 = cv2.warpPerspective(image0, mat, (image0.shape[1], image0.shape[0]))
            mask1 = cv2.warpPerspective(mask0, mat, (mask0.shape[1], mask0.shape[0]))

            image0 = torch.from_numpy(image0).float()[None] / 255
            image1 = torch.from_numpy(image1).float()[None] / 255
            mask0 = torch.from_numpy(mask0==255)
            mask1 = torch.from_numpy(mask1==255)
            # read the intrinsic of depthmap
            K_0 = K_1 = torch.tensor(self.intrinsics, dtype=torch.float).reshape(3, 3)

            data = {
                'image0': image0,   # (1, h, w)
                'image1': image1,
                'K0': K_0,  # (3, 3)
                'K1': K_1,
                'mat': torch.tensor(mat, dtype=torch.float32),
                'dataset_name': 'C3VD_undistort',
                'scene_id': scene_name,
                'pair_id': idx,
                'pair_names': (f'{scene_name}_{stem_name_0}',
                            f'{scene_name}_{stem_name_1}'),
            }

            # # for LoFTR training
            # if mask0 is not None:  # img_padding is True
            #     if self.coarse_scale:
            #         [ts_mask_0, ts_mask_1] = F.interpolate(torch.stack([mask0, mask1], dim=0)[None].float(),
            #                                             scale_factor=self.coarse_scale,
            #                                             mode='nearest',
            #                                             recompute_scale_factor=False)[0].bool()
            #     # data.update({'mask0': ts_mask_0, 'mask1': ts_mask_1})
            #     data.update({'mask0_spv': ts_mask_0, 'mask1_spv': ts_mask_1})

            return data