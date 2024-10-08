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
    read_scannet_pose,
    read_scannet_intrinsic,
    sample_homography_np
)
import cv2
import os


class ScanNetDataset(utils.data.Dataset):
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
        with np.load(npz_path) as data:
            self.data_names = data['name']
            if 'score' in data.keys() and mode not in ['val' or 'test']:
                kept_mask = data['score'] > min_overlap_score
                self.data_names = self.data_names[kept_mask]
        self.intrinsics = dict(np.load(intrinsic_path))

        # for training LoFTR
        self.augment_fn = augment_fn if mode == 'train' else None

    def __len__(self):
        return len(self.data_names)

    def _read_abs_pose(self, scene_name, name):
        pth = osp.join(self.pose_dir,
                       scene_name,
                       'pose', f'{name}.txt')
        return read_scannet_pose(pth)

    def _compute_rel_pose(self, scene_name, name0, name1):
        pose0 = self._read_abs_pose(scene_name, name0)
        pose1 = self._read_abs_pose(scene_name, name1)
        
        return np.matmul(pose1, inv(pose0))  # (4, 4)

    def __getitem__(self, idx):
        data_name = self.data_names[idx]
        scene_name, scene_sub_name, stem_name_0, stem_name_1 = data_name
        scene_name = f'scene{scene_name:04d}_{scene_sub_name:02d}'
        
        if not self.homo:
            # read the grayscale image which will be resized to (1, 480, 640)
            img_name0 = osp.join(self.root_dir, scene_name, 'color', f'{stem_name_0}.jpg')
            img_name1 = osp.join(self.root_dir, scene_name, 'color', f'{stem_name_1}.jpg')
            
            # TODO: Support augmentation & handle seeds for each worker correctly.
            image0 = read_scannet_gray(img_name0, resize=(640, 480), augment_fn=None)
                                    #    augment_fn=np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))
            image1 = read_scannet_gray(img_name1, resize=(640, 480), augment_fn=None)
                                    #    augment_fn=np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))

            # read the intrinsic of depthmap
            K_0 = K_1 = torch.tensor(self.intrinsics[scene_name].copy(), dtype=torch.float).reshape(3, 3)

            # read the depthmap which is stored as (480, 640)

            depth0, hha0 = read_scannet_depth(osp.join(self.root_dir, scene_name, 'depth', f'{stem_name_0}.png'), self.intrinsics[scene_name].copy(),cross_modal=self.cross_modal)
            depth1, hha1 = read_scannet_depth(osp.join(self.root_dir, scene_name, 'depth', f'{stem_name_1}.png'), self.intrinsics[scene_name].copy(),cross_modal=self.cross_modal)

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
                'dataset_name': 'ScanNet',
                'scene_id': scene_name,
                'pair_id': idx,
                'pair_names': (f'{scene_name}_{stem_name_0}',
                            f'{scene_name}_{stem_name_1}'),
            }

            if hha0 is not None:
                data.update({'hha0': hha0, 'hha1': hha1})

            return data
        else:
            np.random.seed(torch.randint(100,(1,))+idx)
            # read the grayscale image which will be resized to (1, 480, 640)
            img_name0 = osp.join(self.root_dir, scene_name, 'color', f'{stem_name_0}.jpg')
            
            image0 = read_scannet_gray(img_name0, resize=(640, 480), augment_fn=None, homo=True)

            #for edge from depth
            
            sobelx = cv2.Sobel(image0, cv2.CV_16S, 1, 0, ksize=-1)  
            sobely = cv2.Sobel(image0, cv2.CV_16S, 0, 1, ksize=-1)  
            abx = cv2.convertScaleAbs(sobelx) 
            aby = cv2.convertScaleAbs(sobely)  
            dst = cv2.addWeighted(abx, 0.5, aby, 0.5, 0)  

            blurred_edges = dst
            blurred_edges = cv2.GaussianBlur(blurred_edges, (7, 7), 0)
            blurred_edges = blurred_edges>np.quantile(blurred_edges, 0.85)
            mask0 = blurred_edges.astype(np.uint8) * 255
            
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
            K_0 = K_1 = torch.tensor(self.intrinsics[scene_name].copy(), dtype=torch.float).reshape(3, 3)

            # # read the depthmap which is stored as (480, 640)
            # if self.mode in ['train', 'val']:
            #     depth0, hha0 = read_scannet_depth(osp.join(self.root_dir, scene_name, 'depth', f'{stem_name_0}.png'), self.intrinsics[scene_name].copy(),cross_modal=self.cross_modal)
            #     depth1, hha1 = read_scannet_depth(osp.join(self.root_dir, scene_name, 'depth', f'{stem_name_1}.png'), self.intrinsics[scene_name].copy(),cross_modal=self.cross_modal)
            # else:
            #     depth0 = depth1 = torch.tensor([])

            # # read and compute relative poses
            # T_0to1 = torch.tensor(self._compute_rel_pose(scene_name, stem_name_0, stem_name_1),
            #                     dtype=torch.float32)
            # T_1to0 = T_0to1.inverse()

            data = {
                'image0': image0,   # (1, h, w)
                'image1': image1,
                'K0': K_0,  # (3, 3)
                'K1': K_1,
                'mat': torch.tensor(mat, dtype=torch.float32),
                'dataset_name': 'ScanNet',
                'scene_id': scene_name,
                'pair_id': idx,
                'pair_names': (f'{scene_name}_{stem_name_0}',
                            f'{scene_name}_{stem_name_1}'),
            }

            # if self.mode in ['train', 'val'] and hha0 is not None:
            #     data.update({'hha0': hha0, 'hha1': hha1})

            if mask0 is not None:  # img_padding is True
                [ts_mask_0, ts_mask_1] = F.interpolate(torch.stack([mask0, mask1], dim=0)[None].float(),
                                                    scale_factor=0.125,
                                                    mode='nearest',
                                                    recompute_scale_factor=False)[0].bool()
                # data.update({'mask0': ts_mask_0, 'mask1': ts_mask_1})
                data.update({'mask0_spv': ts_mask_0, 'mask1_spv': ts_mask_1})
                # cv2.imwrite(osp.join('test', f'{scene_name}_{stem_name_0}.jpg'), np.concatenate([cv2.resize(image0[0].numpy()*255,(80,60)),ts_mask_0.numpy().astype(np.uint8)*255],axis=0))

            return data