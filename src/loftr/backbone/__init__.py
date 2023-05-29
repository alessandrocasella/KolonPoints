from .resnet_fpn import ResNetFPN_8_2, ResNetFPN_16_4, ResNetFPN_8_2_cross
from .dual_segformer import mit_b4
import torch.nn as nn
def build_backbone(config):
    if config['backbone_type'] == 'ResNetFPN':
        if config['resolution'] == (8, 2):
            return ResNetFPN_8_2(config['resnetfpn'])
        elif config['resolution'] == (16, 4):
            return ResNetFPN_16_4(config['resnetfpn'])
    elif config['backbone_type'] == 'CrossModal':
        # return mit_b4(norm_fuse=nn.BatchNorm2d)     
        return  ResNetFPN_8_2_cross(config['resnetfpn'])   
    else:
        raise ValueError(f"LOFTR.BACKBONE_TYPE {config['backbone_type']} not supported.")
