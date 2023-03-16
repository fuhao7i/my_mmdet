import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.builder import BACKBONES
from mmdet.models.utils import ResLayer
from mmdet.models.backbones.resnet import ResNet

import sys
sys.path.append('@fuhao/')
# from custom_mmdet.models.backbones.stem.scene_mining import *

import torch.nn.functional as F

import seaborn as sns
sns.set(font_scale=1.5)

from torchvision.utils import save_image

# from custom_mmdet.models.backbones.resnet.resnet18 import ResNet18_FPN

import sys
sys.path.append('@fuhao/')
# from custom_mmdet.models.backbones.stem.scene_mining import *

import torch.nn.functional as F

import seaborn as sns
sns.set(font_scale=1.5)

from torchvision.utils import save_image

import torchvision
import numpy as np
from torchvision.ops import DeformConv2d

class DCNv2(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride, padding):
        super(DCNv2, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1) #原卷积
 
        self.conv_offset = nn.Conv2d(in_c, 2 * kernel_size * kernel_size, kernel_size=3, stride=1, padding=1)
        init_offset = torch.Tensor(np.zeros([2 * kernel_size * kernel_size, in_c, 3, 3]))
        self.conv_offset.weight = torch.nn.Parameter(init_offset) #初始化为0
 
        self.conv_mask = nn.Conv2d(in_c, kernel_size * kernel_size, kernel_size=3, stride=1, padding=1)
        init_mask = torch.Tensor(np.zeros([kernel_size * kernel_size, in_c, 3, 3])+np.array([0.5]))
        self.conv_mask.weight = torch.nn.Parameter(init_mask) #初始化为0.5

        self.dcn =  DeformConv2d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
 
    def forward(self, x):
        offset = self.conv_offset(x)
        mask = torch.sigmoid(self.conv_mask(x)) #保证在0到1之间
        # out = torchvision.ops.deform_conv2d(input=x, offset=offset, 
        #                                     weight=self.conv.weight,
        #                                      mask=mask, padding=(1, 1))
        out = self.dcn(x, offset,  mask=mask)
        return out

if __name__ == "__main__":
    
    # model = ResNet_v328(50)
    model = DCNv2(256, 64, 3, 1, 1)
    x = torch.randn((2, 256, 224, 224))
    x = model(x)
    print(x.shape)
