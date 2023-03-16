# from ..builder import DETECTORS
# from .single_stage import SingleStageDetector
from cProfile import label
from locale import normalize
from tkinter import E
from mmdet.core.bbox.transforms import bbox_mapping
import torch
import torch.nn as nn
from mmdet.core import bbox2result
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.single_stage import SingleStageDetector

import sys
sys.path.append('@fuhao/')
# from custom_mmdet.models.backbones.resnet.resnet18 import ResNet18_FPN

import random
from torchvision.utils import save_image
import numpy as np

import os
import torch
import numpy as np
from PIL import Image, ImageFilter
from torchvision.transforms import ToTensor

# ----------------------------------------
# FPN
import torch
import warnings
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
from mmcv.runner import auto_fp16
import torch.nn as nn
from torch.nn import BatchNorm2d
from mmcv.cnn import (ConvModule, caffe2_xavier_init, constant_init, is_norm,
                      normal_init)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class Unet_Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        if x2 != None:
            # input is CHW
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            # if you have padding issues, see
            # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
            # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
            x = torch.cat([x2, x1], dim=1)
            x = self.conv(x)
        else:
            x = self.conv(x1)
        return x


class Unet_Decoder(nn.Module):
    def __init__(self, input_list=None):
        super(Unet_Decoder, self).__init__()
        
        self.up5 = Unet_Up(3072, 1024)
        self.up4 = Unet_Up(1536, 512)
        self.up3 = Unet_Up(768, 256)
        self.up2 = Unet_Up(256, 128)
        self.up1 = Unet_Up(128, 64)
        self.out = nn.Conv2d(64, 3, 3, 1, 1)
        self.act = nn.Sigmoid()
    def forward(self, x):
        C_2, C_3, C_4, C_5 = x
        
        x = self.up5(C_5, C_4)
        x = self.up4(x, C_3)
        x = self.up3(x, C_2)
        x = self.up2(x)
        x = self.up1(x)
        x = self.out(x)
        x = self.act(x)
        return x      


class TNet(torch.nn.Module):
    def __init__(self, num=64):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(3, num, 3, 1, 0),
            torch.nn.InstanceNorm2d(num),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(num, num, 3, 1, 0),
            torch.nn.InstanceNorm2d(num),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(num, num, 3, 1, 0),
            torch.nn.InstanceNorm2d(num),
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(num, num, 3, 1, 0),
            torch.nn.InstanceNorm2d(num),
            torch.nn.ReLU()
        )
        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(num, 3, 1, 1, 0),
            torch.nn.Sigmoid()
        )

    def forward(self, data):
        data = self.conv1(data)
        data = self.conv2(data)
        data = self.conv3(data)
        data = self.conv4(data)
        data1 = self.final(data)

        return data1



import torch.nn as nn
class Upsampler(nn.Module):
    def __init__(self, n_feat, out_feat, act=False, bias=True):
        super(Upsampler, self).__init__()
        # self.conv0 = nn.Conv2d(n_feat, 4 * n_feat, 3, padding= 1)
        # self.ps = nn.PixelShuffle(2)
        self.conv0 = nn.Conv2d(n_feat, n_feat, 3, 1, 1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(n_feat, out_feat, 3, 1, 1)
        self.bn = nn.BatchNorm2d(n_feat)
        self.act = nn.ReLU(True)
    def forward(self, x):
        # x = self.conv0(x)
        # x = self.ps(x)
        x = self.conv0(x)
        x = self.bn(x)
        x = self.act(x)
        x  = self.up(x)
        x = self.conv1(x)
        # x = self.bn(x)
        # x = self.act(x)
        return x
    
class CycleGAN_Upsampler(nn.Module):
    def __init__(self, n_feat, out_feat, act=False, bias=True):
        super(CycleGAN_Upsampler, self).__init__()
        
        self.conv0 = nn.ConvTranspose2d(n_feat, out_feat,
                                        kernel_size=4, stride=2,
                                        padding=1, output_padding=1,
                                        bias=True)
        self.bn = nn.BatchNorm2d(out_feat)
        self.act = nn.ReLU(True)
        
    def forward(self, x):
        x = self.conv0(x)
        x = self.bn(x)
        x = self.act(x)
        return x
    
class Cycle_Decoder(nn.Module):
    def __init__(self, input_list=None):
        super(Cycle_Decoder, self).__init__()
        
        self.up5 = CycleGAN_Upsampler(2048, 1024)
        self.up4 = CycleGAN_Upsampler(2048, 512)
        self.up3 = CycleGAN_Upsampler(1024, 256)
        self.up2 = CycleGAN_Upsampler(512, 128)
        self.up1 = CycleGAN_Upsampler(128, 64)
        self.out = nn.Conv2d(64, 3, 3, 1, 1)
    def forward(self, x):
        #256,512,1024,2048
        C_2, C_3, C_4, C_5 = x
        
        x = self.up5(C_5)
        # x = x + C_4
        x = torch.cat((x, C_4), 1)
        x = self.up4(x)
        # x = x + C_3
        x = torch.cat((x, C_3), 1)
        x = self.up3(x)
        # x = x + C_2
        x = torch.cat((x, C_2), 1)
        x = self.up2(x)
        
        x = self.up1(x)
        x = self.out(x)
        return x   
        
    

class Decoder(nn.Module):
    def __init__(self, input_list=None):
        super(Decoder, self).__init__()
        
        self.up5 = Upsampler(2048, 1024)
        self.up4 = Upsampler(1024, 512)
        self.up3 = Upsampler(512, 256)
        self.up2 = Upsampler(256, 128)
        self.up1 = Upsampler(128, 3)
        self.act = nn.Sigmoid()
    def forward(self, x):
        C_2, C_3, C_4, C_5 = x
        
        x = self.up5(C_5)
        x = x + C_4
        # x = torch.cat((x, C_4), 1)
        x = self.up4(x)
        x = x + C_3
        # x = torch.cat((x, C_3), 1)
        x = self.up3(x)
        x = x + C_2
        # x = torch.cat((x, C_2), 1)
        x = self.up2(x)
        
        x = self.up1(x)
        x = self.act(x)
        return x     
    

class Decoder_cat(nn.Module):
    def __init__(self, input_list=None):
        super(Decoder_cat, self).__init__()
        
        self.up5 = Upsampler(2048, 1024)
        self.up4 = Upsampler(2048, 512)
        self.up3 = Upsampler(1024, 256)
        self.up2 = Upsampler(512, 128)
        self.up1 = Upsampler(128, 3)
        self.act = nn.Sigmoid()
    def forward(self, x):
        C_2, C_3, C_4, C_5 = x
        
        x = self.up5(C_5)
        # x = x + C_4
        x = torch.cat((x, C_4), 1)
        x = self.up4(x)
        # x = x + C_3
        x = torch.cat((x, C_3), 1)
        x = self.up3(x)
        # x = x + C_2
        x = torch.cat((x, C_2), 1)
        x = self.up2(x)
        
        x = self.up1(x)
        x = self.act(x)
        return x    
     


class FPN_Decoder(nn.Module):
    # def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
    #     super(FPN_torch, self).__init__()
    def __init__(self):
        super(FPN_Decoder, self).__init__()

        C2_size, C3_size, C4_size, C5_size = [256, 512, 1024, 2048]
        feature_size = 256
        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='bilinear')
        # self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='bilinear')
        # self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='bilinear')
        # self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P2_1 = nn.Conv2d(C2_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P2_upsampled = nn.Upsample(scale_factor=2, mode='bilinear')
        self.P2_2 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),)
        
        self.stem0 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
            )
        self.stem0_up = nn.Upsample(scale_factor=2, mode='bilinear')

        
        self.out = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        self.act = nn.Sigmoid()
  

    def forward(self, inputs, stem0, j_out_x):
        C2, C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        # P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        # P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_upsampled_x = self.P3_upsampled(P3_x)
        # P3_x = self.P3_2(P3_x)

        P2_x = self.P2_1(C2)
        P2_x = P2_x + P3_upsampled_x
        P2_up = self.P2_upsampled(P2_x)
        P2_up = self.P2_2(P2_up)
        
        stem0 = stem0 + P2_up
        stem0_x = self.stem0(stem0)
        
        
        stem0_up = self.stem0_up(stem0)
        
        j_out_x = j_out_x + stem0_up
        
        out = self.out(j_out_x)
        out = self.act(out)

        return out
    
    
class FPN_Decoder21(nn.Module):
    # def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
    #     super(FPN_torch, self).__init__()
    def __init__(self):
        super(FPN_Decoder21, self).__init__()

        C2_size, C3_size, C4_size, C5_size = [256, 512, 1024, 2048]
        feature_size = 256
        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='bilinear')
        self.P5_2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        # self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='bilinear')
        self.P4_2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        # self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='bilinear')
        self.P3_2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        # self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P2_1 = nn.Conv2d(C2_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P2_upsampled = nn.Upsample(scale_factor=2, mode='bilinear')
        self.P2_2 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
            )
        
        self.stem0 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
            )
        self.stem0_up = nn.Upsample(scale_factor=2, mode='bilinear')

        
        self.out = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        self.act = nn.Sigmoid()
  

    def forward(self, inputs, stem0):
        C2, C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_upsampled_x = self.P5_2(P5_upsampled_x)
        # P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_upsampled_x = self.P4_2(P4_upsampled_x)
        # P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_upsampled_x = self.P3_upsampled(P3_x)
        P3_upsampled_x = self.P3_2(P3_upsampled_x)
        # P3_x = self.P3_2(P3_x)

        P2_x = self.P2_1(C2)
        P2_x = P2_x + P3_upsampled_x
        P2_up = self.P2_upsampled(P2_x)
        P2_up = self.P2_2(P2_up)
        
        stem0 = stem0 + P2_up
        
        # stem0_x = self.stem0(stem0)
        
        
        # stem0_up = self.stem0_up(stem0)
        
        # j_out_x = j_out_x + stem0_up
        
        out = self.out(stem0)
        out = self.act(out)

        return out
    

class FPN_DecoderH4(nn.Module):
    # def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
    #     super(FPN_torch, self).__init__()
    def __init__(self):
        super(FPN_DecoderH4, self).__init__()

        C2_size, C3_size, C4_size, C5_size = [256, 512, 1024, 2048]
        feature_size = 256
        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='bilinear')
        # self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='bilinear')
        # self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='bilinear')
        # self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P2_1 = nn.Conv2d(C2_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P2_upsampled = nn.Upsample(scale_factor=2, mode='bilinear')
        self.P2_2 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),)
        
        self.stem0 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
            )
        self.stem0_up = nn.Upsample(scale_factor=2, mode='bilinear')

        
        self.out = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        self.act = nn.Sigmoid()
  

    def forward(self, inputs, stem0, j_out_x):
        C2, C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        # P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        # P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_upsampled_x = self.P3_upsampled(P3_x)
        # P3_x = self.P3_2(P3_x)

        P2_x = self.P2_1(C2)
        P2_x = P2_x + P3_upsampled_x
        P2_up = self.P2_upsampled(P2_x)
        P2_up = self.P2_2(P2_up)
        
        stem0 = stem0 + P2_up
        stem0_x = self.stem0(stem0)
        
        
        stem0_up = self.stem0_up(stem0)
        
        j_out_x = j_out_x + stem0_up
        
        out = self.out(j_out_x)
        out = self.act(out)

        return out, P3_x



class FPN_github_torch(nn.Module):
    # def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
    #     super(FPN_torch, self).__init__()
    def __init__(self):
        super(FPN_github_torch, self).__init__()
        
        # self.j_net = JNet()

        C2_size, C3_size, C4_size, C5_size = [256, 512, 1024, 2048]
        feature_size = 256
        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        
        self.P2_1 = nn.Conv2d(C2_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P2_upsampled = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.P2_2 = nn.Conv2d(feature_size, 64, kernel_size=3, stride=1, padding=1)
        self.P2_3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        
        self.stem2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.stem1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.stem0 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.P1_1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.P1_upsampled = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.P1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        self.out = nn.Conv2d(32, 3, 3, 1, 1)
        self.act = nn.Sigmoid()
  
        # self.init_weights()
    # def init_weights(self):
    #     from mmcv.cnn import ConvModule, xavier_init
    #     from timm.models.layers import DropPath, to_2tuple, trunc_normal_
    #     from mmcv.cnn import (ConvModule, caffe2_xavier_init, constant_init, is_norm,
    #                 normal_init)
    #     from torch.nn.modules.batchnorm import _BatchNorm
    #     from mmcv.cnn import constant_init, kaiming_init
    #     """Initialize the weights of FPN module."""
    #     for m in self.modules():
    #     #     if isinstance(m, nn.Conv2d):
    #     #         xavier_init(m, distribution='uniform')
            
    #     # for m in self.fm5.modules():
    #         if isinstance(m, nn.Conv2d):
    #             kaiming_init(m)
    #         elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
    #             constant_init(m, 1)
    #         elif isinstance(m, nn.Linear):
    #             trunc_normal_(m.weight, std=.02)
    #             if isinstance(m, nn.Linear) and m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.LayerNorm):
    #             nn.init.constant_(m.bias, 0)
    #             nn.init.constant_(m.weight, 1.0)

    def forward(self, inputs, stem0, j_out_x):
        
        # stem0, stem1, stem2
        #   2      2     4
        #4,  8, 16, 32
        C2, C3, C4, C5 = inputs
        
        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        # 16
        #========================================
        
        P4_x = self.P4_1(C4)
        P4_x = P4_x + P5_upsampled_x
        P4_x = self.P4_2(P4_x)
        
        P4_upsampled_x = self.P4_upsampled(P4_x)
        # 8
        #========================================
        
        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)
        
        P3_upsampled_x = self.P3_upsampled(P3_x)
        # 4
        #========================================

        P2_x = self.P2_1(C2)
        P2_x = P2_x + P3_upsampled_x
        P2_x = self.P2_2(P2_x)
        
        # stem2 = P2_x + stem2
        # stem2 = self.stem2(stem2)
        
        P2_upsampled_x = self.P2_upsampled(P2_x)
        
        P2_upsampled_x = self.P2_3(P2_upsampled_x)
        # 2
        #========================================
        stem0 = stem0 + P2_upsampled_x
        stem0 = self.stem1(stem0)
        
        # stem0 = stem0 + stem1
        # stem0 = self.stem0(stem0)
        
        stem0 = self.P1_upsampled(stem0)

        out = self.out(stem0)
        out = self.act(out)
        
        

        # P5_x = self.P5_1(C5)
        # P5_upsampled_x = self.P5_upsampled(P5_x)
        # # P5_x = self.P5_2(P5_x)

        # P4_x = self.P4_1(C4)
        # P4_x = P5_upsampled_x + P4_x
        # P4_upsampled_x = self.P4_upsampled(P4_x)
        # # P4_x = self.P4_2(P4_x)

        # P3_x = self.P3_1(C3)
        # P3_x = P3_x + P4_upsampled_x
        # P3_upsampled_x = self.P3_upsampled(P3_x)
        # # P3_x = self.P3_2(P3_x)


        # P2_x = self.P2_1(C2)
        # P2_x = P2_x + P3_upsampled_x
        # # P2_upsampled_x = self.P2_upsampled(P2_x)
        # P2_x = self.P2_2(P2_x)
        
        # P2_x = P2_x + stem2
        
        
        
        
        # P1_x = self.P1_1(stem2)
        # P1_x = P1_x + P2_upsampled_x
        # P1_upsampled_x = self.P1_upsampled(P1_x)

        # out = self.out(P1_upsampled_x)
        # out = self.act(out)
        return out

class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)
        
class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
class Unet_Decoder(nn.Module):
    # def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
    #     super(FPN_torch, self).__init__()
    def __init__(self):
        super(Unet_Decoder, self).__init__()

        C2_size, C3_size, C4_size, C5_size = [256, 512, 1024, 2048]
        feature_size = 256
        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.UpsamplingBilinear2d(scale_factor=2)
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        # self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        # self.P4_upsampled = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        
        self.P4_1 = DecoderBlock(1024, 512, 512) # C4, C3
        
        self.P3_1 = DecoderBlock(512, 256, 256) # C3, C2
        
        self.P2_1 = DecoderBlock(256, 64, 64) # C2, stem2
        
        self.P1_1 = DecoderBlock(256, 32, 32) # stem2, stem1
        
        # self.P0_1 = DecoderBlock(32, 3)

        # add P4 elementwise to C3
        # self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        # self.P3_upsampled = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        
        # self.P2_1 = nn.Conv2d(C2_size, feature_size, kernel_size=1, stride=1, padding=0)
        # self.P2_upsampled = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.P2_2 = nn.Conv2d(feature_size, 64, kernel_size=3, stride=1, padding=1)
        # self.P2_3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        
        # self.stem2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # self.stem1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        # self.stem0 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        # self.P1_1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        # self.P1_upsampled = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.P1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.out_upsampled = nn.UpsamplingBilinear2d(scale_factor=2)
        self.out = nn.Conv2d(32, 3, 3, 1, 1)
        # self.act = nn.Sigmoid()

    def forward(self, inputs, stem0, stem1, stem2):
        # stem0, stem1, stem2
        #   2      2     4
        #4,  8, 16, 32
        C2, C3, C4, C5 = inputs
        #256, 512, 1024, 2048
        
        # P4_x = self.P4_1(C4, C3)
        P3_x = self.P3_1(C3, C2)
        # P2_x = self.P2_1(P3_x, stem2)
        P1_x = self.P1_1(P3_x, stem1)
        
        P1_x = self.out_upsampled(P1_x)
        out = self.out(P1_x)
        
        return out

   

import torch
from torch import nn


class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()

    def forward(self, x):
        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        Dr = torch.pow(mr-0.5, 2)
        Dg = torch.pow(mg-0.5, 2)
        Db = torch.pow(mb-0.5, 2)
        k = torch.pow(torch.pow(Dr, 2) + torch.pow(Dg, 2) + torch.pow(Db, 2), 0.5)
        return k


from .homology_utils.get_a import get_A
@DETECTORS.register_module()
class RetinaNet_H1(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 bbox_head_1=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RetinaNet_H1, self).__init__(backbone, neck, bbox_head, bbox_head_1, train_cfg,
                                        test_cfg, pretrained)
        
        # self.net_decoder = Decoder().cuda()
        # self.net_decoder = Cycle_Decoder().cuda()
        # self.net_decoder = Unet_Decoder().cuda()
        # self.net_decoder = Decoder_cat().cuda()
        self.net_decoder = FPN_github_torch()
        self.net_t = TNet().cuda()
        self.mse_loss = torch.nn.MSELoss().cuda()
        self.color_loss = ColorLoss().cuda()

    def extract_feat_test(self, img, img_metas):
        """Directly extract features from the backbone+neck."""
        x, stem0, stem1, stem2 = self.backbone(img) 
        # ====================================================
        # Backbone + Decoder = Generator
        j_out = self.net_decoder(x, stem0, stem1, stem2)
        save_image(j_out[0], '@exp_homology/10_urpc/11/results/' + img_metas[0]['filename'].split('/')[-1], normalize=True)
        # ====================================================
        
        
        if self.with_neck:
            x = self.neck(x)
        return x
        
    def extract_feat(self, img, img_metas):
        """Directly extract features from the backbone+neck."""
        # print(img_metas)
        # print(img_metas[0]['filename'].split('/')[-1], img_metas[1]['filename'].split('/')[-1])
        x, stem0, stem1, stem2 = self.backbone(img)
        # ======================================================
        # Homology loss
        # ======================================================
        j_out = self.net_decoder(x, stem0, stem1, stem2)
        t_out = self.net_t(img)
        
        a_out1 = get_A(img[0].unsqueeze(0)).cuda()
        a_out2 = get_A(img[1].unsqueeze(0)).cuda()
        a_out = torch.cat([a_out1, a_out2], 0)
        
        I_rec = j_out * t_out + (1 - t_out) * a_out
        loss_1 = self.mse_loss(I_rec, img)
        # -------------------------------------------------------
        lam = np.random.beta(1, 1)
        input_mix = lam * img + (1 - lam) * j_out

        x_mix, stem0, stem1, stem2 = self.backbone(input_mix)
        j_out_mix = self.net_decoder(x_mix, stem0, stem1, stem2)
        # t_out_mix = self.net_t(input_mix)

        loss_2 = self.mse_loss(j_out_mix, j_out.detach())

        loss_3 = self.color_loss(j_out)

        total_loss = 1 * loss_1 + 1 * loss_2 + 0.01 * loss_3
        total_loss = total_loss 
        # save_image(img[0], '@fuhao/transfer/img.jpg', normalize=True)
        # save_image(j_out[0], '@fuhao/transfer/img_gen.jpg', normalize=True)
        # save_image(a_out[0], '@fuhao/transfer/a_out.jpg', normalize=True)
        # save_image(I_rec[0], '@fuhao/transfer/I_rec.jpg', normalize=True)
        # -------------------------------------------------------
        
        
        # if self.with_neck:
        #     x = self.neck(x)
        x = img
        return x, total_loss

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        
        x, loss_h = self.extract_feat(img, img_metas)
        # x, en_neck, loss_label_guide, label_guide_x = self.extract_feat_label_guide(img, img_metas, gt_bboxes)

        losses = {}
        # losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
        #                                       gt_labels, gt_bboxes_ignore, 
        #                                       label_mask=None
        #                                         )
        
        # en_losses = self.bbox_head_1.forward_train(en_neck, img_metas, gt_bboxes,
        #                                       gt_labels, gt_bboxes_ignore, 
        #                                       label_mask=label_guide_x
        #                                         )
        
        # losses.update({'loss_en_neck_cls':en_losses['loss_cls']})
        # losses.update({'loss_en_neck_bbox':en_losses['loss_bbox']})
        losses.update({'loss_homology':loss_h})
        
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat_test(img, img_metas)
        # x, en_neck, label_guide_x = self.test_extract_feat_label_guide(img)
        
        outs = self.bbox_head(x)
        # get origin input shape to support onnx dynamic shape
        if torch.onnx.is_in_onnx_export():
            # get shape as tensor
            img_shape = torch._shape_as_tensor(img)[2:]
            img_metas[0]['img_shape_for_onnx'] = img_shape
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        # skip post-processing when exporting to ONNX
        if torch.onnx.is_in_onnx_export():
            return bbox_list

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results


@DETECTORS.register_module()
class RetinaNet_H2(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 bbox_head_1=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RetinaNet_H2, self).__init__(backbone, neck, bbox_head, bbox_head_1, train_cfg,
                                        test_cfg, pretrained)
        
        # self.net_decoder = Decoder().cuda()
        # self.net_decoder = Cycle_Decoder().cuda()
        # self.net_decoder = Unet_Decoder().cuda()
        # self.net_decoder = Decoder_cat().cuda()
        # self.net_decoder = FPN_github_torch()
        # self.net_decoder = Unet_Decoder().cuda()
        self.net_decoder = FPN_Decoder()
        self.net_t = TNet().cuda()
        self.mse_loss = torch.nn.MSELoss().cuda()
        self.color_loss = ColorLoss().cuda()

    def extract_feat_test(self, img, img_metas):
        """Directly extract features from the backbone+neck."""
        x, stem0, j_out_x = self.backbone(img) 
        # ====================================================
        # Backbone + Decoder = Generator
        j_out = self.net_decoder(x, stem0, j_out_x)
        save_image(j_out[0], '@exp_homology/10_urpc/12/results/' + img_metas[0]['filename'].split('/')[-1], normalize=True)
        # ====================================================
        
        
        if self.with_neck:
            x = self.neck(x)
        return x
        
    def extract_feat(self, img, img_metas):
        """Directly extract features from the backbone+neck."""
        # print(img_metas)
        # print(img_metas[0]['filename'].split('/')[-1], img_metas[1]['filename'].split('/')[-1])
        x, stem0, j_out_x = self.backbone(img)
        # ======================================================
        # Homology loss
        # ======================================================
        j_out = self.net_decoder(x, stem0, j_out_x)
        t_out = self.net_t(img)
        
        a_out1 = get_A(img[0].unsqueeze(0)).cuda()
        a_out2 = get_A(img[1].unsqueeze(0)).cuda()
        a_out = torch.cat([a_out1, a_out2], 0)
        
        I_rec = j_out * t_out + (1 - t_out) * a_out
        loss_1 = self.mse_loss(I_rec, img)
        # -------------------------------------------------------
        lam = np.random.beta(1, 1)
        input_mix = lam * img + (1 - lam) * j_out

        x_mix, stem0, j_out_x = self.backbone(input_mix)
        j_out_mix = self.net_decoder(x_mix, stem0, j_out_x)
        # t_out_mix = self.net_t(input_mix)

        loss_2 = self.mse_loss(j_out_mix, j_out.detach())

        loss_3 = self.color_loss(j_out)

        total_loss = 1 * loss_1 + 1 * loss_2 + 0.01 * loss_3
        total_loss = total_loss 
        # save_image(img[0], '@fuhao/transfer/img.jpg', normalize=True)
        # save_image(j_out[0], '@fuhao/transfer/img_gen.jpg', normalize=True)
        # save_image(a_out[0], '@fuhao/transfer/a_out.jpg', normalize=True)
        # save_image(I_rec[0], '@fuhao/transfer/I_rec.jpg', normalize=True)
        # -------------------------------------------------------
        
        
        # if self.with_neck:
        #     x = self.neck(x)
        x = img
        return x, total_loss

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        
        x, loss_h = self.extract_feat(img, img_metas)
        # x, en_neck, loss_label_guide, label_guide_x = self.extract_feat_label_guide(img, img_metas, gt_bboxes)

        losses = {}
        # losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
        #                                       gt_labels, gt_bboxes_ignore, 
        #                                       label_mask=None
        #                                         )
        
        # en_losses = self.bbox_head_1.forward_train(en_neck, img_metas, gt_bboxes,
        #                                       gt_labels, gt_bboxes_ignore, 
        #                                       label_mask=label_guide_x
        #                                         )
        
        # losses.update({'loss_en_neck_cls':en_losses['loss_cls']})
        # losses.update({'loss_en_neck_bbox':en_losses['loss_bbox']})
        losses.update({'loss_homology':loss_h})
        
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat_test(img, img_metas)
        # x, en_neck, label_guide_x = self.test_extract_feat_label_guide(img)
        
        outs = self.bbox_head(x)
        # get origin input shape to support onnx dynamic shape
        if torch.onnx.is_in_onnx_export():
            # get shape as tensor
            img_shape = torch._shape_as_tensor(img)[2:]
            img_metas[0]['img_shape_for_onnx'] = img_shape
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        # skip post-processing when exporting to ONNX
        if torch.onnx.is_in_onnx_export():
            return bbox_list

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results




@DETECTORS.register_module()
class RetinaNet_H3(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 bbox_head_1=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RetinaNet_H3, self).__init__(backbone, neck, bbox_head, bbox_head_1, train_cfg,
                                        test_cfg, pretrained)
        
        # self.net_decoder = Decoder().cuda()
        # self.net_decoder = Cycle_Decoder().cuda()
        # self.net_decoder = Unet_Decoder().cuda()
        # self.net_decoder = Decoder_cat().cuda()
        # self.net_decoder = FPN_github_torch()
        # self.net_decoder = Unet_Decoder().cuda()
        self.net_decoder = FPN_Decoder()
        self.net_t = TNet().cuda()
        self.mse_loss = torch.nn.MSELoss().cuda()
        self.color_loss = ColorLoss().cuda()

    def extract_feat_test(self, img, img_metas):
        """Directly extract features from the backbone+neck."""
        x, stem0, j_out_x = self.backbone(img) 
        # ====================================================
        # Backbone + Decoder = Generator
        j_out = self.net_decoder(x, stem0, j_out_x)
        save_image(j_out[0], '@exp_homology/10_urpc/13/results/' + img_metas[0]['filename'].split('/')[-1], normalize=True)
        # ====================================================
        
        
        if self.with_neck:
            x = self.neck(x)
        return x
        
    def extract_feat(self, img, img_metas):
        """Directly extract features from the backbone+neck."""
        # print(img_metas)
        # print(img_metas[0]['filename'].split('/')[-1], img_metas[1]['filename'].split('/')[-1])
        x, stem0, j_out_x = self.backbone(img)
        # ======================================================
        # Homology loss
        # ======================================================
        j_out = self.net_decoder(x, stem0, j_out_x)
        t_out = self.net_t(img)
        
        a_out1 = get_A(img[0].unsqueeze(0)).cuda()
        a_out2 = get_A(img[1].unsqueeze(0)).cuda()
        a_out = torch.cat([a_out1, a_out2], 0)
        
        I_rec = j_out * t_out + (1 - t_out) * a_out
        loss_1 = self.mse_loss(I_rec, img)
        # -------------------------------------------------------
        lam = np.random.beta(1, 1)
        input_mix = lam * img + (1 - lam) * j_out

        x_mix, stem0, j_out_x = self.backbone(input_mix)
        j_out_mix = self.net_decoder(x_mix, stem0, j_out_x)
        # t_out_mix = self.net_t(input_mix)

        loss_2 = self.mse_loss(j_out_mix, j_out.detach())

        loss_3 = self.color_loss(j_out)

        total_loss = 1 * loss_1 + 1 * loss_2 + 0.01 * loss_3
        total_loss = total_loss 
        # save_image(img[0], '@fuhao/transfer/img.jpg', normalize=True)
        # save_image(j_out[0], '@fuhao/transfer/img_gen.jpg', normalize=True)
        # save_image(a_out[0], '@fuhao/transfer/a_out.jpg', normalize=True)
        # save_image(I_rec[0], '@fuhao/transfer/I_rec.jpg', normalize=True)
        # -------------------------------------------------------
        
        
        if self.with_neck:
            x = self.neck(x)
        # x = img
        return x, total_loss

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        
        x, loss_h = self.extract_feat(img, img_metas)
        # x, en_neck, loss_label_guide, label_guide_x = self.extract_feat_label_guide(img, img_metas, gt_bboxes)

        losses = {}
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore, 
                                              label_mask=None
                                                )
        
        # en_losses = self.bbox_head_1.forward_train(en_neck, img_metas, gt_bboxes,
        #                                       gt_labels, gt_bboxes_ignore, 
        #                                       label_mask=label_guide_x
        #                                         )
        
        # losses.update({'loss_en_neck_cls':en_losses['loss_cls']})
        # losses.update({'loss_en_neck_bbox':en_losses['loss_bbox']})
        losses.update({'loss_homology':loss_h})
        
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat_test(img, img_metas)
        # x, en_neck, label_guide_x = self.test_extract_feat_label_guide(img)
        
        outs = self.bbox_head(x)
        # get origin input shape to support onnx dynamic shape
        if torch.onnx.is_in_onnx_export():
            # get shape as tensor
            img_shape = torch._shape_as_tensor(img)[2:]
            img_metas[0]['img_shape_for_onnx'] = img_shape
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        # skip post-processing when exporting to ONNX
        if torch.onnx.is_in_onnx_export():
            return bbox_list

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results
    
    



@DETECTORS.register_module()
class RetinaNet_H4(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 bbox_head_1=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RetinaNet_H4, self).__init__(backbone, neck, bbox_head, bbox_head_1, train_cfg,
                                        test_cfg, pretrained)
        
        # self.net_decoder = Decoder().cuda()
        # self.net_decoder = Cycle_Decoder().cuda()
        # self.net_decoder = Unet_Decoder().cuda()
        # self.net_decoder = Decoder_cat().cuda()
        # self.net_decoder = FPN_github_torch()
        # self.net_decoder = Unet_Decoder().cuda()
        self.net_decoder = FPN_DecoderH4()
        self.net_t = TNet().cuda()
        self.mse_loss = torch.nn.MSELoss().cuda()
        self.color_loss = ColorLoss().cuda()

    def extract_feat_test(self, img, img_metas):
        """Directly extract features from the backbone+neck."""
        x, stem0, j_out_x = self.backbone(img) 
        # ====================================================
        # Backbone + Decoder = Generator
        j_out, P3_x = self.net_decoder(x, stem0, j_out_x)
        save_image(j_out[0], '@exp_homology/10_urpc/14/results/' + img_metas[0]['filename'].split('/')[-1], normalize=True)
        # ====================================================
        
        
        if self.with_neck:
            x = self.neck(x, P3_x)
        return x
        
    def extract_feat(self, img, img_metas):
        """Directly extract features from the backbone+neck."""
        # print(img_metas)
        # print(img_metas[0]['filename'].split('/')[-1], img_metas[1]['filename'].split('/')[-1])
        x, stem0, j_out_x = self.backbone(img)
        # ======================================================
        # Homology loss
        # ======================================================
        j_out, P3_x = self.net_decoder(x, stem0, j_out_x)
        t_out = self.net_t(img)
        
        a_out1 = get_A(img[0].unsqueeze(0)).cuda()
        a_out2 = get_A(img[1].unsqueeze(0)).cuda()
        a_out = torch.cat([a_out1, a_out2], 0)
        
        I_rec = j_out * t_out + (1 - t_out) * a_out
        loss_1 = self.mse_loss(I_rec, img)
        # -------------------------------------------------------
        lam = np.random.beta(1, 1)
        input_mix = lam * img + (1 - lam) * j_out

        x_mix, stem0, j_out_x = self.backbone(input_mix)
        j_out_mix, P3_x = self.net_decoder(x_mix, stem0, j_out_x)
        # t_out_mix = self.net_t(input_mix)

        loss_2 = self.mse_loss(j_out_mix, j_out.detach())

        loss_3 = self.color_loss(j_out)

        total_loss = 1 * loss_1 + 1 * loss_2 + 0.01 * loss_3
        total_loss = total_loss 
        # save_image(img[0], '@fuhao/transfer/img.jpg', normalize=True)
        # save_image(j_out[0], '@fuhao/transfer/img_gen.jpg', normalize=True)
        # save_image(a_out[0], '@fuhao/transfer/a_out.jpg', normalize=True)
        # save_image(I_rec[0], '@fuhao/transfer/I_rec.jpg', normalize=True)
        # -------------------------------------------------------
        
        
        if self.with_neck:
            x = self.neck(x, P3_x)
        # x = img
        return x, total_loss

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        
        x, loss_h = self.extract_feat(img, img_metas)
        # x, en_neck, loss_label_guide, label_guide_x = self.extract_feat_label_guide(img, img_metas, gt_bboxes)

        losses = {}
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore, 
                                              label_mask=None
                                                )
        
        # en_losses = self.bbox_head_1.forward_train(en_neck, img_metas, gt_bboxes,
        #                                       gt_labels, gt_bboxes_ignore, 
        #                                       label_mask=label_guide_x
        #                                         )
        
        # losses.update({'loss_en_neck_cls':en_losses['loss_cls']})
        # losses.update({'loss_en_neck_bbox':en_losses['loss_bbox']})
        losses.update({'loss_homology':loss_h})
        
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat_test(img, img_metas)
        # x, en_neck, label_guide_x = self.test_extract_feat_label_guide(img)
        
        outs = self.bbox_head(x)
        # get origin input shape to support onnx dynamic shape
        if torch.onnx.is_in_onnx_export():
            # get shape as tensor
            img_shape = torch._shape_as_tensor(img)[2:]
            img_metas[0]['img_shape_for_onnx'] = img_shape
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        # skip post-processing when exporting to ONNX
        if torch.onnx.is_in_onnx_export():
            return bbox_list

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results




@DETECTORS.register_module()
class RetinaNet_H21(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 bbox_head_1=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RetinaNet_H21, self).__init__(backbone, neck, bbox_head, bbox_head_1, train_cfg,
                                        test_cfg, pretrained)
        
        # self.net_decoder = Decoder().cuda()
        # self.net_decoder = Cycle_Decoder().cuda()
        # self.net_decoder = Unet_Decoder().cuda()
        # self.net_decoder = Decoder_cat().cuda()
        # self.net_decoder = FPN_github_torch()
        # self.net_decoder = Unet_Decoder().cuda()
        self.net_decoder = FPN_Decoder21()
        self.net_t = TNet().cuda()
        self.mse_loss = torch.nn.MSELoss().cuda()
        self.color_loss = ColorLoss().cuda()

    def extract_feat_test(self, img, img_metas):
        """Directly extract features from the backbone+neck."""
        x, stem0 = self.backbone(img) 
        # ====================================================
        # Backbone + Decoder = Generator
        j_out = self.net_decoder(x, stem0)
        save_image(j_out[0], '@exp_homology/20/21/results/' + img_metas[0]['filename'].split('/')[-1], normalize=True)
        # ====================================================
        
        
        if self.with_neck:
            x = self.neck(x)
        return x
        
    def extract_feat(self, img, img_metas):
        """Directly extract features from the backbone+neck."""
        # print(img_metas)
        # print(img_metas[0]['filename'].split('/')[-1], img_metas[1]['filename'].split('/')[-1])
        B, C, H, W = img.shape
        mini_img = nn.functional.interpolate(img, size=[H//2, W//2], mode='bilinear', align_corners=True)
        x, stem0 = self.backbone(img)
        # ======================================================
        # Homology loss
        # ======================================================
        j_out = self.net_decoder(x, stem0)
        t_out = self.net_t(mini_img)
        
        a_out1 = get_A(mini_img[0].unsqueeze(0)).cuda()
        a_out2 = get_A(mini_img[1].unsqueeze(0)).cuda()
        a_out = torch.cat([a_out1, a_out2], 0)
        
        I_rec = j_out * t_out + (1 - t_out) * a_out
        loss_1 = self.mse_loss(I_rec, mini_img)
        # -------------------------------------------------------
        lam = np.random.beta(1, 1)
        large_j_out = nn.functional.interpolate(j_out, size=[H, W], mode='bilinear', align_corners=True)
        input_mix = lam * img + (1 - lam) * large_j_out

        x_mix, stem0 = self.backbone(input_mix)
        j_out_mix = self.net_decoder(x_mix, stem0)
        # t_out_mix = self.net_t(input_mix)

        loss_2 = self.mse_loss(j_out_mix, j_out.detach())

        loss_3 = self.color_loss(j_out)

        total_loss = 1 * loss_1 + 1 * loss_2 + 0.01 * loss_3
        total_loss = total_loss 
        save_image(img[0], '@fuhao/transfer/img.jpg', normalize=True)
        save_image(j_out[0], '@fuhao/transfer/img_gen.jpg', normalize=True)
        save_image(a_out[0], '@fuhao/transfer/a_out.jpg', normalize=True)
        save_image(I_rec[0], '@fuhao/transfer/I_rec.jpg', normalize=True)
        # -------------------------------------------------------
        
        
        # if self.with_neck:
        #     x = self.neck(x)
        x = img
        return x, total_loss

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        
        x, loss_h = self.extract_feat(img, img_metas)
        # x, en_neck, loss_label_guide, label_guide_x = self.extract_feat_label_guide(img, img_metas, gt_bboxes)

        losses = {}
        # losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
        #                                       gt_labels, gt_bboxes_ignore, 
        #                                       label_mask=None
        #                                         )
        
        # en_losses = self.bbox_head_1.forward_train(en_neck, img_metas, gt_bboxes,
        #                                       gt_labels, gt_bboxes_ignore, 
        #                                       label_mask=label_guide_x
        #                                         )
        
        # losses.update({'loss_en_neck_cls':en_losses['loss_cls']})
        # losses.update({'loss_en_neck_bbox':en_losses['loss_bbox']})
        losses.update({'loss_homology':loss_h})
        
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        x = self.extract_feat_test(img, img_metas)
        # x, en_neck, label_guide_x = self.test_extract_feat_label_guide(img)
        
        outs = self.bbox_head(x)
        # get origin input shape to support onnx dynamic shape
        if torch.onnx.is_in_onnx_export():
            # get shape as tensor
            img_shape = torch._shape_as_tensor(img)[2:]
            img_metas[0]['img_shape_for_onnx'] = img_shape
        bbox_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        # skip post-processing when exporting to ONNX
        if torch.onnx.is_in_onnx_export():
            return bbox_list

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        return bbox_results





if __name__ == "__main__":
    net = Cycle_Decoder().cuda()