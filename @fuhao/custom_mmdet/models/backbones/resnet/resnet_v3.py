import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.builder import BACKBONES
from mmdet.models.utils import ResLayer
from mmdet.models.backbones.resnet import ResNet

from .resnet18 import ResNet18_FPN

import sys
sys.path.append('@fuhao/')
# from custom_mmdet.models.backbones.stem.scene_mining import *

import torch.nn.functional as F

import seaborn as sns
sns.set(font_scale=1.5)

from torchvision.utils import save_image


# """ AE
from re import X
from tkinter import W
from turtle import forward
from requests import patch
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import os
import math

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_plugin_layer,
                      constant_init, kaiming_init)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_size, momentum=0.8),
            nn.GELU(),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


class LKFFN(nn.Module):
    def __init__(self, hidden_dim, dropout=0.):
        super().__init__()
        out = hidden_dim
        hidden_dim = hidden_dim * 4
        self.net = nn.Sequential(
            nn.Conv2d(out, hidden_dim, 7, 1,
                        3, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            # nn.Dropout(p=.20),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 3, groups=hidden_dim, dilation=3, bias=False), 
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            # nn.Dropout(p=.20),
            # SE(hidden_dim, hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0, bias=False),
            # SE(hidden_dim, hidden_dim),
            # nn.BatchNorm2d(hidden_dim),
        )

        # self.act = nn.GELU()

    def forward(self, x):
        # print(x.shape)
        # return self.act(self.net(x) + x)
        return self.net(x) + x

class LKStem(nn.Module):
    def __init__(self, in_channels, hidden_dim, dropout=0.):
        super().__init__()
        out = hidden_dim
        # hidden_dim = hidden_dim * 4
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 7, 2,
                        3, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            # nn.Dropout(p=.20),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 3, groups=hidden_dim, dilation=3, bias=False), 
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            # nn.Dropout(p=.20),
            # SE(hidden_dim, hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
            # SE(hidden_dim, hidden_dim),
            # nn.BatchNorm2d(hidden_dim),
        )

        # self.act = nn.GELU()

    def forward(self, x):
        # print(x.shape)
        # return self.act(self.net(x) + x)
        # return self.net(x) + x
        return self.net(x)

class LKMB(nn.Module):
    def __init__(self, inp, oup, downsample=False, expansion=4):
        super().__init__()
        self.downsample = downsample
        stride = 1 if self.downsample == False else 2
        hidden_dim = int(inp * expansion)

        self.bn = nn.BatchNorm2d(oup)
        self.act = nn.GELU()
        # hidden_dim = oup
        
        if self.downsample:
            # self.pool = nn.MaxPool2d(3, 2, 1)
            self.proj = nn.Sequential(
                    nn.Conv2d(inp, oup, 3, 2, 1, bias=False),
                    nn.BatchNorm2d(oup),
                    nn.GELU()
            )

        else:
            self.conv = nn.Sequential(
                # pw
                # down-sample in the first conv
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                # nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # dw


                nn.Conv2d(hidden_dim, hidden_dim, 7, 1,
                            3, groups=hidden_dim, bias=False),
                # nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # nn.Dropout(p=.20),
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 3, groups=hidden_dim, dilation=3, bias=False), 
                # nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # nn.Dropout(p=.20),


                # SE(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                # SE(oup, oup),
                # nn.BatchNorm2d(oup),
                # nn.GELU()

            )
            
        # self.act = nn.GELU()

    def forward(self, x):
        if self.downsample:
            # return self.proj(self.pool(x)) + self.conv(x)
            # return self.act(self.proj(x) + self.conv(x))
            # return self.proj(x) + self.conv(x) 
            return self.proj(x)
        else:
            # return self.act(x + self.conv(x))
            # return x + self.conv(x)
            return self.conv(x)
            # return self.act(self.bn( x + self.conv(x)))
            # x = self.conv(x)
            # return torch.cat((x, ori), 1)



class AE(nn.Module):
    def __init__(self, conv=None):
        super().__init__()

        # self.lk_stem0 = LKStem(3, 64)
        # self.lk_stem1 = LKStem(64, 64)
        
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.GELU()
        )
        self.mix1 = LKMB(64, 64)
        self.mix2 = LKMB(64, 64)
        self.drop = nn.Dropout(p=.20)

        self.mix3 = nn.Sequential(
                nn.Conv2d(64, 64, 3, 2, 1),
                # nn.Conv2d(64, 256, 3, 2, 1),
                nn.BatchNorm2d(64),
                nn.GELU(),
                # nn.Conv2d(64, 96, 3, 2, 1),
        )

        self.norm = nn.LayerNorm(96)


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


    def forward(self, x):
        ori = x
        # x = self.lk_stem0(x)
        # x = self.lk_stem1(x)
        x = self.stem(x)
        stem1 = x
        x = self.mix1(x)
        x = x + stem1
        mix1 = x
        x = self.mix2(x)
        mix2= x + mix1
        # x = self.mix3(x)
        mix3 = x

        return x, mix2, mix3
        


# """

class trans(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        # self.conv1 = nn.Conv2d(64, 64, 3, 1, 1)
        # self.norm1 = nn.BatchNorm2d(64)
        # self.conv2 = nn.Conv2d(64, 64, 3, 1 ,1)
        # self.norm2 = nn.BatchNorm2d(64)
        # self.conv3 = nn.Conv2d(64, 64, 3, 1, 1)
        # self.norm3 = nn.BatchNorm2d(64)
        # self.conv4 = nn.Conv2d(64, 64, 3, 1, 1)
        # self.norm4 = nn.BatchNorm2d(64)
        # self.conv5 = nn.Conv2d(64, 64, 3, 1, 1)
        # self.align = nn.Conv2d(64, 256, 1, 1, 0)

        # self.conv1 = nn.Conv2d(64, 128, 3, 1, 1)
        # self.norm1 = nn.BatchNorm2d(128)
        # self.conv2 = nn.Conv2d(128, 256, 3, 1 ,1)
        # self.norm2 = nn.BatchNorm2d(256)
        # self.conv3 = nn.Conv2d(256, 512, 3, 1, 1)
        # self.norm3 = nn.BatchNorm2d(512)
        # self.conv4 = nn.Conv2d(512, 256, 3, 1, 1)
        # self.norm4 = nn.BatchNorm2d(256)
        # self.conv5 = nn.Conv2d(256, 128, 3, 1, 1)
        # self.norm5 = nn.BatchNorm2d(128)
        # self.conv6 = nn.Conv2d(128, 64, 3, 1, 1)

        # self.align = nn.Conv2d(64, 256, 1, 1, 0)


        self.conv1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.norm1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 512, 3, 1 ,1)
        self.norm2 = nn.BatchNorm2d(512)
        self.conv3 = nn.Conv2d(512, 1024, 3, 1, 1)
        self.norm3 = nn.BatchNorm2d(1024)
        self.conv4 = nn.Conv2d(1024, 512, 3, 1, 1)
        self.norm4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 128, 3, 1, 1)
        self.norm5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 64, 3, 1, 1)

        self.align = nn.Conv2d(64, 256, 1, 1, 0)

        self.act = nn.GELU()
    def forward(self, x):
        x = self.stem(x)
        x0 = x 

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x1 = x

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)
        x2 = x

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.act(x)
        x3 = x

        # x = x + x1
        x = self.conv4(x)
        x = self.act(x)
        x4 = x

        x = x + x2
        x = self.conv5(x)
        x = self.act(x)
        x5 = x

        x = x + x1
        x = self.conv6(x)

        # out = x1 + x2 + x3 + x4 + x5
        out = x

        align = self.align(out)

        return out



@BACKBONES.register_module()
class ResNet_v321(ResNet):
    
    def __init__(self,
                 depth,
                 **kwargs
                ):
        super(ResNet_v321, self).__init__(depth, **kwargs)
        
        self.c1 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            # nn.ReLU(inplace=True),
        )

        # self.c2 = nn.Sequential(
        #     nn.Conv2d(32, 64, 3, 2, 1),
        #     nn.BatchNorm2d(64),
        #     nn.GELU(),
        #     # nn.ReLU(inplace=True),
        # )

        self.ae = AE(conv=True)


        # self.custom_init_weights(self.c1)
        # self.custom_init_weights(self.ae)



    def partition(self, x):
        B, H, W, C = x.shape

        # padding
        # pad_input = (H % 2 == 1) or (W % 2 == 1)
        # if pad_input:
        #     x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, :, 0::2, 0::2]  # B C H/2 W/2 
        x1 = x[:, :, 1::2, 0::2]  # B C H/2 W/2 
        x2 = x[:, :, 0::2, 1::2]  # B C H/2 W/2 
        x3 = x[:, :, 1::2, 1::2]  # B C H/2 W/2 

        x = torch.cat([x0, x1, x2, x3], 1)  # B 4*C H/2 W/2

        return x

    def adaptive_ae(self, x):
        input = x
        
        # 下采样
        # print('ori_shape', input.shape)
        # x = self.partition(x)
        # print(x.shape)
        x = self.c1(x)
        # x = self.c2(x)
        # print(x.shape)
        mix2, mix3 = self.ae(x)
        # print(x.shape)

        # """ mix3 归一化

        # S_t = mix2.pow(2).mean(1)
        # N_T,H_T,W_T = S_t.shape
        # S_t_min = S_t.detach().view(N_T, -1).min(1).values.unsqueeze(1).unsqueeze(1)
        # S_t_max = S_t.detach().view(N_T,  -1).max(1).values.unsqueeze(1).unsqueeze(1)

        # # print(S_t.shape, S_t_min.shape)

        # S_t = ( S_t - S_t_min ) / ( S_t_max - S_t_min + 1e-4) 
        # mix2 = S_t
        # """


        # ----------------------------------------------------------------
        # x[x < x.mean()] = 0
        # print(x.shape)
        # from torchvision.utils import save_image
        # save_image(x, "./x.jpg" , normalize=True)
        # 上采样 + 通道数降为3
        # print(input.shape)
        # print(x.shape)
        # x = torch.sigmoid(x)
        # x = input * x + input

        # x = input * x
        
        # from torchvision.utils import save_image
        # save_image(input, "./input.jpg" , normalize=True)
        # save_image(x, "./input-x.jpg" , normalize=True)

        # import seaborn as sns
        # sns.set(font_scale=1.5)
        # t = input.detach().pow(2).mean(1)
        # t = t.data.cpu().numpy()
        # # fig = sns.heatmap(data=t[0], vmax=9., vmin=6.5, cmap='viridis') 
        # # fig = sns.heatmap(data=t[0], vmax=4., vmin=3.4, cmap='viridis')
        # fig = sns.heatmap(data=t[0], cmap='viridis')  
        # heatmap = fig.get_figure()
        # # heatmap.savefig(output_images_path + name[0], dpi=400)
        # heatmap.savefig('./1.png', dpi=400)
        # heatmap.clear()

        # t = x.detach().pow(2).mean(1)
        # t = t.data.cpu().numpy()
        # # fig = sns.heatmap(data=t[0], vmax=9., vmin=6.5, cmap='viridis') 
        # # fig = sns.heatmap(data=t[0], vmax=4., vmin=3.4, cmap='viridis')
        # fig = sns.heatmap(data=t[0], cmap='viridis')  
        # heatmap = fig.get_figure()
        # # heatmap.savefig(output_images_path + name[0], dpi=400)
        # heatmap.savefig('./2.png', dpi=400)
        # heatmap.clear()
        # raise
        # ----------------------------------------------------------------

        # 上采样 + 通道数降为3
        # print(input.shape)
        # print(x.shape)
        # x = input * x * 0.7 + input * 0.3
        # print(input.shape, x.shape, x[:, 0, :, :].shape, x[:, 0, :, :].unsqueeze(1).shape)

        # x = input * x[:, 0, :, :].unsqueeze(1) - input * x[:, 1, :, :].unsqueeze(1)

        # temp= 0.5
        # N, C, H, W= x.shape
        # x = torch.abs(x) 
        # # Bs*W*H
        # x = x.mean(axis=1, keepdim=True)

        # temp= 0.5
        # N, C, H, W= mix2.shape
        # x = torch.abs(mix2) 
        # # Bs*W*H
        # x = x.mean(axis=1, keepdim=True)
        # mix2 = mix2.unsqueeze(1)
        # print(mix2.shape)
        # mix2 = F.interpolate(mix2, scale_factor=4, mode='bilinear', align_corners=False)

        # mix2[mix2<0.2] = 0.
        # mix2[mix2>0.6] = 1.
        # x = (H * W * F.softmax((x/temp).view(N,-1), dim=1)).view(N, H, W)
        # print(x)
        # raise
        # x = x.unsqueeze(1)
        # print(x.shape)

        # import seaborn as sns
        # sns.set(font_scale=1.5)
        # t = x.detach().cpu()
        # t = t.data.cpu().numpy()
        # # fig = sns.heatmap(data=t[0], vmax=9., vmin=6.5, cmap='viridis') 
        # # fig = sns.heatmap(data=t[0], vmax=4., vmin=3.4, cmap='viridis')
        # fig = sns.heatmap(data=t[0].squeeze(), cmap='viridis')  
        # heatmap = fig.get_figure()
        # # heatmap.savefig(output_images_path + name[0], dpi=400)
        # heatmap.savefig('./1.png', dpi=400)
        # heatmap.clear()



        # x = torch.sigmoid(x)
        # x = (x - x.min()) / (x.max() - x.min())
        # x[x<0.5] = 0
        # x[x>0] = 1
        # print(x)
        # x = input * mix2 
        # x = x
        # x = mix2
        # from torchvision.utils import save_image
        # save_image(x, "@fuhao/transfer/input-x.jpg" , normalize=True)
        # save_image(input, "./input.jpg" , normalize=True)

        return mix2, mix3

    def forward(self, x):
        # self.ae.eval()
        # self.c1.eval()
        input = x
        # mix2, mix3 = self.adaptive_ae(x)

        # feat = x
        # feat = feat.unsqueeze(1)
        """Forward function."""
        if self.deep_stem:
            x = self.stem(input)
        else:
            x = self.conv1(input)
            x = self.norm1(x)
            x = self.relu(x)

            x, mix2, mix3 = self.ae(x)
        # x = self.maxpool(x)

        # x = x + mix2
        # x = x * mix2[mix2<0.3]
        # x = mix2

        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)




@BACKBONES.register_module()
class ResNet_v322(ResNet):
    
    def __init__(self,
                 depth,
                 conv_cfg=None,
                 **kwargs
                ):
        super(ResNet_v322, self).__init__(depth, **kwargs)
        
        # self.c1 = nn.Sequential(
        #     nn.Conv2d(3, 3, 3, 1, 1),
        #     # nn.BatchNorm2d(1),
        #     nn.GELU(),
        #     # nn.ReLU(inplace=True),
        # )

        # self.c2 = nn.Sequential(
        #     nn.Conv2d(256, 64, 1, 1, 0)
        # )

        self.trans = trans()


        # self.c3 = nn.Sequential(
        #     nn.Conv2d(256, 64, 1, 1, 0)
        # )

        self.conv_cfg = conv_cfg
        # self.conv1 = build_conv_layer(
        #     self.conv_cfg,
        #     1,
        #     64,
        #     kernel_size=7,
        #     stride=2,
        #     padding=3,
        #     bias=False)
        # self.c2 = nn.Sequential(
        #     nn.Conv2d(32, 64, 3, 2, 1),
        #     nn.BatchNorm2d(64),
        #     nn.GELU(),
        #     # nn.ReLU(inplace=True),
        # )

        # self.ae = AE(conv=True)


        # self.custom_init_weights(self.c1)
        # self.custom_init_weights(self.ae)


    def adaptive_ae(self, x):
        input = x
        
        # 下采样
        # print('ori_shape', input.shape)
        # x = self.partition(x)
        # print(x.shape)
        x = self.c1(x)
        # x = self.c2(x)
        # print(x.shape)
        mix2, mix3 = self.ae(x)
        # print(x.shape)

        # """ mix3 归一化

        # S_t = mix2.pow(2).mean(1)
        # N_T,H_T,W_T = S_t.shape
        # S_t_min = S_t.detach().view(N_T, -1).min(1).values.unsqueeze(1).unsqueeze(1)
        # S_t_max = S_t.detach().view(N_T,  -1).max(1).values.unsqueeze(1).unsqueeze(1)

        # # print(S_t.shape, S_t_min.shape)

        # S_t = ( S_t - S_t_min ) / ( S_t_max - S_t_min + 1e-4) 
        # mix2 = S_t
        # """


        # ----------------------------------------------------------------
        # x[x < x.mean()] = 0
        # print(x.shape)
        # from torchvision.utils import save_image
        # save_image(x, "./x.jpg" , normalize=True)
        # 上采样 + 通道数降为3
        # print(input.shape)
        # print(x.shape)
        # x = torch.sigmoid(x)
        # x = input * x + input

        # x = input * x
        
        # from torchvision.utils import save_image
        # save_image(input, "./input.jpg" , normalize=True)
        # save_image(x, "./input-x.jpg" , normalize=True)

        # import seaborn as sns
        # sns.set(font_scale=1.5)
        # t = input.detach().pow(2).mean(1)
        # t = t.data.cpu().numpy()
        # # fig = sns.heatmap(data=t[0], vmax=9., vmin=6.5, cmap='viridis') 
        # # fig = sns.heatmap(data=t[0], vmax=4., vmin=3.4, cmap='viridis')
        # fig = sns.heatmap(data=t[0], cmap='viridis')  
        # heatmap = fig.get_figure()
        # # heatmap.savefig(output_images_path + name[0], dpi=400)
        # heatmap.savefig('./1.png', dpi=400)
        # heatmap.clear()

        # t = x.detach().pow(2).mean(1)
        # t = t.data.cpu().numpy()
        # # fig = sns.heatmap(data=t[0], vmax=9., vmin=6.5, cmap='viridis') 
        # # fig = sns.heatmap(data=t[0], vmax=4., vmin=3.4, cmap='viridis')
        # fig = sns.heatmap(data=t[0], cmap='viridis')  
        # heatmap = fig.get_figure()
        # # heatmap.savefig(output_images_path + name[0], dpi=400)
        # heatmap.savefig('./2.png', dpi=400)
        # heatmap.clear()
        # raise
        # ----------------------------------------------------------------

        # 上采样 + 通道数降为3
        # print(input.shape)
        # print(x.shape)
        # x = input * x * 0.7 + input * 0.3
        # print(input.shape, x.shape, x[:, 0, :, :].shape, x[:, 0, :, :].unsqueeze(1).shape)

        # x = input * x[:, 0, :, :].unsqueeze(1) - input * x[:, 1, :, :].unsqueeze(1)

        # temp= 0.5
        # N, C, H, W= x.shape
        # x = torch.abs(x) 
        # # Bs*W*H
        # x = x.mean(axis=1, keepdim=True)

        # temp= 0.5
        # N, C, H, W= mix2.shape
        # x = torch.abs(mix2) 
        # # Bs*W*H
        # x = x.mean(axis=1, keepdim=True)
        # mix2 = mix2.unsqueeze(1)
        # print(mix2.shape)
        # mix2 = F.interpolate(mix2, scale_factor=4, mode='bilinear', align_corners=False)

        # mix2[mix2<0.2] = 0.
        # mix2[mix2>0.6] = 1.
        # x = (H * W * F.softmax((x/temp).view(N,-1), dim=1)).view(N, H, W)
        # print(x)
        # raise
        # x = x.unsqueeze(1)
        # print(x.shape)

        # import seaborn as sns
        # sns.set(font_scale=1.5)
        # t = x.detach().cpu()
        # t = t.data.cpu().numpy()
        # # fig = sns.heatmap(data=t[0], vmax=9., vmin=6.5, cmap='viridis') 
        # # fig = sns.heatmap(data=t[0], vmax=4., vmin=3.4, cmap='viridis')
        # fig = sns.heatmap(data=t[0].squeeze(), cmap='viridis')  
        # heatmap = fig.get_figure()
        # # heatmap.savefig(output_images_path + name[0], dpi=400)
        # heatmap.savefig('./1.png', dpi=400)
        # heatmap.clear()



        # x = torch.sigmoid(x)
        # x = (x - x.min()) / (x.max() - x.min())
        # x[x<0.5] = 0
        # x[x>0] = 1
        # print(x)
        # x = input * mix2 
        # x = x
        # x = mix2
        # from torchvision.utils import save_image
        # save_image(x, "@fuhao/transfer/input-x.jpg" , normalize=True)
        # save_image(input, "./input.jpg" , normalize=True)

        return mix2, mix3

    def forward(self, x):

        input = x
        input = self.maxpool(input)
        input = self.maxpool(input)
        # input, mix2, mix3 = self.ae(input)
        mix2 = self.trans(input)

        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)

        # neck = self.c2(data['feat'])
        x = x + mix2
        # x = neck

        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)


@BACKBONES.register_module()
class ResNet_v324(ResNet):
    
    def __init__(self,
                 depth,
                 conv_cfg=None,
                 **kwargs
                ):
        super(ResNet_v324, self).__init__(depth, **kwargs)
        

        self.trans = ResNet18_FPN()

        self.align = nn.Conv2d(256, 64, 1, 1, 0)

        self.conv_cfg = conv_cfg


    def forward(self, x):

        input = x

        # self.trans.eval()
        mix2 = self.trans(input)

        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)

        # neck = self.c2(data['feat'])
        """Combine Method."""
        x = x + self.align(mix2[0])
        # x = x * self.align(mix2[0])

        # x = neck
        """Over"""
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)


@BACKBONES.register_module()
class ResNet_v326(ResNet):
    
    def __init__(self,
                 depth,
                 conv_cfg=None,
                 **kwargs
                ):
        super(ResNet_v326, self).__init__(depth, **kwargs)
        

        self.trans = ResNet18_FPN()

        self.align = nn.Conv2d(256, 64, 1, 1, 0)

        self.mask = nn.Conv2d(256, 1, 1, 1, 0)

        self.conv_cfg = conv_cfg


    def forward(self, x):

        input = x

        # self.trans.eval()
        mix2 = self.trans(input)

        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)

        # neck = self.c2(data['feat'])
        """Combine Method."""
        x = x + self.align(mix2[0])
        # x = x * self.mask(mix2[0])
        # x = x * self.align(mix2[0])

        # x = neck
        """Over"""
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

@BACKBONES.register_module()
class ResNet_v327(ResNet):
    
    def __init__(self,
                 depth,
                 conv_cfg=None,
                 **kwargs
                ):
        super(ResNet_v327, self).__init__(depth, **kwargs)
        

        self.trans = ResNet18_FPN()

        self.align = nn.Conv2d(256, 64, 1, 1, 0)

        self.mask = nn.Conv2d(256, 1, 1, 1, 0)

        self.conv_cfg = conv_cfg


    def forward(self, x):

        input = x

        # self.trans.eval()
        mix2 = self.trans(input)

        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)

        # neck = self.c2(data['feat'])
        """Combine Method."""
        x = x + self.align(mix2[0])
        # x = x * self.mask(mix2[0])
        # x = x * self.align(mix2[0])

        # x = neck
        """Over"""
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

from torchvision.ops import DeformConv2d
class DConv(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False):
        super(DConv, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, 2 * kernel_size * kernel_size, kernel_size=kernel_size,
                            stride=stride, padding=padding, bias=bias)
        self.conv2 = DeformConv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(x, out)
        return out

import torchvision
import numpy as np
from torchvision.ops import DeformConv2d

class DCNv2(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride, padding):
        super(DCNv2, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1) #原卷积
 
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


@BACKBONES.register_module()
class ResNet_v328(ResNet):

    def __init__(self,
                 depth,
                 conv_cfg=None,
                 **kwargs
                ):
        super(ResNet_v328, self).__init__(depth, **kwargs)

        # -----------------------------------------------
        # Deformable Convolution
        # -----------------------------------------------

        self.align = DCNv2(256, 64, 3, 1, 1)
        # self.align = nn.Conv2d(256, 64, 3, 1, 1)

        # self.align = DConv(256, 64, 3, 1, 1)
        # -----------------------------------------------
        

        self.trans = ResNet18_FPN()

        self.mask = nn.Conv2d(256, 1, 1, 1, 0)

        self.conv_cfg = conv_cfg


    def forward(self, x):
        input = x

        # self.trans.eval()
        mix2 = self.trans(input)

        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)

        # neck = self.c2(data['feat'])
        """Combine Method."""
        x = x + self.align(mix2[0])
        # print(input.shape, x.shape)
        # x = x * self.mask(mix2[0])
        # x = x * self.align(mix2[0])

        # x = neck
        """Over"""
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)


# -------------------------------
# ResNet_v329
# -------------------------------

class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w1 = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w1 = torch.nn.Parameter(w1, requires_grad=True)
        self.w1 = w1

        w2 = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w2 = torch.nn.Parameter(w2, requires_grad=True)
        self.w2 = w2

        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):

        factor_w1 = self.mix_block(self.w1)
        factor_w2 = self.mix_block(self.w2)

        out = fea1 * factor_w1.expand_as(fea1) + fea2 * factor_w2.expand_as(fea2)
        return out

@BACKBONES.register_module()
class ResNet_v329(ResNet):
    
    def __init__(self,
                 depth,
                 conv_cfg=None,
                 **kwargs
                ):
        super(ResNet_v329, self).__init__(depth, **kwargs)
        

        self.trans = ResNet18_FPN()

        self.align = nn.Conv2d(256, 64, 1, 1, 0)

        self.mask = nn.Conv2d(256, 1, 1, 1, 0)

        self.conv_cfg = conv_cfg

        self.mixop = Mix(m=0.)


    def forward(self, x):

        input = x

        # self.trans.eval()
        mix2 = self.trans(input)

        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)

        # neck = self.c2(data['feat'])
        """Combine Method."""
        align = self.align(mix2[0])
        x = self.mixop(x, align)

        # x = neck
        """Over"""
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)