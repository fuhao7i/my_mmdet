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


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.bn = nn.BatchNorm2d(embed_dim)
        self.act = nn.GELU()

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

    def forward(self, x, mode='BN'):
        x = self.proj(x)

        _, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2).view(-1, C, H, W)

        return x     


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

class Upsample(nn.Module):
    def __init__(self, num_feats):
        super().__init__()

        self.m = nn.Sequential(
            nn.Conv2d(num_feats, 4*num_feats, 3, 1, 1),
            nn.BatchNorm2d(4*num_feats),
            nn.GELU(),
            nn.PixelShuffle(2)
        )
    
    def forward(self, x):

        x = self.m(x)
        return x

class SE(nn.Module):
    def __init__(self, oup, expansion=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(oup * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(oup * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

def make_layer(block, inp, oup, depth, image_size):
    layers = nn.ModuleList([])
    for i in range(depth):
        if i == 0:
            layers.append(block(inp, oup, image_size, downsample=True))
        else:
            layers.append(block(oup, oup, image_size))
    return nn.Sequential(*layers)

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
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # dw


                nn.Conv2d(hidden_dim, hidden_dim, 7, 1,
                            3, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # nn.Dropout(p=.20),
                nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 3, groups=hidden_dim, dilation=3, bias=False), 
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                # nn.Dropout(p=.20),


                # SE(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                # SE(oup, oup),
                nn.BatchNorm2d(oup),
                nn.GELU()

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

# class Adaptable_Embedding(nn.Module):
#     def __init__(self):
#         super().__init__()
    
#         self.input = nn.Sequential(
#              nn.Conv2d(3, 64, 3, 1, 1, bias=False),
#              nn.BatchNorm2d(64),
#             # nn.LayerNorm((256, 256)),
#              nn.GELU()
#         )

#         self.output = nn.Conv2d(in_channels=99, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)

#         self.stem_block0 = LKMB(64, 64, downsample=True)
#         self.stem_block1 = LKMB(64, 64)
#         self.stem_block2 = LKFFN(64)

#         self.stem_block3 = LKMB(64, 96, downsample=True)
#         self.stem_block4 = LKMB(96, 96)
#         self.stem_block5 = LKFFN(96)
#         self.stem_block6 = LKMB(96, 96)
#         self.stem_block7 = LKFFN(96)

#         self.ir1 = Upsample(128)
#         self.ir2 = Upsample(96)

#         self.zhuanzhi0 = nn.Sequential(
#             nn.ConvTranspose2d(96, 64, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(64, momentum=0.8),
#             nn.GELU(),
#         )

#         self.zhuanzhi1 = nn.Sequential(
#             nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(32, momentum=0.8),
#             nn.GELU(),
#         )

#         self.sigmoid = nn.Sigmoid()

#         self.up1 = UNetUp(96, 64)

#         self.final0 = nn.Sequential(
#             nn.Conv2d(131, 32, 3, 1, 1, bias=False),
#              nn.BatchNorm2d(32),
#              nn.GELU()
#         )
#         self.final = nn.Sequential(
#             # nn.Upsample(scale_factor=2),
#             # nn.ZeroPad2d((1, 0, 1, 0)),
#             nn.Conv2d(35, 3, 3, 1, padding=1),
#             nn.Tanh(),
#         )

#     def forward(self, x):
#         ori = x

#         x = self.input(x)
#         x = self.stem_block0(x)
#         x = self.stem_block1(x)
#         l1 = self.stem_block2(x)
#         x = self.stem_block3(l1)
#         x = self.stem_block4(x)
#         x = self.stem_block5(x)
#         x = self.stem_block6(x)
#         x = self.stem_block7(x)

#         #fuhao7i
#         x = self.zhuanzhi0(x)
#         x = x + l1
#         x = self.zhuanzhi1(x)
#         x = torch.cat((x, ori), 1)
#         x = self.final(x)
#         return x
#         #over

#         out = self.up1(x, l1)
#         out = self.ir1(out)
#         # out = torch.cat((out, l1), 1)
#         # out = self.ir2(out)

#         out = torch.cat((out, ori), 1)
#         out = self.final0(out)
#         out = torch.cat((out, ori), 1)
#         out = self.final(out)
#         # out = self.up1(out)
#         # out = self.up2(out)
#         # out = self.output(out)
#         # out = self.sigmoid(out)
#         # print(out.shape)
#         return out


class AE(nn.Module):
    def __init__(self, conv=None):
        super().__init__()
    
        # self.input = nn.Sequential(
        #      nn.Conv2d(3, 64, 3, 2, 1, bias=False),
        #      nn.BatchNorm2d(64),
        #     # nn.LayerNorm((256, 256)),
        #      nn.GELU()
        # )

        # self.patch1 = OverlapPatchEmbed(3, 2, 3, 64)
        self.mix1 = LKMB(64, 64)
        self.mix2 = LKMB(64, 64)
        # self.patch2 = OverlapPatchEmbed(3, 2, 64, 96)
        self.drop = nn.Dropout(p=.20)
        """BigBoss
        self.input = LKMB(3, 64, downsample=True)

        self.tiny1 = LKMB(3, 29)
        self.tiny2 = LKMB(32, 29)

        # self.output = nn.Conv2d(in_channels=99, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)

        # self.stem_block0 = LKMB(64, 64, downsample=True)
        # self.stem_block1 = LKMB(64, 64)
        # self.stem_block2 = LKFFN(64)
        self.stem_b2 = LKMB(64, 64)

        self.stem_block3 = LKMB(64, 96, downsample=True)
        # self.stem_block4 = LKMB(96, 96)
        # self.stem_block5 = LKFFN(96)
        self.stem_b5 = LKMB(96, 96)
        Over"""
        # self.stem_6 = nn.Sequential(
        #         nn.Conv2d(96, 96, 1, 1, 0),
        # )
        # self.stem = nn.Sequential(
        #         nn.Conv2d(3, 64, 3, 2, 1),
        #         nn.BatchNorm2d(64),
        #         nn.GELU(),
        #         nn.Conv2d(64, 96, 3, 2, 1),
        # )
        self.mix3 = nn.Sequential(
                nn.Conv2d(64, 256, 1, 1, 0),
                # nn.Conv2d(64, 256, 3, 2, 1),
                # nn.BatchNorm2d(256),
                # nn.GELU(),
                # nn.Conv2d(64, 96, 3, 2, 1),
        )

        self.norm = nn.LayerNorm(96)

        # self.conv = conv
        # if self.conv:
        #     self.ac = nn.Sequential(
        #         nn.Conv2d(64, 64, 3, 2, 1),
        #         nn.BatchNorm2d(64),
        #         nn.ReLU(inplace=True)
        #     )

        # self.stem_up = nn.Sequential(
        #     nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(32, momentum=0.8),
        #     nn.GELU(),

        #     nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(3, momentum=0.8),
        #     nn.GELU(),
        # )

        # self.stem_up = nn.Sequential(
        #     nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(32, momentum=0.8),
        #     nn.GELU(),

        #     nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(3, momentum=0.8),
        #     nn.GELU(),
        # )

        # self.stem_block6 = LKMB(96, 96)
        # self.stem_block7 = LKFFN(96)

        # self.se0 = SE(64)
        # self.se1 = SE(96)

        # self.ir1 = Upsample(128)
        # self.ir2 = Upsample(96)

        # self.zhuanzhi0 = nn.Sequential(
        #     nn.ConvTranspose2d(96, 64, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(64, momentum=0.8),
        #     nn.GELU(),
        # )

        # self.zhuanzhi1 = nn.Sequential(
        #     nn.ConvTranspose2d(128, 32, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(32, momentum=0.8),
        #     nn.GELU(),
        # )

        # self.sigmoid = nn.Sigmoid()

        # self.up1 = UNetUp(96, 64)

        # self.final0 = nn.Sequential(
        #     nn.Conv2d(131, 32, 3, 1, 1, bias=False),
        #      nn.BatchNorm2d(32),
        #      nn.GELU()
        # )

        # self.final = nn.Sequential(
        #     # nn.Upsample(scale_factor=2),
        #     # nn.ZeroPad2d((1, 0, 1, 0)),
        #     nn.Conv2d(35, 3, 3, 1, padding=1),
        #     # nn.Tanh(),
        # )

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

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         kaiming_init(m)
        #     elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
        #         constant_init(m, 1)

    def forward(self, x):
        ori = x
        """BigBoss
        x = self.input(x)
        # x = self.stem_block0(x)
        # x = self.stem_block1(x)
        
        # x = self.stem_block2(x)
        x = self.stem_b2(x)
        # x = self.se0(x)
        # l1 = x
        x = self.stem_block3(x)
        # x = self.stem_block4(x)
        # x = self.stem_block5(x)
        x = self.stem_b5(x)
        # x = self.se1(x)
        # x = self.stem_block6(x)
        # x = self.stem_block7(x)
        # x = self.stem_6(x)
        # over"""
        
        # x = self.patch1(x)
        # x = self.drop(x)
        x = self.mix1(x)
        # x = self.drop(x)
        x = self.mix2(x)

        mix2= x

        x = self.mix3(x)

        mix3 = x
        # x = self.ac(x)
        # x = self.stem_up(x)
        return mix2, mix3
        
        if self.conv:
            x = self.ac(x)
            return x
        # x = self.drop(x)
        x = self.patch2(x, mode='LN')
        # x = self.tiny1(x, ori)
        # x = self.tiny2(x, ori)
        # x = self.stem(x)
        #fuhao7i
        # x = self.zhuanzhi0(x)
        # # x = x + l1
        # x = torch.cat((x, l1), 1)
        # x = self.zhuanzhi1(x)
        # x = torch.cat((x, ori), 1)
        # x = self.final(x)

        # Wh, Ww = x.size(2), x.size(3)
        # x = x.flatten(2).transpose(1, 2)
        # x = self.norm(x)
        # x = x.transpose(1, 2).view(-1, 96, Wh, Ww)

        return x
        #over

        out = self.up1(x, l1)
        out = self.ir1(out)
        # out = torch.cat((out, l1), 1)
        # out = self.ir2(out)

        out = torch.cat((out, ori), 1)
        out = self.final0(out)
        out = torch.cat((out, ori), 1)
        out = self.final(out)


        # out = self.up1(out)
        # out = self.up2(out)
        # out = self.output(out)
        # out = self.sigmoid(out)
        # print(out.shape)
        return out


if __name__ == "__main__":

    model = AE(conv=True)