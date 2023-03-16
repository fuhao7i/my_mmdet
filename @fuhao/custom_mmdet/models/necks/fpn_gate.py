import warnings

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
from mmcv.runner import auto_fp16

# from ..builder import NECKS
from mmdet.models.builder import NECKS

import torch.nn as nn
from torch.nn import BatchNorm2d
from mmcv.cnn import (ConvModule, caffe2_xavier_init, constant_init, is_norm,
                      normal_init)


import torch
from torch import nn


class Bottleneck(nn.Module):
    """Bottleneck block for DilatedEncoder used in `YOLOF.

    <https://arxiv.org/abs/2103.09460>`.

    The Bottleneck contains three ConvLayers and one residual connection.

    Args:
        in_channels (int): The number of input channels.
        mid_channels (int): The number of middle output channels.
        dilation (int): Dilation rate.
        norm_cfg (dict): Dictionary to construct and config norm layer.
    """

    def __init__(self,
                 in_channels,
                 mid_channels,
                 dilation,
                 norm_cfg=dict(type='BN', requires_grad=True)):
        super(Bottleneck, self).__init__()
        self.conv1 = ConvModule(
            in_channels, mid_channels, 1, norm_cfg=norm_cfg)
        self.conv2 = ConvModule(
            mid_channels,
            mid_channels,
            3,
            padding=dilation,
            dilation=dilation,
            norm_cfg=norm_cfg)
        self.conv3 = ConvModule(
            mid_channels, in_channels, 1, norm_cfg=norm_cfg)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + identity
        return out    

@NECKS.register_module()
class FPN_gate_YOLOF_f(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest')):
        super(FPN_gate_YOLOF_f, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels_torch = in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # TODO: deprecate `extra_convs_on_inputs`
                warnings.simplefilter('once')
                warnings.warn(
                    '"extra_convs_on_inputs" will be deprecated in v2.9.0,'
                    'Please use "add_extra_convs"', DeprecationWarning)
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)
                
        # print('=> ', self.in_channels_torch)
        
        # self.fm5 = FocalModulation(dim=512)
        # self.fm4 = FocalModulation(dim=512)
        # self.fm3 = FocalModulation(dim=512)
        self.act = nn.GELU()
        
        C2_size, C3_size, C4_size, C5_size = self.in_channels_torch
        feature_size = out_channels
        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.P5_3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.P4_3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)    
        
        self.gate5 = nn.Linear(C5_size, 2 * feature_size + 1, bias=True)
        self.gate4 = nn.Linear(C4_size, 2 * feature_size + 1, bias=True)
        self.gate3 = nn.Linear(C3_size, 2 * feature_size + 1, bias=True)
        

        self.in_channels_f = 256
        self.out_channels_f = out_channels
        self.block_mid_channels = 128
        self.num_residual_blocks = 4
        self.block_dilations = [2, 4, 6, 8]   
        
        self.map5_0 = nn.Conv2d(self.out_channels, self.out_channels + 1, 3, 1, 1)
        self.map5_1 = nn.Conv2d(self.out_channels, self.out_channels + 1, 3, 1, 1)
        self.map5_2 = nn.Conv2d(self.out_channels, self.out_channels + 1, 3, 1, 1)
        self.map5_3 = nn.Conv2d(self.out_channels, self.out_channels + 1, 3, 1, 1)
        self.map5_d0 = Bottleneck(self.out_channels, self.out_channels // 2, dilation=2)
        self.map5_d1 = Bottleneck(self.out_channels, self.out_channels // 2, dilation=4)
        self.map5_d2 = Bottleneck(self.out_channels, self.out_channels // 2, dilation=6)
        self.map5_d3 = Bottleneck(self.out_channels, self.out_channels // 2, dilation=8)

        self.map4_0 = nn.Conv2d(self.out_channels, self.out_channels + 1, 3, 1, 1)
        self.map4_1 = nn.Conv2d(self.out_channels, self.out_channels + 1, 3, 1, 1)
        self.map4_2 = nn.Conv2d(self.out_channels, self.out_channels + 1, 3, 1, 1)
        self.map4_3 = nn.Conv2d(self.out_channels, self.out_channels + 1, 3, 1, 1)
        self.map4_d0 = Bottleneck(self.out_channels, self.out_channels // 2, dilation=2)
        self.map4_d1 = Bottleneck(self.out_channels, self.out_channels // 2, dilation=4)
        self.map4_d2 = Bottleneck(self.out_channels, self.out_channels // 2, dilation=6)
        self.map4_d3 = Bottleneck(self.out_channels, self.out_channels // 2, dilation=8)

        self.map3_0 = nn.Conv2d(self.out_channels, self.out_channels + 1, 3, 1, 1)
        self.map3_1 = nn.Conv2d(self.out_channels, self.out_channels + 1, 3, 1, 1)
        self.map3_2 = nn.Conv2d(self.out_channels, self.out_channels + 1, 3, 1, 1)
        self.map3_3 = nn.Conv2d(self.out_channels, self.out_channels + 1, 3, 1, 1)
        self.map3_d0 = Bottleneck(self.out_channels, self.out_channels // 2, dilation=2)
        self.map3_d1 = Bottleneck(self.out_channels, self.out_channels // 2, dilation=4)
        self.map3_d2 = Bottleneck(self.out_channels, self.out_channels // 2, dilation=6)
        self.map3_d3 = Bottleneck(self.out_channels, self.out_channels // 2, dilation=8)

        self.init_weights()

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        from mmcv.cnn import ConvModule, xavier_init
        from timm.models.layers import DropPath, to_2tuple, trunc_normal_
        from mmcv.cnn import (ConvModule, caffe2_xavier_init, constant_init, is_norm,
                      normal_init)
        from torch.nn.modules.batchnorm import _BatchNorm
        from mmcv.cnn import constant_init, kaiming_init
        """Initialize the weights of FPN module."""
        for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         xavier_init(m, distribution='uniform')
            
        # for m in self.fm5.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        # for m in self.fm5.modules():
        #     if isinstance(m, nn.Conv2d):
        #         kaiming_init(m)
        #     elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
        #         constant_init(m, 1)
        #     elif isinstance(m, nn.Linear):
        #         trunc_normal_(m.weight, std=.02)
        #         if isinstance(m, nn.Linear) and m.bias is not None:
        #             nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.LayerNorm):
        #         nn.init.constant_(m.bias, 0)
        #         nn.init.constant_(m.weight, 1.0)
                
        # for m in self.fm4.modules():
        #     if isinstance(m, nn.Conv2d):
        #         kaiming_init(m)
        #     elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
        #         constant_init(m, 1)
        #     elif isinstance(m, nn.Linear):
        #         trunc_normal_(m.weight, std=.02)
        #         if isinstance(m, nn.Linear) and m.bias is not None:
        #             nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.LayerNorm):
        #         nn.init.constant_(m.bias, 0)
        #         nn.init.constant_(m.weight, 1.0)
                
        # for m in self.fm3.modules():
        #     if isinstance(m, nn.Conv2d):
        #         kaiming_init(m)
        #     elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
        #         constant_init(m, 1)
        #     elif isinstance(m, nn.Linear):
        #         trunc_normal_(m.weight, std=.02)
        #         if isinstance(m, nn.Linear) and m.bias is not None:
        #             nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.LayerNorm):
        #         nn.init.constant_(m.bias, 0)
        #         nn.init.constant_(m.weight, 1.0)

    
    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        C2, C3, C4, C5 = list(inputs)
        # print(C2.shape)
        # """
        
        
        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        
        # C5 = C5.permute(0, 2, 3, 1).contiguous() # B, C, H, W -> B, H, W, C
        # P5_x = self.gate5(C5)
        # P5_x = P5_x.permute(0, 3, 1, 2).contiguous() # B, H, W, C -> B, C, H, W
        # C5 = C5.permute(0, 3, 1, 2).contiguous() # B, H, W, C -> B, C, H, W
        # _, C, _, _ = C5.shape
        # P5_x, up_P5_x, gates5 = torch.split(P5_x, (C, C, 1), 1)
        # P5_upsampled_x = self.P5_upsampled(up_P5_x * gates5)
        
        m5_0 = self.map5_0(P5_x)
        m5_0, g5_0 = m5_0[:, :self.out_channels, :, :], m5_0[:, self.out_channels, :, :]
        P5_d0 = self.map5_d0(m5_0) * g5_0.unsqueeze(1)

        m5_1 = self.map5_1(P5_x)
        m5_1, g5_1 = m5_1[:, :self.out_channels, :, :], m5_1[:, self.out_channels, :, :]
        P5_d1 = self.map5_d1(m5_1) * g5_1.unsqueeze(1)

        m5_2 = self.map5_2(P5_x)
        m5_2, g5_2 = m5_2[:, :self.out_channels, :, :], m5_2[:, self.out_channels, :, :]
        P5_d2 = self.map5_d2(m5_2) * g5_2.unsqueeze(1)

        m5_3 = self.map5_3(P5_x)
        m5_3, g5_3 = m5_3[:, :self.out_channels, :, :], m5_3[:, self.out_channels, :, :]
        P5_d3 = self.map5_d3(m5_3) * g5_3.unsqueeze(1) 
        
        P5_x = P5_d0 + P5_d1 + P5_d2 + P5_d3
        P5_x = self.act(P5_x) 
        P5_x = self.P5_2(P5_x)
        # P5_x = self.fm5(P5_x)
        # P5_x = self.act(P5_x)
        # P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_upsampled_x = self.P4_upsampled(P4_x)
        # C4 = C4.permute(0, 2, 3, 1).contiguous() # B, C, H, W -> B, H, W, C
        # P4_x = self.gate4(C4)
        # P4_x = P4_x.permute(0, 3, 1, 2).contiguous() # B, H, W, C -> B, C, H, W
        # _, C, _, _ = C4.shape
        # P4_x, up_P4_x, gates4 = torch.split(P4_x, (C, C, 1), 1)
        # P4_x = P4_x + P5_upsampled_x
        # P4_upsampled_x = self.P4_upsampled(P4_x * gates4)

        m5_0 = self.map4_0(P4_x)
        m5_0, g5_0 = m5_0[:, :self.out_channels, :, :], m5_0[:, self.out_channels, :, :]
        P5_d0 = self.map4_d0(m5_0) * g5_0.unsqueeze(1)

        m5_1 = self.map4_1(P4_x)
        m5_1, g5_1 = m5_1[:, :self.out_channels, :, :], m5_1[:, self.out_channels, :, :]
        P5_d1 = self.map4_d1(m5_1) * g5_1.unsqueeze(1)

        m5_2 = self.map4_2(P4_x)
        m5_2, g5_2 = m5_2[:, :self.out_channels, :, :], m5_2[:, self.out_channels, :, :]
        P5_d2 = self.map4_d2(m5_2) * g5_2.unsqueeze(1)

        m5_3 = self.map4_3(P4_x)
        m5_3, g5_3 = m5_3[:, :self.out_channels, :, :], m5_3[:, self.out_channels, :, :]
        P5_d3 = self.map4_d3(m5_3) * g5_3.unsqueeze(1) 
        
        P4_x = P5_d0 + P5_d1 + P5_d2 + P5_d3
        P4_x = self.act(P4_x)
        
        P4_x = self.P4_2(P4_x)
        # P4_x = self.fm4(P4_x)
        # P4_x = self.act(P4_x)
        # P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        # C3 = C3.permute(0, 2, 3, 1).contiguous() # B, C, H, W -> B, H, W, C
        # P3_x = self.gate3(C3)
        # P3_x = P3_x.permute(0, 3, 1, 2).contiguous() # B, H, W, C -> B, C, H, W
        # _, C, _, _ = P3_x.shape
        # P3_x, gates3 = torch.split(P3_x, (C - 1, 1), 1)
        P3_x = P3_x + P4_upsampled_x

        m5_0 = self.map3_0(P3_x)
        m5_0, g5_0 = m5_0[:, :self.out_channels, :, :], m5_0[:, self.out_channels, :, :]
        P5_d0 = self.map3_d0(m5_0) * g5_0.unsqueeze(1)

        m5_1 = self.map3_1(P3_x)
        m5_1, g5_1 = m5_1[:, :self.out_channels, :, :], m5_1[:, self.out_channels, :, :]
        P5_d1 = self.map3_d1(m5_1) * g5_1.unsqueeze(1)

        m5_2 = self.map3_2(P3_x)
        m5_2, g5_2 = m5_2[:, :self.out_channels, :, :], m5_2[:, self.out_channels, :, :]
        P5_d2 = self.map3_d2(m5_2) * g5_2.unsqueeze(1)

        m5_3 = self.map3_3(P3_x)
        m5_3, g5_3 = m5_3[:, :self.out_channels, :, :], m5_3[:, self.out_channels, :, :]
        P5_d3 = self.map3_d3(m5_3) * g5_3.unsqueeze(1)         

        P3_x = P5_d0 + P5_d1 + P5_d2 + P5_d3
        P3_x = self.act(P3_x)

        P3_x = self.P3_2(P3_x)
        # P3_x = self.fm3(P3_x)
        # P3_x = self.act(P3_x)
        # P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)
        # print(P3_x.shape, P4_x.shape, P5_x.shape, P6_x.shape, P7_x.shape)
        return tuple([P3_x, P4_x, P5_x, P6_x, P7_x])
