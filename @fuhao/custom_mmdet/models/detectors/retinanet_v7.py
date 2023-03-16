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
import warnings
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
from mmcv.runner import auto_fp16
import torch.nn as nn
from torch.nn import BatchNorm2d
from mmcv.cnn import (ConvModule, caffe2_xavier_init, constant_init, is_norm,
                      normal_init)



@DETECTORS.register_module()
class RetinaNet_ori(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 bbox_head_1=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RetinaNet_ori, self).__init__(backbone, neck, bbox_head, bbox_head_1, train_cfg,
                                        test_cfg, pretrained)
        
    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

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
        
        x = self.extract_feat(img)
        # x, en_neck, loss_label_guide, label_guide_x = self.extract_feat_label_guide(img, img_metas, gt_bboxes)

        
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
        # losses.update({'loss_gen':loss_label_guide})
        
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
        x = self.extract_feat(img)
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

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        return [self.bbox_head.aug_test(feats, img_metas, rescale=rescale)]
    
    
    

class Decoder(nn.Module):
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
        super(Decoder, self).__init__()
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
        C1_size = 64
        C2_size, C3_size, C4_size, C5_size = self.in_channels_torch
        feature_size = out_channels
        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        
        self.P2_1 = nn.Conv2d(C2_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P2_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P2_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P1_1 = nn.Conv2d(C1_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P1_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P1_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)    

        self.output = nn.Conv2d(256, 3, 3, 1, 1)
        self.tanh = nn.Tanh()

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        """Initialize the weights of FPN module."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        C1, C2, C3, C4, C5 = list(inputs)
        
        #2,  4,  8, 16, 32 
        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_upsampled_x = self.P3_upsampled(P3_x)
        P3_x = self.P3_2(P3_x)
        
        # C2 
        P2_x = self.P2_1(C2)
        P2_x = P2_x + P3_upsampled_x
        # P2_upsampled_x = self.P2_upsampled(P2_x)
        
        # C1 
        P1_x = self.P1_1(C1)
        P1_x = P1_x + P2_x
        P1_upsampled_x = self.P1_upsampled(P1_x)
        
        gen_img = self.output(P1_upsampled_x)
        # gen_img = self.tanh(gen_img) # 规范化一下输出

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)
        # print(P3_x.shape, P4_x.shape, P5_x.shape, P6_x.shape, P7_x.shape)
        return tuple([P4_x, P5_x, P6_x, P7_x]), gen_img
# ----------------------------------------

# ----------------------------------------
# Resnet18
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_plugin_layer,
                      constant_init, kaiming_init)
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.utils import get_root_logger
from mmdet.models.utils import ResLayer



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None):
        super(BasicBlock, self).__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None):
        """Bottleneck block for ResNet.

        If style is "pytorch", the stride-two layer is the 3x3 conv layer, if
        it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        assert plugins is None or isinstance(plugins, list)
        if plugins is not None:
            allowed_position = ['after_conv1', 'after_conv2', 'after_conv3']
            assert all(p['position'] in allowed_position for p in plugins)

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.plugins = plugins
        self.with_plugins = plugins is not None

        if self.with_plugins:
            # collect plugins for conv1/conv2/conv3
            self.after_conv1_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv1'
            ]
            self.after_conv2_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv2'
            ]
            self.after_conv3_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv3'
            ]

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = dcn.pop('fallback_on_stride', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = build_conv_layer(
                conv_cfg,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)
        else:
            assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            self.conv2 = build_conv_layer(
                dcn,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        if self.with_plugins:
            self.after_conv1_plugin_names = self.make_block_plugins(
                planes, self.after_conv1_plugins)
            self.after_conv2_plugin_names = self.make_block_plugins(
                planes, self.after_conv2_plugins)
            self.after_conv3_plugin_names = self.make_block_plugins(
                planes * self.expansion, self.after_conv3_plugins)

    def make_block_plugins(self, in_channels, plugins):
        """make plugins for block.

        Args:
            in_channels (int): Input channels of plugin.
            plugins (list[dict]): List of plugins cfg to build.

        Returns:
            list[str]: List of the names of plugin.
        """
        assert isinstance(plugins, list)
        plugin_names = []
        for plugin in plugins:
            plugin = plugin.copy()
            name, layer = build_plugin_layer(
                plugin,
                in_channels=in_channels,
                postfix=plugin.pop('postfix', ''))
            assert not hasattr(self, name), f'duplicate plugin {name}'
            self.add_module(name, layer)
            plugin_names.append(name)
        return plugin_names

    def forward_plugin(self, x, plugin_names):
        out = x
        for name in plugin_names:
            out = getattr(self, name)(x)
        return out

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        """nn.Module: normalization layer after the third convolution layer"""
        return getattr(self, self.norm3_name)

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x
            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out
    
class ResNet18(nn.Module):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        stem_channels (int | None): Number of stem channels. If not specified,
            it will be the same as `base_channels`. Default: None.
        base_channels (int): Number of base channels of res layer. Default: 64.
        in_channels (int): Number of input image channels. Default: 3.
        num_stages (int): Resnet stages. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        plugins (list[dict]): List of plugins for stages, each dict contains:

            - cfg (dict, required): Cfg dict to build plugin.
            - position (str, required): Position inside block to insert
              plugin, options are 'after_conv1', 'after_conv2', 'after_conv3'.
            - stages (tuple[bool], optional): Stages to apply plugin, length
              should be same as 'num_stages'.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity.

    Example:
        >>> from mmdet.models import ResNet
        >>> import torch
        >>> self = ResNet(depth=18)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
    """

    arch_settings = {
         9: (BasicBlock, (1, 1, 1, 1)),
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    """
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    """

    def __init__(self,
                 depth=18,
                 in_channels=3,
                 stem_channels=None,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 plugins=None,
                 with_cp=False,
                 zero_init_residual=True):
        super(ResNet18, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.depth = depth
        if stem_channels is None:
            stem_channels = base_channels
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.plugins = plugins
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            if plugins is not None:
                stage_plugins = self.make_stage_plugins(plugins, i)
            else:
                stage_plugins = None
            planes = base_channels * 2**i
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                plugins=stage_plugins)
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.init_weights(pretrained='torchvision://resnet18')

        self.feat_dim = self.block.expansion * base_channels * 2**(
            len(self.stage_blocks) - 1)

    def make_stage_plugins(self, plugins, stage_idx):
        """Make plugins for ResNet ``stage_idx`` th stage.

        Currently we support to insert ``context_block``,
        ``empirical_attention_block``, ``nonlocal_block`` into the backbone
        like ResNet/ResNeXt. They could be inserted after conv1/conv2/conv3 of
        Bottleneck.

        An example of plugins format could be:

        Examples:
            >>> plugins=[
            ...     dict(cfg=dict(type='xxx', arg1='xxx'),
            ...          stages=(False, True, True, True),
            ...          position='after_conv2'),
            ...     dict(cfg=dict(type='yyy'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3'),
            ...     dict(cfg=dict(type='zzz', postfix='1'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3'),
            ...     dict(cfg=dict(type='zzz', postfix='2'),
            ...          stages=(True, True, True, True),
            ...          position='after_conv3')
            ... ]
            >>> self = ResNet(depth=18)
            >>> stage_plugins = self.make_stage_plugins(plugins, 0)
            >>> assert len(stage_plugins) == 3

        Suppose ``stage_idx=0``, the structure of blocks in the stage would be:

        .. code-block:: none

            conv1-> conv2->conv3->yyy->zzz1->zzz2

        Suppose 'stage_idx=1', the structure of blocks in the stage would be:

        .. code-block:: none

            conv1-> conv2->xxx->conv3->yyy->zzz1->zzz2

        If stages is missing, the plugin would be applied to all stages.

        Args:
            plugins (list[dict]): List of plugins cfg to build. The postfix is
                required if multiple same type plugins are inserted.
            stage_idx (int): Index of stage to build

        Returns:
            list[dict]: Plugins for current stage
        """
        stage_plugins = []
        for plugin in plugins:
            plugin = plugin.copy()
            stages = plugin.pop('stages', None)
            assert stages is None or len(stages) == self.num_stages
            # whether to insert plugin into current stage
            if stages is None or stages[stage_idx]:
                stage_plugins.append(plugin)

        return stage_plugins

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``."""
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem:
            self.stem = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels)[1],
                nn.ReLU(inplace=True))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.dcn is not None:
                for m in self.modules():
                    if isinstance(m, Bottleneck) and hasattr(
                            m.conv2, 'conv_offset'):
                        constant_init(m.conv2.conv_offset, 0)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        # x = self.maxpool(x) # 把Maxpooling操作给删了，防止尺寸下降太大
        outs = []
        outs.append(x)
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(ResNet18, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
# ----------------------------------------






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

def pil_to_np(img_PIL, with_transpose=True):
    """
    Converts image in PIL format to np.array.

    From W x H x C [0...255] to C x W x H [0..1]
    """
    ar = np.array(img_PIL)
    if len(ar.shape) == 3 and ar.shape[-1] == 4:
        ar = ar[:, :, :3]
        # this is alpha channel
    if with_transpose:
        if len(ar.shape) == 3:
            ar = ar.transpose(2, 0, 1)
        else:
            ar = ar[None, ...]

    return ar.astype(np.float32) / 255.


def np_to_torch(img_np):
    """
    Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]

    :param img_np:
    :return:
    """
    return torch.from_numpy(img_np)[None, :]


def torch_to_np(img_var):
    """
    Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    :param img_var:
    :return:
    """
    return img_var.detach().cpu().numpy()[0]


def np_to_pil(img_np):
    """
    Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    :param img_np:
    :return:
    """
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        assert img_np.shape[0] == 3, img_np.shape
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)


def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


def get_A(x):
    x_np = np.clip(torch_to_np(x), 0, 1)
    # x_np = np.clip(torch_to_np(x), -1, 1)
    x_pil = np_to_pil(x_np)
    h, w = x_pil.size
    windows = (h + w) / 2
    A = x_pil.filter(ImageFilter.GaussianBlur(windows))
    A = ToTensor()(A)
    return A.unsqueeze(0)


class ResNet18_Decoder(nn.Module):
    def __init__(self, in_channels=3, depth=18):
        super().__init__()

        self.backbone = ResNet18(depth=depth, in_channels=in_channels)
        self.neck = Decoder( 
            in_channels=[64, 128, 256, 512],
            out_channels=256,
            num_outs=5)
    
    def forward(self, x):
        outs = self.backbone(x)
        # for i in outs:
        #     print(i.shape)
        outs = self.neck(outs)
        return outs

@DETECTORS.register_module()
class RetinaNet_v711(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RetinaNet_v711, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained)
        
        self.mse_loss = torch.nn.MSELoss().cuda()
        
        self.fpn0 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.transfernet = ResNet18_Decoder(in_channels=3)

        # weights_path = '@fuhao/exp/500Adaptive/540mask_retina/541ori_retina/checkpoints/4450_epoch_12.pth'
        # print('+---------------------------------------------+')
        # print('+ Loading weights into ' + 'uwnet' + ' state dict.. +')
        # print('+ Weights path: ', weights_path)
        # print('+---------------------------------------------+')
        # model_dict = self.transfernet.state_dict()
        
        # num = 0
        # for k in model_dict.keys():
        #     print('model =>', k)
        #     num += 1
        # print('total num: ', num)
        # num = 0

        # pretrained_dict = torch.load(weights_path)
        # pretrained_dict = pretrained_dict['state_dict']

        # for k in pretrained_dict.keys():
        #     print('checkpoint =>', k)
        #     num += 1
        # print('total num: ', num)
        # num = 0

        # momo_dict = {}
        # for k, v in pretrained_dict.items(): 
        #     # if k in model_dict.keys() and cfg.load_checkpoint['load_keys'] in k: # 注意要修改的地方
        #     if k in model_dict.keys():
        #         if pretrained_dict[k].size() == model_dict[k].size():
        #             momo_dict.update({k: v})

        # # momo_dict = {k:v for 'module.'+k, v in pretrained_dict['state_dict'].items() if k in model_dict.keys() and pretrained_dict['state_dict'][k].shape() == model_dict[k].shape() }
        # for k, v in momo_dict.items():
        #     print('model load => ', k)
        #     num += 1
        # print('total num: ', num)

        # model_dict.update(momo_dict)
        # self.transfernet.load_state_dict(model_dict)
        # print('+---------------------------------------------+')
        # print('+                 Finished！                  +')
        # print('+---------------------------------------------+')
        
    def test_extract_feat_label_guide(self, img):
        """Directly extract features from the backbone+neck."""
        en_neck, gen_img = self.transfernet(img)
        x = self.backbone(gen_img)
        if self.with_neck:
            x = self.neck(x)
        
        x = en_neck + x
        
        loss = 0.0
        bbox_feat = None
        return x, bbox_feat
    
    
    def extract_feat_label_guide(self, img, img_metas, gt_bboxes):
        """Directly extract features from the backbone+neck."""
        en_neck, gen_img = self.transfernet(img)
        # with torch.no_grad():
            # self.backbone.eval()
        x = self.backbone(gen_img)
        if self.with_neck:
            x = self.neck(x)
        
        x = en_neck + x
        
        loss = 0.0
        bbox_feat = None
        return x, loss, bbox_feat

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

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
        
        # x = self.extract_feat(img)
        x, loss_lable_guide, label_guide_x = self.extract_feat_label_guide(img, img_metas, gt_bboxes)

        
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore, 
                                              label_mask=label_guide_x
                                                )
        
        # losses.update({'loss_label_guide':loss_lable_guide})
        
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
        # x = self.extract_feat(img)
        x, label_guide_x = self.test_extract_feat_label_guide(img)
        
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

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        return [self.bbox_head.aug_test(feats, img_metas, rescale=rescale)]




import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, bn=True):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if bn: layers.append(nn.BatchNorm2d(out_size, momentum=0.8))
        layers.append(nn.LeakyReLU(0.2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_size, momentum=0.8),
            nn.ReLU(inplace=True),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


class GeneratorFunieGAN(nn.Module):
    """ A 5-layer UNet-based generator as described in the paper
    """
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorFunieGAN, self).__init__()
        # encoding layers
        self.down1 = UNetDown(in_channels, 32, bn=False)
        self.down2 = UNetDown(32, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 256)
        self.down5 = UNetDown(256, 256, bn=False)
        # decoding layers
        self.up1 = UNetUp(256, 256)
        self.up2 = UNetUp(512, 256)
        self.up3 = UNetUp(512, 128)
        self.up4 = UNetUp(256, 32)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(64, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        u1 = self.up1(d5, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u45 = self.up4(u3, d1)
        return self.final(u45)
    
    

@DETECTORS.register_module()
class RetinaNet_v712(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 bbox_head_1=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RetinaNet_v712, self).__init__(backbone, neck, bbox_head, bbox_head_1, train_cfg,
                                        test_cfg, pretrained)
        
        self.mse_loss = torch.nn.MSELoss().cuda()
        
        self.fpn0 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.transfernet = ResNet18_Decoder(in_channels=3, depth=9)
        # self.funie = GeneratorFunieGAN()

        # weights_path = '@fuhao/exp/500Adaptive/540mask_retina/541ori_retina/checkpoints/4450_epoch_12.pth'
        # print('+---------------------------------------------+')
        # print('+ Loading weights into ' + 'uwnet' + ' state dict.. +')
        # print('+ Weights path: ', weights_path)
        # print('+---------------------------------------------+')
        # model_dict = self.transfernet.state_dict()
        
        # num = 0
        # for k in model_dict.keys():
        #     print('model =>', k)
        #     num += 1
        # print('total num: ', num)
        # num = 0

        # pretrained_dict = torch.load(weights_path)
        # pretrained_dict = pretrained_dict['state_dict']

        # for k in pretrained_dict.keys():
        #     print('checkpoint =>', k)
        #     num += 1
        # print('total num: ', num)
        # num = 0

        # momo_dict = {}
        # for k, v in pretrained_dict.items(): 
        #     # if k in model_dict.keys() and cfg.load_checkpoint['load_keys'] in k: # 注意要修改的地方
        #     if k in model_dict.keys():
        #         if pretrained_dict[k].size() == model_dict[k].size():
        #             momo_dict.update({k: v})

        # # momo_dict = {k:v for 'module.'+k, v in pretrained_dict['state_dict'].items() if k in model_dict.keys() and pretrained_dict['state_dict'][k].shape() == model_dict[k].shape() }
        # for k, v in momo_dict.items():
        #     print('model load => ', k)
        #     num += 1
        # print('total num: ', num)

        # model_dict.update(momo_dict)
        # self.transfernet.load_state_dict(model_dict)
        # print('+---------------------------------------------+')
        # print('+                 Finished！                  +')
        # print('+---------------------------------------------+')
        
    def test_extract_feat_label_guide(self, img):
        """Directly extract features from the backbone+neck."""
        en_neck, gen_img = self.transfernet(img)
        x = self.backbone(gen_img)
        if self.with_neck:
            x = self.neck(x)
        
        # x = en_neck + x
        
        loss = 0.0
        bbox_feat = None
        return x, en_neck, bbox_feat
    
    
    def extract_feat_label_guide(self, img, img_metas, gt_bboxes):
        """Directly extract features from the backbone+neck."""
        en_neck, gen_img = self.transfernet(img)
        # with torch.no_grad():
            # self.backbone.eval()
        x = self.backbone(gen_img)
        if self.with_neck:
            x = self.neck(x)
        
        # x = en_neck + x
        
        loss = 0.0
        bbox_feat = None
        return x, en_neck, loss, bbox_feat

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

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
        
        # x = self.extract_feat(img)
        x, en_neck, loss_lable_guide, label_guide_x = self.extract_feat_label_guide(img, img_metas, gt_bboxes)

        
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore, 
                                              label_mask=label_guide_x
                                                )
        en_losses = self.bbox_head_1.forward_train(en_neck, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore, 
                                              label_mask=label_guide_x
                                                )
        
        losses.update({'loss_en_neck_cls':en_losses['loss_cls']})
        losses.update({'loss_en_neck_bbox':en_losses['loss_bbox']})
        
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
        # x = self.extract_feat(img)
        x, en_neck, label_guide_x = self.test_extract_feat_label_guide(img)
        
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

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        return [self.bbox_head.aug_test(feats, img_metas, rescale=rescale)]


@DETECTORS.register_module()
class RetinaNet_v712_test(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 bbox_head_1=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RetinaNet_v712_test, self).__init__(backbone, neck, bbox_head, bbox_head_1, train_cfg,
                                        test_cfg, pretrained)
        
        self.mse_loss = torch.nn.MSELoss().cuda()
        
        self.fpn0 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.transfernet = ResNet18_Decoder(in_channels=3, depth=9)
        # self.funie = GeneratorFunieGAN()

        # weights_path = '@fuhao/exp/500Adaptive/540mask_retina/541ori_retina/checkpoints/4450_epoch_12.pth'
        weights_path = '@fuhao/exp/700/710/712/checkpoints/epoch_12.pth'
        print('+---------------------------------------------+')
        print('+ Loading weights into ' + 'uwnet' + ' state dict.. +')
        print('+ Weights path: ', weights_path)
        print('+---------------------------------------------+')
        model_dict = self.transfernet.state_dict()
        
        num = 0
        for k in model_dict.keys():
            print('model =>', k)
            num += 1
        print('total num: ', num)
        num = 0

        pretrained_dict = torch.load(weights_path)
        pretrained_dict = pretrained_dict['state_dict']

        for k in pretrained_dict.keys():
            print('checkpoint =>', k)
            num += 1
        print('total num: ', num)
        num = 0

        momo_dict = {}
        for k, v in pretrained_dict.items(): 
            # if k in model_dict.keys() and cfg.load_checkpoint['load_keys'] in k: # 注意要修改的地方
            if 'transfernet' not in k:
                continue
            k_model = k.replace('transfernet.', '')
            if k_model in model_dict.keys():
                if pretrained_dict[k].size() == model_dict[k_model].size():
                    momo_dict.update({k_model: v})

        # momo_dict = {k:v for 'module.'+k, v in pretrained_dict['state_dict'].items() if k in model_dict.keys() and pretrained_dict['state_dict'][k].shape() == model_dict[k].shape() }
        for k, v in momo_dict.items():
            print('model load => ', k)
            num += 1
        print('total num: ', num)

        model_dict.update(momo_dict)
        self.transfernet.load_state_dict(model_dict)
        print('+---------------------------------------------+')
        print('+                 Finished！                  +')
        print('+---------------------------------------------+')
        
    def test_extract_feat_label_guide(self, img):
        """Directly extract features from the backbone+neck."""
        en_neck, gen_img = self.transfernet(img)
        x = self.backbone(gen_img)
        if self.with_neck:
            x = self.neck(x)
        
        # x = en_neck + x
        
        loss = 0.0
        bbox_feat = None
        return x, en_neck, bbox_feat
    
    
    def extract_feat_label_guide(self, img, img_metas, gt_bboxes):
        """Directly extract features from the backbone+neck."""
        with torch.no_grad():
            self.transfernet.eval()
            en_neck, gen_img = self.transfernet(img)
            # save_image(img, './@fuhao/transfer/input.jpg', normalize=True)
            # save_image(gen_img, './@fuhao/transfer/gen_img.jpg', normalize=True)
        # with torch.no_grad():
            # self.backbone.eval()
        x = self.backbone(gen_img)
        if self.with_neck:
            x = self.neck(x)
        
        # x = en_neck + x
        
        loss = 0.0
        bbox_feat = None
        return x, en_neck, loss, bbox_feat

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

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
        
        # x = self.extract_feat(img)
        x, en_neck, loss_lable_guide, label_guide_x = self.extract_feat_label_guide(img, img_metas, gt_bboxes)

        
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore, 
                                              label_mask=label_guide_x
                                                )
        # en_losses = self.bbox_head_1.forward_train(en_neck, img_metas, gt_bboxes,
        #                                       gt_labels, gt_bboxes_ignore, 
        #                                       label_mask=label_guide_x
        #                                         )
        
        # losses.update({'loss_en_neck_cls':en_losses['loss_cls']})
        # losses.update({'loss_en_neck_bbox':en_losses['loss_bbox']})
        
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
        # x = self.extract_feat(img)
        x, en_neck, label_guide_x = self.test_extract_feat_label_guide(img)
        
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

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        return [self.bbox_head.aug_test(feats, img_metas, rescale=rescale)]
    
    
@DETECTORS.register_module()
class RetinaNet_v713(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 bbox_head_1=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RetinaNet_v713, self).__init__(backbone, neck, bbox_head, bbox_head_1, train_cfg,
                                        test_cfg, pretrained)
        
        self.mse_loss = torch.nn.MSELoss().cuda()
        
        self.fpn0 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.transfernet = ResNet18_Decoder(in_channels=3, depth=9)
        self.transfernet.cuda()
        # self.funie = GeneratorFunieGAN()

        weights_path = '@fuhao/exp/700/710/713/checkpoints/epoch_12.pth'
        print('+---------------------------------------------+')
        print('+ Loading weights into ' + 'uwnet' + ' state dict.. +')
        print('+ Weights path: ', weights_path)
        print('+---------------------------------------------+')
        model_dict = self.transfernet.state_dict()
        
        num = 0
        for k in model_dict.keys():
            print('model =>', k)
            num += 1
        print('total num: ', num)
        num = 0

        pretrained_dict = torch.load(weights_path)
        pretrained_dict = pretrained_dict['state_dict']

        for k in pretrained_dict.keys():
            print('checkpoint =>', k)
            num += 1
        print('total num: ', num)
        num = 0

        momo_dict = {}
        for k, v in pretrained_dict.items(): 
            # if k in model_dict.keys() and cfg.load_checkpoint['load_keys'] in k: # 注意要修改的地方
            if 'transfernet' not in k:
                continue
            k_model = k.replace('transfernet.', '')
            if k_model in model_dict.keys():
                if pretrained_dict[k].size() == model_dict[k_model].size():
                    print('state_dict:', k, 'model_dict:', k_model)
                    momo_dict.update({k_model: v})

        # momo_dict = {k:v for 'module.'+k, v in pretrained_dict['state_dict'].items() if k in model_dict.keys() and pretrained_dict['state_dict'][k].shape() == model_dict[k].shape() }
        for k, v in momo_dict.items():
            print('model load => ', k)
            num += 1
        print('total num: ', num)

        model_dict.update(momo_dict)
        self.transfernet.load_state_dict(model_dict)
        print('+---------------------------------------------+')
        print('+                 Finished！                  +')
        print('+---------------------------------------------+')
        
    def test_extract_feat_label_guide(self, img):
        """Directly extract features from the backbone+neck."""
        en_neck, gen_img = self.transfernet(img)
        x = self.backbone(gen_img)
        if self.with_neck:
            x = self.neck(x)
        
        # x = en_neck + x
        
        loss = 0.0
        bbox_feat = None
        return x, en_neck, bbox_feat
    
    
    def extract_feat_label_guide(self, img, img_metas, gt_bboxes):
        """Directly extract features from the backbone+neck."""
        en_neck, gen_img = self.transfernet(img)
        
        lam = np.random.beta(1, 1)
        input_mix = lam * img + (1 - lam) * gen_img
        _, gen_img_mix = self.transfernet(input_mix)
        loss_gen = self.mse_loss(gen_img_mix, gen_img.detach())

        # save_image(img, './@fuhao/transfer/input.jpg', normalize=True)
        # save_image(input_mix, './@fuhao/transfer/input_mix.jpg', normalize=True)
        # save_image(gen_img, './@fuhao/transfer/gen_img.jpg', normalize=True)
        # save_image(gen_img_mix, './@fuhao/transfer/gen_img_mix.jpg', normalize=True)
        # with torch.no_grad():
        # self.backbone.eval()
        x = self.backbone(gen_img)
        if self.with_neck:
            x = self.neck(x)
        
        # x = en_neck + x
        
        loss = loss_gen
        bbox_feat = None
        return x, en_neck, loss, bbox_feat

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

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
        
        # x = self.extract_feat(img)
        x, en_neck, loss_label_guide, label_guide_x = self.extract_feat_label_guide(img, img_metas, gt_bboxes)

        
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore, 
                                              label_mask=label_guide_x
                                                )
        en_losses = self.bbox_head_1.forward_train(en_neck, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore, 
                                              label_mask=label_guide_x
                                                )
        
        losses.update({'loss_en_neck_cls':en_losses['loss_cls']})
        losses.update({'loss_en_neck_bbox':en_losses['loss_bbox']})
        losses.update({'loss_gen':loss_label_guide})
        
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
        # x = self.extract_feat(img)
        x, en_neck, label_guide_x = self.test_extract_feat_label_guide(img)
        
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

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        return [self.bbox_head.aug_test(feats, img_metas, rescale=rescale)]

@DETECTORS.register_module()
class RetinaNet_v713_test(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 bbox_head_1=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RetinaNet_v713_test, self).__init__(backbone, neck, bbox_head, bbox_head_1, train_cfg,
                                        test_cfg, pretrained)
        
        self.mse_loss = torch.nn.MSELoss().cuda()
        
        self.fpn0 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.transfernet = ResNet18_Decoder(in_channels=3, depth=9)
        self.transfernet.cuda()
        # self.funie = GeneratorFunieGAN()

        weights_path = '@fuhao/exp/700/710/713/checkpoints/epoch_12.pth'
        print('+---------------------------------------------+')
        print('+ Loading weights into ' + 'uwnet' + ' state dict.. +')
        print('+ Weights path: ', weights_path)
        print('+---------------------------------------------+')
        model_dict = self.transfernet.state_dict()
        
        num = 0
        for k in model_dict.keys():
            print('model =>', k)
            num += 1
        print('total num: ', num)
        num = 0

        pretrained_dict = torch.load(weights_path)
        pretrained_dict = pretrained_dict['state_dict']

        for k in pretrained_dict.keys():
            print('checkpoint =>', k)
            num += 1
        print('total num: ', num)
        num = 0

        momo_dict = {}
        for k, v in pretrained_dict.items(): 
            # if k in model_dict.keys() and cfg.load_checkpoint['load_keys'] in k: # 注意要修改的地方
            if 'transfernet' not in k:
                continue
            k_model = k.replace('transfernet.', '')
            if k_model in model_dict.keys():
                if pretrained_dict[k].size() == model_dict[k_model].size():
                    print('state_dict:', k, 'model_dict:', k_model)
                    momo_dict.update({k_model: v})

        # momo_dict = {k:v for 'module.'+k, v in pretrained_dict['state_dict'].items() if k in model_dict.keys() and pretrained_dict['state_dict'][k].shape() == model_dict[k].shape() }
        for k, v in momo_dict.items():
            print('model load => ', k)
            num += 1
        print('total num: ', num)

        model_dict.update(momo_dict)
        self.transfernet.load_state_dict(model_dict)
        print('+---------------------------------------------+')
        print('+                 Finished！                  +')
        print('+---------------------------------------------+')
        
    def test_extract_feat_label_guide(self, img):
        """Directly extract features from the backbone+neck."""
        en_neck, gen_img = self.transfernet(img)
        x = self.backbone(gen_img)
        if self.with_neck:
            x = self.neck(x)
        
        # x = en_neck + x
        
        loss = 0.0
        bbox_feat = None
        return x, en_neck, bbox_feat
    
    
    def extract_feat_label_guide(self, img, img_metas, gt_bboxes):
        """Directly extract features from the backbone+neck."""
        # with torch.no_grad():
        #     self.transfernet.eval()
        en_neck, gen_img = self.transfernet(img)
        
        # lam = np.random.beta(1, 1)
        # input_mix = lam * img + (1 - lam) * gen_img
        # _, gen_img_mix = self.transfernet(input_mix)
        # loss_gen = self.mse_loss(gen_img_mix, gen_img.detach())

        # save_image(img, './@fuhao/transfer/input.jpg', normalize=True)
        # save_image(input_mix, './@fuhao/transfer/input_mix.jpg', normalize=True)
        # save_image(gen_img, './@fuhao/transfer/gen_img.jpg', normalize=True)
        # save_image(gen_img_mix, './@fuhao/transfer/gen_img_mix.jpg', normalize=True)
        # with torch.no_grad():
        # self.backbone.eval()
        x = self.backbone(gen_img)
        if self.with_neck:
            x = self.neck(x)
        
        # x = en_neck + x
        
        loss = 0.0
        bbox_feat = None
        return x, en_neck, loss, bbox_feat

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

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
        
        # x = self.extract_feat(img)
        x, en_neck, loss_label_guide, label_guide_x = self.extract_feat_label_guide(img, img_metas, gt_bboxes)

        
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore, 
                                              label_mask=label_guide_x
                                                )
        # en_losses = self.bbox_head_1.forward_train(en_neck, img_metas, gt_bboxes,
        #                                       gt_labels, gt_bboxes_ignore, 
        #                                       label_mask=label_guide_x
        #                                         )
        
        # losses.update({'loss_en_neck_cls':en_losses['loss_cls']})
        # losses.update({'loss_en_neck_bbox':en_losses['loss_bbox']})
        # losses.update({'loss_gen':loss_label_guide})
        
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
        # x = self.extract_feat(img)
        x, en_neck, label_guide_x = self.test_extract_feat_label_guide(img)
        
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

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        return [self.bbox_head.aug_test(feats, img_metas, rescale=rescale)]
    
    

# -------------------------------------------------------------------------------
# 714 
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalModulation(nn.Module):
    """ Focal Modulation

    Args:
        dim (int): Number of input channels.
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        focal_level (int): Number of focal levels
        focal_window (int): Focal window size at focal level 1
        focal_factor (int, default=2): Step to increase the focal window
        use_postln (bool, default=False): Whether use post-modulation layernorm
    """

    def __init__(self, dim, proj_drop=0., focal_level=2, focal_window=7, focal_factor=2, use_postln=False):

        super().__init__()
        self.dim = dim

        # specific args for focalv3
        self.focal_level = focal_level
        self.focal_window = focal_window
        self.focal_factor = focal_factor
        self.use_postln = use_postln

        self.f = nn.Linear(dim, 2*dim+(self.focal_level+1), bias=True)
        self.h = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, groups=1, bias=True)

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.focal_layers = nn.ModuleList()

        if self.use_postln:
            self.ln = nn.LayerNorm(dim)

        for k in range(self.focal_level):
            kernel_size = self.focal_factor*k + self.focal_window
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, groups=dim, 
                        padding=kernel_size//2, bias=False),
                    nn.GELU(),
                    )
                )

    def forward(self, x):
        """ Forward function.

        Args:
            x: input features with shape of (B, H, W, C)
        """
        # 需要改一下通道位置 B, C, H, W -> B, H, W, C 
        x = x.permute(0, 2, 3, 1).contiguous()
        
        B, nH, nW, C = x.shape
        x = self.f(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        q, ctx, gates = torch.split(x, (C, C, self.focal_level+1), 1)
        
        ctx_all = 0
        for l in range(self.focal_level):                     
            ctx = self.focal_layers[l](ctx)
            ctx_all = ctx_all + ctx*gates[:, l:l+1]
        ctx_global = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
        ctx_all = ctx_all + ctx_global*gates[:,self.focal_level:]

        x_out = q * self.h(ctx_all)
        x_out = x_out.permute(0, 2, 3, 1).contiguous()
        if self.use_postln:
            x_out = self.ln(x_out)            
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        
        x_out = x_out.permute(0, 3, 1, 2).contiguous()
        return x_out

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, bn=True):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 3, 2, 1, bias=False)]
        if bn: layers.append(nn.BatchNorm2d(out_size, momentum=0.8))
        # layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.GELU())
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUp, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_size, out_size, 3, 1, 1)
        self.focalmodulation = FocalModulation(dim=out_size)
        # layers = [
        #     nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
        #     nn.BatchNorm2d(out_size, momentum=0.8),
        #     nn.ReLU(inplace=True),
        # ]
        # self.model = nn.Sequential(*layers)
        

    def forward(self, x, skip_input):
        # x = self.model(x)
        # x = torch.cat((x, skip_input), 1)
        x = self.upsample(x)
        x = self.conv(x)
        x = x + skip_input
        x = self.focalmodulation(x)
        return x


class GeneratorFocalModulation(nn.Module):
    """ A 5-layer UNet-based generator as described in the paper
    """
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorFocalModulation, self).__init__()
        # encoding layers
        self.down1 = UNetDown(in_channels, 32, bn=False)
        self.down2 = UNetDown(32, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 256)
        self.down5 = UNetDown(256, 256, bn=False)
        # decoding layers
        self.up1 = UNetUp(256, 256)
        self.up2 = UNetUp(256, 256)
        self.up3 = UNetUp(256, 128)
        self.up4 = UNetUp(128, 32)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            # nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(32, out_channels, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        u1 = self.up1(d5, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u45 = self.up4(u3, d1)
        return self.final(u45)


@DETECTORS.register_module()
class RetinaNet_v714(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 bbox_head_1=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RetinaNet_v714, self).__init__(backbone, neck, bbox_head, bbox_head_1, train_cfg,
                                        test_cfg, pretrained)
        
        self.generator = GeneratorFocalModulation()
        
    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        img = self.generator(img)
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

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
        
        x = self.extract_feat(img)
        # x, en_neck, loss_label_guide, label_guide_x = self.extract_feat_label_guide(img, img_metas, gt_bboxes)

        
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
        # losses.update({'loss_gen':loss_label_guide})
        
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
        x = self.extract_feat(img)
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

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        return [self.bbox_head.aug_test(feats, img_metas, rescale=rescale)]