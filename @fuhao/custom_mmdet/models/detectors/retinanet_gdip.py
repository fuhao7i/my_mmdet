# import sys
# sys.path.append('@fuhao/')
# from custom_mmdet.models.backbones.resnet.resnet18 import ResNet18_FPN

from cProfile import label
from locale import normalize
from tkinter import E
from mmdet.core.bbox.transforms import bbox_mapping
import torch
import torch.nn as nn
from mmdet.core import bbox2result
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.single_stage import SingleStageDetector


import random
from torchvision.utils import save_image
import numpy as np

import os
import torch
import numpy as np
from PIL import Image, ImageFilter
from torchvision.transforms import ToTensor

from .tools.GatedDIP import GatedDIP
@DETECTORS.register_module()
class GatedDIP_RetinaNet(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 bbox_head_1=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(GatedDIP_RetinaNet, self).__init__(backbone, neck, bbox_head, bbox_head_1, train_cfg,
                                        test_cfg, pretrained)
        self.gated_dip = GatedDIP(256)  
    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        img, gates = self.gated_dip(img)
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

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
        
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore, 
                                              label_mask=None
                                                )

        
        return losses

    def simple_test(self, img, img_metas, rescale=False):

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


from .tools.GatedDIP_stem_gate import GatedDIP_stem_gate
@DETECTORS.register_module()
class GatedDIP_stem_gate_RetinaNet(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 bbox_head_1=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(GatedDIP_stem_gate_RetinaNet, self).__init__(backbone, neck, bbox_head, bbox_head_1, train_cfg,
                                        test_cfg, pretrained)
        self.gated_dip = GatedDIP_stem_gate(256)  
    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        img, gates = self.gated_dip(img)
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

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
        
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore, 
                                              label_mask=None
                                                )

        
        return losses

    def simple_test(self, img, img_metas, rescale=False):

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
    

from .tools.GatedAttentionDIP import GatedAttentionDIP
@DETECTORS.register_module()
class GatedAttentionDIP_RetinaNet(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 bbox_head_1=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(GatedAttentionDIP_RetinaNet, self).__init__(backbone, neck, bbox_head, bbox_head_1, train_cfg,
                                        test_cfg, pretrained)
        self.gated_dip = GatedAttentionDIP(256)  
    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        img, gates = self.gated_dip(img)
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

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
        
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore, 
                                              label_mask=None
                                                )

        
        return losses

    def simple_test(self, img, img_metas, rescale=False):

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


if __name__ == "__main__":
    net = GatedDIP_RetinaNet(
            backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages= -1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN_github',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        # in_channels=[64, 128, 256, 512],
        # out_channels=256,
        
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),

    bbox_head=dict(
        type='RetinaHead',
        num_classes=12,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
          
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))