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
from custom_mmdet.models.backbones.resnet.resnet18 import ResNet18_FPN

import random
from torchvision.utils import save_image
import numpy as np

import os
import torch
import numpy as np
from PIL import Image, ImageFilter
from torchvision.transforms import ToTensor

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


@DETECTORS.register_module()
class Custom_RetinaNet(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(Custom_RetinaNet, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained)
        
        self.mse_loss = torch.nn.MSELoss().cuda()
        
        self.fpn0 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # self.labelnet = ResNet18_FPN(in_channels=1)
        
        # self.masknet = ResNet18_FPN(in_channels=3)

        self.transfernet = ResNet18_FPN(in_channels=3)
        # self.transfernet.eval()
        self.convtransfer = nn.Sequential(
            nn.Conv2d(256, 3, 3, 1, 1),
            nn.Sigmoid()
        )

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

        with torch.no_grad():
            self.transfernet.eval()
            label_x = self.transfernet(img)
            label_x = list(label_x)
            T = self.convtransfer(label_x[0])
            T = nn.functional.interpolate(T, size=[img.shape[2], img.shape[3]], mode='bilinear', align_corners=False)
            img = img / (1 - T)
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        
        # x = list(x)
        # label_guide_x = self.masknet(img)
        # label_guide_x0 = self.fpn0(x[0])
        # label_guide_x1 = self.fpn1(x[1])
        # label_guide_x2 = self.fpn2(x[2])
        # label_guide_x3 = self.fpn3(x[3])
        # label_guide_x4 = self.fpn4(x[4])

        # label_mask0 =  torch.where(label_guide_x0>0.5, 1.0, 0.0)
        # x[0] = x[0] * label_mask0 
        # label_guide_x = tuple([label_guide_x0, label_guide_x1, label_guide_x2, label_guide_x3, label_guide_x4])
        # label_guide_x = tuple([label_guide_x0, None, None, None, None])
        label_guide_x = tuple([None, None, None, None, None])
        
        return x, label_guide_x
    
    
    def extract_feat_label_guide(self, img, img_metas, gt_bboxes):
        """Directly extract features from the backbone+neck."""
        # x = self.backbone(img)
        # if self.with_neck:
        #     x = self.neck(x)
            
        # bbox_feat = self.masknet(img)
            
        # for i in list(x):
        #     print(i.shape)
            
        N, C, H, W = img.shape
        # Mask_fg = torch.zeros_like(S_attention_t)
        # Mask_bg = torch.ones_like(S_attention_t)
        Mask_fg = torch.zeros([N, H, W]).cuda()
        Mask_bg = torch.ones([N, H, W]).cuda()
        wmin,wmax,hmin,hmax = [],[],[],[]
        # for i in range(N):
        #     new_boxxes = torch.ones_like(gt_bboxes[i])

        #     # """ fuhao7i
        #     new_boxxes[:, 0] = gt_bboxes[i][:, 0]/img_metas[i]['img_shape'][1]*W
        #     new_boxxes[:, 2] = gt_bboxes[i][:, 2]/img_metas[i]['img_shape'][1]*W
        #     new_boxxes[:, 1] = gt_bboxes[i][:, 1]/img_metas[i]['img_shape'][0]*H
        #     new_boxxes[:, 3] = gt_bboxes[i][:, 3]/img_metas[i]['img_shape'][0]*H
        #     # """

        #     wmin.append(torch.floor(new_boxxes[:, 0]).int())
        #     wmax.append(torch.ceil(new_boxxes[:, 2]).int())
        #     hmin.append(torch.floor(new_boxxes[:, 1]).int())
        #     hmax.append(torch.ceil(new_boxxes[:, 3]).int())

        #     area = 1.0/(hmax[i].view(1,-1)+1-hmin[i].view(1,-1))/(wmax[i].view(1,-1)+1-wmin[i].view(1,-1))
    

        #     for j in range(len(gt_bboxes[i])):
        #         Mask_fg[i][hmin[i][j]:hmax[i][j]+1, wmin[i][j]:wmax[i][j]+1] = \
        #                 torch.maximum(Mask_fg[i][hmin[i][j]:hmax[i][j]+1, wmin[i][j]:wmax[i][j]+1], area[0][j])
        #     Mask_bg[i] = torch.where(Mask_fg[i]>0, 0, 1)
        #     if torch.sum(Mask_bg[i]):
        #         Mask_bg[i] /= torch.sum(Mask_bg[i])
        
        for i in range(N):
            new_boxxes = torch.ones_like(gt_bboxes[i])
            # print(gt_bboxes[i], img_metas[i]['img_shape'][1], W)
            # """ fuhao7i
            new_boxxes[:, 0] = gt_bboxes[i][:, 0]/img_metas[i]['img_shape'][1]*W
            new_boxxes[:, 2] = gt_bboxes[i][:, 2]/img_metas[i]['img_shape'][1]*W
            new_boxxes[:, 1] = gt_bboxes[i][:, 1]/img_metas[i]['img_shape'][0]*H
            new_boxxes[:, 3] = gt_bboxes[i][:, 3]/img_metas[i]['img_shape'][0]*H
            # """
            # print('new_bboxes: ', new_boxxes)
            wmin.append(torch.floor(new_boxxes[:, 0]).int())
            wmax.append(torch.ceil(new_boxxes[:, 2]).int())
            hmin.append(torch.floor(new_boxxes[:, 1]).int())
            hmax.append(torch.ceil(new_boxxes[:, 3]).int())

            # area = 1.0/(hmax[i].view(1,-1)+1-hmin[i].view(1,-1))/(wmax[i].view(1,-1)+1-wmin[i].view(1,-1))
    

            for j in range(len(gt_bboxes[i])):
                Mask_fg[i][hmin[i][j]:hmax[i][j]+1, wmin[i][j]:wmax[i][j]+1] = 1.0
                # Mask_fg[i][max(hmin[i][j]-20, 0):min(hmax[i][j]+20, H), max(wmin[i][j]-20, 0):min(wmax[i][j]+20, W)] = 1.0
                # Mask_fg[i][hmin[i][j]:hmax[i][j]+1, wmin[i][j]:wmax[i][j]+1] = torch.maximum(Mask_fg[i][hmin[i][j]:hmax[i][j]+1, wmin[i][j]:wmax[i][j]+1], area[0][j])
                # print(j, '置为1')
                # print(Mask_fg[i].shape)
                        

            Mask_bg[i] = torch.where(Mask_fg[i]>0, 0, 1)
                
                
        # t = Mask_fg.data.cpu().numpy()
        # # fig = sns.heatmap(data=t[0], vmax=9., vmin=6.5, cmap='viridis') 
        # # fig = sns.heatmap(data=t[0], vmax=4., vmin=3.4, cmap='viridis')
        # fig = sns.heatmap(data=t[0], cmap='viridis')  
        # heatmap = fig.get_figure()
        # # heatmap.savefig(output_images_path + name[0], dpi=400)
        # heatmap.savefig('@fuhao/transfer/Mask_fg.png', dpi=400)
        # heatmap.clear()
        
        # Mask_fg = Mask_fg.unsqueeze(1)
        # mask_img = Mask_fg * img
        # save_image(mask_img, '@fuhao/transfer/mask_img.jpg', normalize=True)
        # Mask_fg = Mask_fg.squeeze()
        
        # x = list(x)

        Mask_fg = Mask_fg.unsqueeze(1)
        
        # mask_input = img * Mask_fg
        # with torch.no_grad():
            # self.transfernet.eval()
        img_t = img.detach()
        label_guide_x = self.transfernet(img_t)
        
        # bbox_feat = list(bbox_feat)
        label_guide_x = list(label_guide_x)

        T = self.convtransfer(label_guide_x[0])
        T = nn.functional.interpolate(T, size=[img.shape[2], img.shape[3]], mode='bilinear', align_corners=False)

        img = img / (1 - T)

        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        
        # loss0 = self.mse_loss(bbox_feat[0], label_guide_x[0])
        # loss1 = self.mse_loss(bbox_feat[1], label_guide_x[1])
        # loss2 = self.mse_loss(bbox_feat[2], label_guide_x[2])
        # loss3 = self.mse_loss(bbox_feat[3], label_guide_x[3])
        # loss4 = self.mse_loss(bbox_feat[4], label_guide_x[4])
        
        # bbox_feat = tuple(bbox_feat)
        # label_guide_x = tuple(label_guide_x)
        
        loss = 0.0
        
        # label_guide_x0 = self.fpn0(x[0])
        # label_guide_x1 = self.fpn1(x[1])
        # label_guide_x2 = self.fpn2(x[2])
        # label_guide_x3 = self.fpn3(x[3])
        # label_guide_x4 = self.fpn4(x[4])
        
        # label_guide_x0 = torch.abs(x[0]).mean(1)
        # label_guide_x1 = torch.abs(x[1]).mean(1)
        # label_guide_x2 = torch.abs(x[2]).mean(1)
        # label_guide_x3 = torch.abs(x[3]).mean(1)
        # label_mask0 =  torch.where(label_guide_x0>0.5, 1.0, 0.0)
        # x[0] = x[0] * label_mask0
        
        # mask0 = nn.functional.interpolate(label_mask0, size=[img.shape[2], img.shape[3]], mode='bilinear', align_corners=False)
        # img = img * mask0
        # save_image(img, '@fuhao/transfer/mask0_img.jpg', normalize=True)
        
        
        # t = label_guide_x0.data.cpu().numpy()
        # # fig = sns.heatmap(data=t[0], vmax=9., vmin=6.5, cmap='viridis') 
        # # fig = sns.heatmap(data=t[0], vmax=4., vmin=3.4, cmap='viridis')
        # fig = sns.heatmap(data=t[0][0], cmap='viridis')  
        # heatmap = fig.get_figure()
        # # heatmap.savefig(output_images_path + name[0], dpi=400)
        # heatmap.savefig('@fuhao/transfer/T.png', dpi=400)
        # heatmap.clear()        

        
        # label_0 = nn.functional.interpolate(Mask_fg, size=[x[0].shape[2], x[0].shape[3]], mode='bilinear', align_corners=False)
        # label_1 = nn.functional.interpolate(Mask_fg, size=[x[1].shape[2], x[1].shape[3]], mode='bilinear', align_corners=False)
        # label_2 = nn.functional.interpolate(Mask_fg, size=[x[2].shape[2], x[2].shape[3]], mode='bilinear', align_corners=False)
        # label_3 = nn.functional.interpolate(Mask_fg, size=[x[3].shape[2], x[3].shape[3]], mode='bilinear', align_corners=False)
        # label_4 = nn.functional.interpolate(Mask_fg, size=[x[4].shape[2], x[4].shape[3]], mode='bilinear', align_corners=False)
        
        
        # loss0 = self.mse_loss(label_guide_x0, label_0)
        # loss1 = self.mse_loss(label_guide_x1, label_1)
        # loss2 = self.mse_loss(label_guide_x2, label_2)
        # loss3 = self.mse_loss(label_guide_x3, label_3)
        # loss4 = self.mse_loss(label_guide_x4, label_4)
        
        # loss = loss0 + loss1 + loss2 + loss3 + loss4
        # loss = loss0

        # t = label_0.data.cpu().numpy()
        # fig = sns.heatmap(data=t[0][0], cmap='viridis')  
        # heatmap = fig.get_figure()
        # heatmap.savefig('@fuhao/transfer/label0.png', dpi=400)
        # heatmap.clear()      
        
        # t = label_mask0.data.cpu().numpy()
        # fig = sns.heatmap(data=t[0][0], cmap='viridis')  
        # heatmap = fig.get_figure()
        # heatmap.savefig('@fuhao/transfer/label_mask0.png', dpi=400)
        # heatmap.clear()       
        
        # label_guide_x = tuple([label_guide_x0, label_guide_x1, label_guide_x2, label_guide_x3, label_guide_x4])
        # label_guide_x = tuple([label_guide_x0, None, None, None, None])


        # mask_img = Mask_fg * img
        # save_image(mask_img, '@fuhao/transfer/mask_img.jpg', normalize=True)
        
        # label_guide_x0 = torch.where(label_guide_x0>0.2, 1.0, 0.)
        # label_guide_x0 = nn.functional.interpolate(label_guide_x0, size=[img.shape[2], img.shape[3]], mode='bilinear', align_corners=False)
        # label_guide_img = label_guide_x0 * img
        # save_image(label_guide_img, '@fuhao/transfer/label_guide_img.jpg', normalize=True)
        
        # x = tuple(x)
        bbox_feat = tuple([None, None, None, None, None])
        # return x, loss, label_guide_x        
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
        
        outs = self.bbox_head(x, label_guide_x)
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
class Custom_RetinaNet_v543(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(Custom_RetinaNet_v543, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained)
        
        self.mse_loss = torch.nn.MSELoss().cuda()
        
        self.fpn0 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.mae_decoder = nn.Sequential(
            nn.Conv2d(256, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

        # self.labelnet = ResNet18_FPN(in_channels=1)
        
        # self.masknet = ResNet18_FPN(in_channels=3)

        # self.transfernet = ResNet18_FPN(in_channels=3)
        # self.transfernet.eval()
        # self.convtransfer = nn.Sequential(
        #     nn.Conv2d(256, 3, 3, 1, 1),
        #     nn.Sigmoid()
        # )

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

        # with torch.no_grad():
        #     self.transfernet.eval()
        #     label_x = self.transfernet(img)
        #     label_x = list(label_x)
        #     T = self.convtransfer(label_x[0])
        #     T = nn.functional.interpolate(T, size=[img.shape[2], img.shape[3]], mode='bilinear', align_corners=False)
        #     # img = img / (1 - T)
        #     img = img * T + img
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        
        # x = list(x)
        # label_guide_x = self.masknet(img)
        # label_guide_x0 = self.fpn0(x[0])
        # label_guide_x1 = self.fpn1(x[1])
        # label_guide_x2 = self.fpn2(x[2])
        # label_guide_x3 = self.fpn3(x[3])
        # label_guide_x4 = self.fpn4(x[4])

        # label_mask0 =  torch.where(label_guide_x0>0.5, 1.0, 0.0)
        # x[0] = x[0] * label_mask0 
        # label_guide_x = tuple([label_guide_x0, label_guide_x1, label_guide_x2, label_guide_x3, label_guide_x4])
        # label_guide_x = tuple([label_guide_x0, None, None, None, None])
        label_guide_x = tuple([None, None, None, None, None])
        
        return x, label_guide_x
    
    
    def extract_feat_label_guide(self, img, img_metas, gt_bboxes):
        """Directly extract features from the backbone+neck."""

        # x = self.backbone(img)
        # if self.with_neck:
        #     x = self.neck(x)
            
        # bbox_feat = self.masknet(img)
            
        # for i in list(x):
        #     print(i.shape)
            
        # N, C, H, W = img.shape
        # # Mask_fg = torch.zeros_like(S_attention_t)
        # # Mask_bg = torch.ones_like(S_attention_t)
        # Mask_fg = torch.ones([N, H, W]).cuda()
        # # Mask_bg = torch.ones([N, H, W]).cuda()
        # wmin,wmax,hmin,hmax = [],[],[],[]
        # # for i in range(N):
        # #     new_boxxes = torch.ones_like(gt_bboxes[i])

        # #     # """ fuhao7i
        # #     new_boxxes[:, 0] = gt_bboxes[i][:, 0]/img_metas[i]['img_shape'][1]*W
        # #     new_boxxes[:, 2] = gt_bboxes[i][:, 2]/img_metas[i]['img_shape'][1]*W
        # #     new_boxxes[:, 1] = gt_bboxes[i][:, 1]/img_metas[i]['img_shape'][0]*H
        # #     new_boxxes[:, 3] = gt_bboxes[i][:, 3]/img_metas[i]['img_shape'][0]*H
        # #     # """

        # #     wmin.append(torch.floor(new_boxxes[:, 0]).int())
        # #     wmax.append(torch.ceil(new_boxxes[:, 2]).int())
        # #     hmin.append(torch.floor(new_boxxes[:, 1]).int())
        # #     hmax.append(torch.ceil(new_boxxes[:, 3]).int())

        # #     area = 1.0/(hmax[i].view(1,-1)+1-hmin[i].view(1,-1))/(wmax[i].view(1,-1)+1-wmin[i].view(1,-1))
    

        # #     for j in range(len(gt_bboxes[i])):
        # #         Mask_fg[i][hmin[i][j]:hmax[i][j]+1, wmin[i][j]:wmax[i][j]+1] = \
        # #                 torch.maximum(Mask_fg[i][hmin[i][j]:hmax[i][j]+1, wmin[i][j]:wmax[i][j]+1], area[0][j])
        # #     Mask_bg[i] = torch.where(Mask_fg[i]>0, 0, 1)
        # #     if torch.sum(Mask_bg[i]):
        # #         Mask_bg[i] /= torch.sum(Mask_bg[i])
        
        # for i in range(N):
        #     new_boxxes = torch.ones_like(gt_bboxes[i])
        #     # print(gt_bboxes[i], img_metas[i]['img_shape'][1], W)
        #     # """ fuhao7i
        #     new_boxxes[:, 0] = gt_bboxes[i][:, 0]/img_metas[i]['img_shape'][1]*W
        #     new_boxxes[:, 2] = gt_bboxes[i][:, 2]/img_metas[i]['img_shape'][1]*W
        #     new_boxxes[:, 1] = gt_bboxes[i][:, 1]/img_metas[i]['img_shape'][0]*H
        #     new_boxxes[:, 3] = gt_bboxes[i][:, 3]/img_metas[i]['img_shape'][0]*H
        #     # """
        #     # print('new_bboxes: ', new_boxxes)
        #     wmin.append(torch.floor(new_boxxes[:, 0]).int())
        #     wmax.append(torch.ceil(new_boxxes[:, 2]).int())
        #     hmin.append(torch.floor(new_boxxes[:, 1]).int())
        #     hmax.append(torch.ceil(new_boxxes[:, 3]).int())

        #     # area = 1.0/(hmax[i].view(1,-1)+1-hmin[i].view(1,-1))/(wmax[i].view(1,-1)+1-wmin[i].view(1,-1))
    

        #     for j in range(len(gt_bboxes[i])):
        #         # print(hmin[i][j], hmax[i][j], wmin[i][j], wmax[i][j])
        #         bbox_h = hmax[i][j] - hmin[i][j] 
        #         bbox_w = wmax[i][j] - wmin[i][j] 

        #         mask_ratio = 0.75
        #         num_patches = bbox_h * bbox_w
        #         num_mask = int(mask_ratio * num_patches)

        #         mask = torch.hstack([
        #             torch.ones(num_patches - num_mask),
        #             torch.zeros(num_mask),
        #         ])
                
        #         # random.shuffle(mask)
        #         # mask = torch.randperm(mask)
        #         idx = torch.randperm(num_patches)
        #         mask = mask.view(-1)[idx].view((bbox_h, bbox_w))

        #         # mask = mask.reshape((bbox_h, bbox_w))

        #         Mask_fg[i][hmin[i][j]:hmax[i][j], wmin[i][j]:wmax[i][j]] = mask
        #         # Mask_fg[i][max(hmin[i][j]-20, 0):min(hmax[i][j]+20, H), max(wmin[i][j]-20, 0):min(wmax[i][j]+20, W)] = 1.0
        #         # Mask_fg[i][hmin[i][j]:hmax[i][j]+1, wmin[i][j]:wmax[i][j]+1] = torch.maximum(Mask_fg[i][hmin[i][j]:hmax[i][j]+1, wmin[i][j]:wmax[i][j]+1], area[0][j])
        #         # print(j, '置为1')
        #         # print(Mask_fg[i].shape)
                        

        #     # Mask_bg[i] = torch.where(Mask_fg[i]>0, 0, 1)
                
                
        # # t = Mask_fg.data.cpu().numpy()
        # # # fig = sns.heatmap(data=t[0], vmax=9., vmin=6.5, cmap='viridis') 
        # # # fig = sns.heatmap(data=t[0], vmax=4., vmin=3.4, cmap='viridis')
        # # fig = sns.heatmap(data=t[0], cmap='viridis')  
        # # heatmap = fig.get_figure()
        # # # heatmap.savefig(output_images_path + name[0], dpi=400)
        # # heatmap.savefig('@fuhao/transfer/Mask_fg.png', dpi=400)
        # # heatmap.clear()
        
        # # Mask_fg = Mask_fg.unsqueeze(1)
        # # mask_img = Mask_fg * img
        # # save_image(mask_img, '@fuhao/transfer/mask_img.jpg', normalize=True)
        # # Mask_fg = Mask_fg.squeeze()
        
        # # x = list(x)

        # Mask_fg = Mask_fg.unsqueeze(1)
        
        # mask_input = img * Mask_fg
        # with torch.no_grad():
        #     self.transfernet.eval()
        #     img_t = img.detach()
        #     label_guide_x = self.transfernet(img_t)
        
        #     # bbox_feat = list(bbox_feat)
        #     label_guide_x = list(label_guide_x)

        #     T = self.convtransfer(label_guide_x[0])
        #     T = nn.functional.interpolate(T, size=[img.shape[2], img.shape[3]], mode='bilinear', align_corners=False)

        #     # img = img / (1 - T)
        #     img = img * T + img

        input_img = img.detach()
        
        bbox_h = img.shape[2]
        bbox_w = img.shape[3]
        mask_ratio = 0.5
        num_patches = bbox_h * bbox_w
        num_mask = int(mask_ratio * num_patches)

        mask = torch.hstack([
            torch.ones(num_patches - num_mask),
            torch.zeros(num_mask),
        ])
        
        # random.shuffle(mask)
        # mask = torch.randperm(mask)
        idx = torch.randperm(num_patches)
        mask = mask.view(-1)[idx].view((bbox_h, bbox_w))

        img = img * mask.cuda()

        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)

        x = list(x)

        mae_out = self.mae_decoder(x[0])
        input_img = nn.functional.interpolate(input_img, size=[x[0].shape[2], x[0].shape[3]], mode='bilinear', align_corners=False)
        loss_mae = self.mse_loss(mae_out, input_img)
        x = tuple(x)
        # loss0 = self.mse_loss(bbox_feat[0], label_guide_x[0])
        # loss1 = self.mse_loss(bbox_feat[1], label_guide_x[1])
        # loss2 = self.mse_loss(bbox_feat[2], label_guide_x[2])
        # loss3 = self.mse_loss(bbox_feat[3], label_guide_x[3])
        # loss4 = self.mse_loss(bbox_feat[4], label_guide_x[4])
        
        # bbox_feat = tuple(bbox_feat)
        # label_guide_x = tuple(label_guide_x)
        
        loss = 0.0
        loss = loss_mae
        
        # label_guide_x0 = self.fpn0(x[0])
        # label_guide_x1 = self.fpn1(x[1])
        # label_guide_x2 = self.fpn2(x[2])
        # label_guide_x3 = self.fpn3(x[3])
        # label_guide_x4 = self.fpn4(x[4])
        
        # label_guide_x0 = torch.abs(x[0]).mean(1)
        # label_guide_x1 = torch.abs(x[1]).mean(1)
        # label_guide_x2 = torch.abs(x[2]).mean(1)
        # label_guide_x3 = torch.abs(x[3]).mean(1)
        # label_mask0 =  torch.where(label_guide_x0>0.5, 1.0, 0.0)
        # x[0] = x[0] * label_mask0
        
        # mask0 = nn.functional.interpolate(label_mask0, size=[img.shape[2], img.shape[3]], mode='bilinear', align_corners=False)
        # img = img * mask0
        # save_image(img, '@fuhao/transfer/mask0_img.jpg', normalize=True)
        
        
        # t = label_guide_x0.data.cpu().numpy()
        # # fig = sns.heatmap(data=t[0], vmax=9., vmin=6.5, cmap='viridis') 
        # # fig = sns.heatmap(data=t[0], vmax=4., vmin=3.4, cmap='viridis')
        # fig = sns.heatmap(data=t[0][0], cmap='viridis')  
        # heatmap = fig.get_figure()
        # # heatmap.savefig(output_images_path + name[0], dpi=400)
        # heatmap.savefig('@fuhao/transfer/T.png', dpi=400)
        # heatmap.clear()        

        
        # label_0 = nn.functional.interpolate(Mask_fg, size=[x[0].shape[2], x[0].shape[3]], mode='bilinear', align_corners=False)
        # label_1 = nn.functional.interpolate(Mask_fg, size=[x[1].shape[2], x[1].shape[3]], mode='bilinear', align_corners=False)
        # label_2 = nn.functional.interpolate(Mask_fg, size=[x[2].shape[2], x[2].shape[3]], mode='bilinear', align_corners=False)
        # label_3 = nn.functional.interpolate(Mask_fg, size=[x[3].shape[2], x[3].shape[3]], mode='bilinear', align_corners=False)
        # label_4 = nn.functional.interpolate(Mask_fg, size=[x[4].shape[2], x[4].shape[3]], mode='bilinear', align_corners=False)
        
        
        # loss0 = self.mse_loss(label_guide_x0, label_0)
        # loss1 = self.mse_loss(label_guide_x1, label_1)
        # loss2 = self.mse_loss(label_guide_x2, label_2)
        # loss3 = self.mse_loss(label_guide_x3, label_3)
        # loss4 = self.mse_loss(label_guide_x4, label_4)
        
        # loss = loss0 + loss1 + loss2 + loss3 + loss4
        # loss = loss0

        # t = label_0.data.cpu().numpy()
        # fig = sns.heatmap(data=t[0][0], cmap='viridis')  
        # heatmap = fig.get_figure()
        # heatmap.savefig('@fuhao/transfer/label0.png', dpi=400)
        # heatmap.clear()      
        
        # t = label_mask0.data.cpu().numpy()
        # fig = sns.heatmap(data=t[0][0], cmap='viridis')  
        # heatmap = fig.get_figure()
        # heatmap.savefig('@fuhao/transfer/label_mask0.png', dpi=400)
        # heatmap.clear()       
        
        # label_guide_x = tuple([label_guide_x0, label_guide_x1, label_guide_x2, label_guide_x3, label_guide_x4])
        # label_guide_x = tuple([label_guide_x0, None, None, None, None])


        # mask_img = Mask_fg * img
        # save_image(mask_img, '@fuhao/transfer/mask_img.jpg', normalize=True)
        
        # label_guide_x0 = torch.where(label_guide_x0>0.2, 1.0, 0.)
        # label_guide_x0 = nn.functional.interpolate(label_guide_x0, size=[img.shape[2], img.shape[3]], mode='bilinear', align_corners=False)
        # label_guide_img = label_guide_x0 * img
        # save_image(label_guide_img, '@fuhao/transfer/label_guide_img.jpg', normalize=True)
        
        # x = tuple(x)
        bbox_feat = tuple([None, None, None, None, None])
        # return x, loss, label_guide_x        
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
        
        losses.update({'loss_label_guide':loss_lable_guide})
        
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
        
        outs = self.bbox_head(x, label_guide_x)
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
class Custom_RetinaNet_v544(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(Custom_RetinaNet_v544, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained)
        
        self.mse_loss = torch.nn.MSELoss().cuda()
        
        # self.fpn0 = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.Sigmoid()
        # )

        # self.mae_decoder = nn.Sequential(
        #     nn.Conv2d(256, 3, kernel_size=3, padding=1),
        #     nn.Tanh()
        # )
        
        self.en_net = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Tanh()
        )

        # self.labelnet = ResNet18_FPN(in_channels=1)
        
        # self.masknet = ResNet18_FPN(in_channels=3)

        # self.transfernet = ResNet18_FPN(in_channels=3)
        # self.transfernet.eval()
        # self.convtransfer = nn.Sequential(
        #     nn.Conv2d(256, 3, 3, 1, 1),
        #     nn.Sigmoid()
        # )

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

        # with torch.no_grad():
        #     self.transfernet.eval()
        #     label_x = self.transfernet(img)
        #     label_x = list(label_x)
        #     T = self.convtransfer(label_x[0])
        #     T = nn.functional.interpolate(T, size=[img.shape[2], img.shape[3]], mode='bilinear', align_corners=False)
        #     # img = img / (1 - T)
        #     img = img * T + img
        
        bbox_h = img.shape[2]
        bbox_w = img.shape[3]
        mask_ratio = 0.75
        num_patches = bbox_h * bbox_w
        num_mask = int(mask_ratio * num_patches)

        mask = torch.hstack([
            torch.ones(num_patches - num_mask),
            torch.zeros(num_mask),
        ])
        idx = torch.randperm(num_patches)
        mask = mask.view(-1)[idx].view((bbox_h, bbox_w))
        
        img = img * mask.cuda()
        
        img = self.en_net(img)    
        
        
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        
        # x = list(x)
        # label_guide_x = self.masknet(img)
        # label_guide_x0 = self.fpn0(x[0])
        # label_guide_x1 = self.fpn1(x[1])
        # label_guide_x2 = self.fpn2(x[2])
        # label_guide_x3 = self.fpn3(x[3])
        # label_guide_x4 = self.fpn4(x[4])

        # label_mask0 =  torch.where(label_guide_x0>0.5, 1.0, 0.0)
        # x[0] = x[0] * label_mask0 
        # label_guide_x = tuple([label_guide_x0, label_guide_x1, label_guide_x2, label_guide_x3, label_guide_x4])
        # label_guide_x = tuple([label_guide_x0, None, None, None, None])
        label_guide_x = tuple([None, None, None, None, None])
        
        return x, label_guide_x
    
    
    def extract_feat_label_guide(self, img, img_metas, gt_bboxes):
        """Directly extract features from the backbone+neck."""

        # x = self.backbone(img)
        # if self.with_neck:
        #     x = self.neck(x)
            
        # bbox_feat = self.masknet(img)
            
        # for i in list(x):
        #     print(i.shape)
        """
        N, C, H, W = img.shape
        Mask_fg = torch.ones([N, H, W]).cuda()
        wmin,wmax,hmin,hmax = [],[],[],[]
        for i in range(N):
            new_boxxes = torch.ones_like(gt_bboxes[i])
            new_boxxes[:, 0] = gt_bboxes[i][:, 0]/img_metas[i]['img_shape'][1]*W
            new_boxxes[:, 2] = gt_bboxes[i][:, 2]/img_metas[i]['img_shape'][1]*W
            new_boxxes[:, 1] = gt_bboxes[i][:, 1]/img_metas[i]['img_shape'][0]*H
            new_boxxes[:, 3] = gt_bboxes[i][:, 3]/img_metas[i]['img_shape'][0]*H

            wmin.append(torch.floor(new_boxxes[:, 0]).int())
            wmax.append(torch.ceil(new_boxxes[:, 2]).int())
            hmin.append(torch.floor(new_boxxes[:, 1]).int())
            hmax.append(torch.ceil(new_boxxes[:, 3]).int())

            for j in range(len(gt_bboxes[i])):
                # print(hmin[i][j], hmax[i][j], wmin[i][j], wmax[i][j])
                bbox_h = hmax[i][j] - hmin[i][j] 
                bbox_w = wmax[i][j] - wmin[i][j] 

                mask_ratio = 0.75
                num_patches = bbox_h * bbox_w
                num_mask = int(mask_ratio * num_patches)

                mask = torch.hstack([
                    torch.ones(num_patches - num_mask),
                    torch.zeros(num_mask),
                ])
                
                idx = torch.randperm(num_patches)
                mask = mask.view(-1)[idx].view((bbox_h, bbox_w))

                # mask = mask.reshape((bbox_h, bbox_w))

                Mask_fg[i][hmin[i][j]:hmax[i][j], wmin[i][j]:wmax[i][j]] = mask
                
        Mask_fg = Mask_fg.unsqueeze(1)
        """
        # mask_input = img * Mask_fg
        # with torch.no_grad():
        #     self.transfernet.eval()
        #     img_t = img.detach()
        #     label_guide_x = self.transfernet(img_t)
        
        #     # bbox_feat = list(bbox_feat)
        #     label_guide_x = list(label_guide_x)

        #     T = self.convtransfer(label_guide_x[0])
        #     T = nn.functional.interpolate(T, size=[img.shape[2], img.shape[3]], mode='bilinear', align_corners=False)

        #     # img = img / (1 - T)
        #     img = img * T + img

        input_img = img.detach()
        
        bbox_h = img.shape[2]
        bbox_w = img.shape[3]
        mask_ratio = 0.75
        num_patches = bbox_h * bbox_w
        num_mask = int(mask_ratio * num_patches)

        mask = torch.hstack([
            torch.ones(num_patches - num_mask),
            torch.zeros(num_mask),
        ])
        
        # random.shuffle(mask)
        # mask = torch.randperm(mask)
        idx = torch.randperm(num_patches)
        mask = mask.view(-1)[idx].view((bbox_h, bbox_w))

        img = img * mask.cuda()
        
        img = self.en_net(img)

        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)

        # x = list(x)

        # mae_out = self.mae_decoder(x[0])
        # input_img = nn.functional.interpolate(img, size=[x[0].shape[2], x[0].shape[3]], mode='bilinear', align_corners=False)
        loss_mae = self.mse_loss(img, input_img)

        # loss0 = self.mse_loss(bbox_feat[0], label_guide_x[0])
        # loss1 = self.mse_loss(bbox_feat[1], label_guide_x[1])
        # loss2 = self.mse_loss(bbox_feat[2], label_guide_x[2])
        # loss3 = self.mse_loss(bbox_feat[3], label_guide_x[3])
        # loss4 = self.mse_loss(bbox_feat[4], label_guide_x[4])
        
        # bbox_feat = tuple(bbox_feat)
        # label_guide_x = tuple(label_guide_x)
        
        loss = 0.0
        loss = loss_mae
        
        # label_guide_x0 = self.fpn0(x[0])
        # label_guide_x1 = self.fpn1(x[1])
        # label_guide_x2 = self.fpn2(x[2])
        # label_guide_x3 = self.fpn3(x[3])
        # label_guide_x4 = self.fpn4(x[4])
        
        # label_guide_x0 = torch.abs(x[0]).mean(1)
        # label_guide_x1 = torch.abs(x[1]).mean(1)
        # label_guide_x2 = torch.abs(x[2]).mean(1)
        # label_guide_x3 = torch.abs(x[3]).mean(1)
        # label_mask0 =  torch.where(label_guide_x0>0.5, 1.0, 0.0)
        # x[0] = x[0] * label_mask0
        
        # mask0 = nn.functional.interpolate(label_mask0, size=[img.shape[2], img.shape[3]], mode='bilinear', align_corners=False)
        # img = img * mask0
        # save_image(img, '@fuhao/transfer/mask0_img.jpg', normalize=True)
        
        
        # t = label_guide_x0.data.cpu().numpy()
        # # fig = sns.heatmap(data=t[0], vmax=9., vmin=6.5, cmap='viridis') 
        # # fig = sns.heatmap(data=t[0], vmax=4., vmin=3.4, cmap='viridis')
        # fig = sns.heatmap(data=t[0][0], cmap='viridis')  
        # heatmap = fig.get_figure()
        # # heatmap.savefig(output_images_path + name[0], dpi=400)
        # heatmap.savefig('@fuhao/transfer/T.png', dpi=400)
        # heatmap.clear()        

        
        # label_0 = nn.functional.interpolate(Mask_fg, size=[x[0].shape[2], x[0].shape[3]], mode='bilinear', align_corners=False)
        # label_1 = nn.functional.interpolate(Mask_fg, size=[x[1].shape[2], x[1].shape[3]], mode='bilinear', align_corners=False)
        # label_2 = nn.functional.interpolate(Mask_fg, size=[x[2].shape[2], x[2].shape[3]], mode='bilinear', align_corners=False)
        # label_3 = nn.functional.interpolate(Mask_fg, size=[x[3].shape[2], x[3].shape[3]], mode='bilinear', align_corners=False)
        # label_4 = nn.functional.interpolate(Mask_fg, size=[x[4].shape[2], x[4].shape[3]], mode='bilinear', align_corners=False)
        
        
        # loss0 = self.mse_loss(label_guide_x0, label_0)
        # loss1 = self.mse_loss(label_guide_x1, label_1)
        # loss2 = self.mse_loss(label_guide_x2, label_2)
        # loss3 = self.mse_loss(label_guide_x3, label_3)
        # loss4 = self.mse_loss(label_guide_x4, label_4)
        
        # loss = loss0 + loss1 + loss2 + loss3 + loss4
        # loss = loss0

        # t = label_0.data.cpu().numpy()
        # fig = sns.heatmap(data=t[0][0], cmap='viridis')  
        # heatmap = fig.get_figure()
        # heatmap.savefig('@fuhao/transfer/label0.png', dpi=400)
        # heatmap.clear()      
        
        # t = label_mask0.data.cpu().numpy()
        # fig = sns.heatmap(data=t[0][0], cmap='viridis')  
        # heatmap = fig.get_figure()
        # heatmap.savefig('@fuhao/transfer/label_mask0.png', dpi=400)
        # heatmap.clear()       
        
        # label_guide_x = tuple([label_guide_x0, label_guide_x1, label_guide_x2, label_guide_x3, label_guide_x4])
        # label_guide_x = tuple([label_guide_x0, None, None, None, None])


        # mask_img = Mask_fg * img
        # save_image(mask_img, '@fuhao/transfer/mask_img.jpg', normalize=True)
        
        # label_guide_x0 = torch.where(label_guide_x0>0.2, 1.0, 0.)
        # label_guide_x0 = nn.functional.interpolate(label_guide_x0, size=[img.shape[2], img.shape[3]], mode='bilinear', align_corners=False)
        # label_guide_img = label_guide_x0 * img
        # save_image(label_guide_img, '@fuhao/transfer/label_guide_img.jpg', normalize=True)
        
        # x = tuple(x)
        bbox_feat = tuple([None, None, None, None, None])
        # return x, loss, label_guide_x        
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
        
        losses.update({'loss_en_net':loss_lable_guide})
        
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
        
        outs = self.bbox_head(x, label_guide_x)
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
class Custom_RetinaNet_v545(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(Custom_RetinaNet_v545, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained)
        
        self.mse_loss = torch.nn.MSELoss().cuda()
        
        # self.fpn0 = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.Sigmoid()
        # )

        self.mae_decoder = nn.Sequential(
            nn.Conv2d(256, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
        # self.en_net = nn.Sequential(
        #     nn.Conv2d(3, 16, 3, 1, 1),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 32, 3, 1, 1),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, 3, 1, 1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 32, 3, 1, 1),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 3, 3, 1, 1),
        #     nn.Tanh()
        # )

        # self.labelnet = ResNet18_FPN(in_channels=1)
        
        # self.masknet = ResNet18_FPN(in_channels=3)

        # self.transfernet = ResNet18_FPN(in_channels=3)
        # self.transfernet.eval()
        # self.convtransfer = nn.Sequential(
        #     nn.Conv2d(256, 3, 3, 1, 1),
        #     nn.Sigmoid()
        # )

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

        # with torch.no_grad():
        #     self.transfernet.eval()
        #     label_x = self.transfernet(img)
        #     label_x = list(label_x)
        #     T = self.convtransfer(label_x[0])
        #     T = nn.functional.interpolate(T, size=[img.shape[2], img.shape[3]], mode='bilinear', align_corners=False)
        #     # img = img / (1 - T)
        #     img = img * T + img
        
        # bbox_h = img.shape[2]
        # bbox_w = img.shape[3]
        # mask_ratio = 0.75
        # num_patches = bbox_h * bbox_w
        # num_mask = int(mask_ratio * num_patches)

        # mask = torch.hstack([
        #     torch.ones(num_patches - num_mask),
        #     torch.zeros(num_mask),
        # ])
        # idx = torch.randperm(num_patches)
        # mask = mask.view(-1)[idx].view((bbox_h, bbox_w))
        
        # img = img * mask.cuda()
        
        # img = self.en_net(img)    
        
        
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        
        # x = list(x)
        # label_guide_x = self.masknet(img)
        # label_guide_x0 = self.fpn0(x[0])
        # label_guide_x1 = self.fpn1(x[1])
        # label_guide_x2 = self.fpn2(x[2])
        # label_guide_x3 = self.fpn3(x[3])
        # label_guide_x4 = self.fpn4(x[4])

        # label_mask0 =  torch.where(label_guide_x0>0.5, 1.0, 0.0)
        # x[0] = x[0] * label_mask0 
        # label_guide_x = tuple([label_guide_x0, label_guide_x1, label_guide_x2, label_guide_x3, label_guide_x4])
        # label_guide_x = tuple([label_guide_x0, None, None, None, None])
        label_guide_x = tuple([None, None, None, None, None])
        
        return x, label_guide_x
    
    
    def extract_feat_label_guide(self, img, img_metas, gt_bboxes):
        """Directly extract features from the backbone+neck."""

        # x = self.backbone(img)
        # if self.with_neck:
        #     x = self.neck(x)
            
        # bbox_feat = self.masknet(img)
            
        # for i in list(x):
        #     print(i.shape)
        # """
        N, C, H, W = img.shape
        Mask_fg = torch.zeros([N, H, W]).cuda()
        wmin,wmax,hmin,hmax = [],[],[],[]
        for i in range(N):
            new_boxxes = torch.ones_like(gt_bboxes[i])
            new_boxxes[:, 0] = gt_bboxes[i][:, 0]/img_metas[i]['img_shape'][1]*W
            new_boxxes[:, 2] = gt_bboxes[i][:, 2]/img_metas[i]['img_shape'][1]*W
            new_boxxes[:, 1] = gt_bboxes[i][:, 1]/img_metas[i]['img_shape'][0]*H
            new_boxxes[:, 3] = gt_bboxes[i][:, 3]/img_metas[i]['img_shape'][0]*H

            wmin.append(torch.floor(new_boxxes[:, 0]).int())
            wmax.append(torch.ceil(new_boxxes[:, 2]).int())
            hmin.append(torch.floor(new_boxxes[:, 1]).int())
            hmax.append(torch.ceil(new_boxxes[:, 3]).int())

            for j in range(len(gt_bboxes[i])):
                # print(hmin[i][j], hmax[i][j], wmin[i][j], wmax[i][j])
                # bbox_h = hmax[i][j] - hmin[i][j] 
                # bbox_w = wmax[i][j] - wmin[i][j] 

                # mask_ratio = 0.75
                # num_patches = bbox_h * bbox_w
                # num_mask = int(mask_ratio * num_patches)

                # mask = torch.hstack([
                #     torch.ones(num_patches - num_mask),
                #     torch.zeros(num_mask),
                # ])
                
                # idx = torch.randperm(num_patches)
                # mask = mask.view(-1)[idx].view((bbox_h, bbox_w))

                Mask_fg[i][hmin[i][j]:hmax[i][j], wmin[i][j]:wmax[i][j]] = 1.0
                
        Mask_fg = Mask_fg.unsqueeze(1)
        # """
        # mask_input = img * Mask_fg
        # with torch.no_grad():
        #     self.transfernet.eval()
        #     img_t = img.detach()
        #     label_guide_x = self.transfernet(img_t)
        
        #     # bbox_feat = list(bbox_feat)
        #     label_guide_x = list(label_guide_x)

        #     T = self.convtransfer(label_guide_x[0])
        #     T = nn.functional.interpolate(T, size=[img.shape[2], img.shape[3]], mode='bilinear', align_corners=False)

        #     # img = img / (1 - T)
        #     img = img * T + img

        input_img = img.detach()
        input_img = input_img * Mask_fg
        # bbox_h = img.shape[2]
        # bbox_w = img.shape[3]
        # mask_ratio = 0.75
        # num_patches = bbox_h * bbox_w
        # num_mask = int(mask_ratio * num_patches)

        # mask = torch.hstack([
        #     torch.ones(num_patches - num_mask),
        #     torch.zeros(num_mask),
        # ])
        
        # # random.shuffle(mask)
        # # mask = torch.randperm(mask)
        # idx = torch.randperm(num_patches)
        # mask = mask.view(-1)[idx].view((bbox_h, bbox_w))

        # img = img * mask.cuda()
        
        # img = self.en_net(img)

        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)

        x = list(x)

        mae_out = self.mae_decoder(x[0])
        input_img = nn.functional.interpolate(input_img, size=[x[0].shape[2], x[0].shape[3]], mode='bilinear', align_corners=False)
        loss_mae = self.mse_loss(mae_out, input_img)
        x = tuple(x)
        # loss0 = self.mse_loss(bbox_feat[0], label_guide_x[0])
        # loss1 = self.mse_loss(bbox_feat[1], label_guide_x[1])
        # loss2 = self.mse_loss(bbox_feat[2], label_guide_x[2])
        # loss3 = self.mse_loss(bbox_feat[3], label_guide_x[3])
        # loss4 = self.mse_loss(bbox_feat[4], label_guide_x[4])
        
        # bbox_feat = tuple(bbox_feat)
        # label_guide_x = tuple(label_guide_x)
        
        loss = 0.0
        loss = loss_mae
        
        # label_guide_x0 = self.fpn0(x[0])
        # label_guide_x1 = self.fpn1(x[1])
        # label_guide_x2 = self.fpn2(x[2])
        # label_guide_x3 = self.fpn3(x[3])
        # label_guide_x4 = self.fpn4(x[4])
        
        # label_guide_x0 = torch.abs(x[0]).mean(1)
        # label_guide_x1 = torch.abs(x[1]).mean(1)
        # label_guide_x2 = torch.abs(x[2]).mean(1)
        # label_guide_x3 = torch.abs(x[3]).mean(1)
        # label_mask0 =  torch.where(label_guide_x0>0.5, 1.0, 0.0)
        # x[0] = x[0] * label_mask0
        
        # mask0 = nn.functional.interpolate(label_mask0, size=[img.shape[2], img.shape[3]], mode='bilinear', align_corners=False)
        # img = img * mask0
        # save_image(img, '@fuhao/transfer/mask0_img.jpg', normalize=True)
        
        
        # t = label_guide_x0.data.cpu().numpy()
        # # fig = sns.heatmap(data=t[0], vmax=9., vmin=6.5, cmap='viridis') 
        # # fig = sns.heatmap(data=t[0], vmax=4., vmin=3.4, cmap='viridis')
        # fig = sns.heatmap(data=t[0][0], cmap='viridis')  
        # heatmap = fig.get_figure()
        # # heatmap.savefig(output_images_path + name[0], dpi=400)
        # heatmap.savefig('@fuhao/transfer/T.png', dpi=400)
        # heatmap.clear()        

        
        # label_0 = nn.functional.interpolate(Mask_fg, size=[x[0].shape[2], x[0].shape[3]], mode='bilinear', align_corners=False)
        # label_1 = nn.functional.interpolate(Mask_fg, size=[x[1].shape[2], x[1].shape[3]], mode='bilinear', align_corners=False)
        # label_2 = nn.functional.interpolate(Mask_fg, size=[x[2].shape[2], x[2].shape[3]], mode='bilinear', align_corners=False)
        # label_3 = nn.functional.interpolate(Mask_fg, size=[x[3].shape[2], x[3].shape[3]], mode='bilinear', align_corners=False)
        # label_4 = nn.functional.interpolate(Mask_fg, size=[x[4].shape[2], x[4].shape[3]], mode='bilinear', align_corners=False)
        
        
        # loss0 = self.mse_loss(label_guide_x0, label_0)
        # loss1 = self.mse_loss(label_guide_x1, label_1)
        # loss2 = self.mse_loss(label_guide_x2, label_2)
        # loss3 = self.mse_loss(label_guide_x3, label_3)
        # loss4 = self.mse_loss(label_guide_x4, label_4)
        
        # loss = loss0 + loss1 + loss2 + loss3 + loss4
        # loss = loss0

        # t = label_0.data.cpu().numpy()
        # fig = sns.heatmap(data=t[0][0], cmap='viridis')  
        # heatmap = fig.get_figure()
        # heatmap.savefig('@fuhao/transfer/label0.png', dpi=400)
        # heatmap.clear()      
        
        # t = label_mask0.data.cpu().numpy()
        # fig = sns.heatmap(data=t[0][0], cmap='viridis')  
        # heatmap = fig.get_figure()
        # heatmap.savefig('@fuhao/transfer/label_mask0.png', dpi=400)
        # heatmap.clear()       
        
        # label_guide_x = tuple([label_guide_x0, label_guide_x1, label_guide_x2, label_guide_x3, label_guide_x4])
        # label_guide_x = tuple([label_guide_x0, None, None, None, None])


        # mask_img = Mask_fg * img
        # save_image(mask_img, '@fuhao/transfer/mask_img.jpg', normalize=True)
        
        # label_guide_x0 = torch.where(label_guide_x0>0.2, 1.0, 0.)
        # label_guide_x0 = nn.functional.interpolate(label_guide_x0, size=[img.shape[2], img.shape[3]], mode='bilinear', align_corners=False)
        # label_guide_img = label_guide_x0 * img
        # save_image(label_guide_img, '@fuhao/transfer/label_guide_img.jpg', normalize=True)
        
        # x = tuple(x)
        bbox_feat = tuple([None, None, None, None, None])
        # return x, loss, label_guide_x        
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
        
        losses.update({'loss_reconstruct_fg':loss_lable_guide})
        
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
        
        outs = self.bbox_head(x, label_guide_x)
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
class Custom_RetinaNet_v546(Custom_RetinaNet_v545):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(Custom_RetinaNet_v546, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained)
        
        self.mse_loss = torch.nn.MSELoss().cuda()
        
        # self.fpn0 = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.Sigmoid()
        # )

        self.mae_decoder = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
        self.fg_block = nn.Conv2d(64, 64, 3, 1, 1)
        self.fg_head = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
        self.bg_block = nn.Conv2d(64, 64, 3, 1, 1)
        self.bg_head = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
        # self.en_net = nn.Sequential(
        #     nn.Conv2d(3, 16, 3, 1, 1),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 32, 3, 1, 1),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, 3, 1, 1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 32, 3, 1, 1),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 3, 3, 1, 1),
        #     nn.Tanh()
        # )

        # self.labelnet = ResNet18_FPN(in_channels=1)
        
        # self.masknet = ResNet18_FPN(in_channels=3)

        # self.transfernet = ResNet18_FPN(in_channels=3)
        # self.transfernet.eval()
        # self.convtransfer = nn.Sequential(
        #     nn.Conv2d(256, 3, 3, 1, 1),
        #     nn.Sigmoid()
        # )

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
        
    
    
    def extract_feat_label_guide(self, img, img_metas, gt_bboxes):
        """Directly extract features from the backbone+neck."""

        # x = self.backbone(img)
        # if self.with_neck:
        #     x = self.neck(x)
            
        # bbox_feat = self.masknet(img)
            
        # for i in list(x):
        #     print(i.shape)
        # """
        N, C, H, W = img.shape
        Mask_fg = torch.zeros([N, H, W]).cuda()
        wmin,wmax,hmin,hmax = [],[],[],[]
        for i in range(N):
            new_boxxes = torch.ones_like(gt_bboxes[i])
            new_boxxes[:, 0] = gt_bboxes[i][:, 0]/img_metas[i]['img_shape'][1]*W
            new_boxxes[:, 2] = gt_bboxes[i][:, 2]/img_metas[i]['img_shape'][1]*W
            new_boxxes[:, 1] = gt_bboxes[i][:, 1]/img_metas[i]['img_shape'][0]*H
            new_boxxes[:, 3] = gt_bboxes[i][:, 3]/img_metas[i]['img_shape'][0]*H

            wmin.append(torch.floor(new_boxxes[:, 0]).int())
            wmax.append(torch.ceil(new_boxxes[:, 2]).int())
            hmin.append(torch.floor(new_boxxes[:, 1]).int())
            hmax.append(torch.ceil(new_boxxes[:, 3]).int())

            for j in range(len(gt_bboxes[i])):
                # print(hmin[i][j], hmax[i][j], wmin[i][j], wmax[i][j])
                # bbox_h = hmax[i][j] - hmin[i][j] 
                # bbox_w = wmax[i][j] - wmin[i][j] 

                # mask_ratio = 0.75
                # num_patches = bbox_h * bbox_w
                # num_mask = int(mask_ratio * num_patches)

                # mask = torch.hstack([
                #     torch.ones(num_patches - num_mask),
                #     torch.zeros(num_mask),
                # ])
                
                # idx = torch.randperm(num_patches)
                # mask = mask.view(-1)[idx].view((bbox_h, bbox_w))

                Mask_fg[i][hmin[i][j]:hmax[i][j], wmin[i][j]:wmax[i][j]] = 1.0
                
        Mask_fg = Mask_fg.unsqueeze(1)
        # """
        # mask_input = img * Mask_fg
        # with torch.no_grad():
        #     self.transfernet.eval()
        #     img_t = img.detach()
        #     label_guide_x = self.transfernet(img_t)
        
        #     # bbox_feat = list(bbox_feat)
        #     label_guide_x = list(label_guide_x)

        #     T = self.convtransfer(label_guide_x[0])
        #     T = nn.functional.interpolate(T, size=[img.shape[2], img.shape[3]], mode='bilinear', align_corners=False)

        #     # img = img / (1 - T)
        #     img = img * T + img

        input_img = img.detach()
        input_img = input_img * Mask_fg
        # bbox_h = img.shape[2]
        # bbox_w = img.shape[3]
        # mask_ratio = 0.75
        # num_patches = bbox_h * bbox_w
        # num_mask = int(mask_ratio * num_patches)

        # mask = torch.hstack([
        #     torch.ones(num_patches - num_mask),
        #     torch.zeros(num_mask),
        # ])
        
        # # random.shuffle(mask)
        # # mask = torch.randperm(mask)
        # idx = torch.randperm(num_patches)
        # mask = mask.view(-1)[idx].view((bbox_h, bbox_w))

        # img = img * mask.cuda()
        
        # img = self.en_net(img)

        backbone_x = self.backbone(img)
        if self.with_neck:
            x = self.neck(backbone_x)

        backbone_x = list(backbone_x)

        mae_out = self.mae_decoder(backbone_x[0])
        input_img = nn.functional.interpolate(input_img, size=[backbone_x[0].shape[2], backbone_x[0].shape[3]], mode='bilinear', align_corners=False)
        loss_mae = self.mse_loss(mae_out, input_img)
        backbone_x = tuple(backbone_x)
        # loss0 = self.mse_loss(bbox_feat[0], label_guide_x[0])
        # loss1 = self.mse_loss(bbox_feat[1], label_guide_x[1])
        # loss2 = self.mse_loss(bbox_feat[2], label_guide_x[2])
        # loss3 = self.mse_loss(bbox_feat[3], label_guide_x[3])
        # loss4 = self.mse_loss(bbox_feat[4], label_guide_x[4])
        
        # bbox_feat = tuple(bbox_feat)
        # label_guide_x = tuple(label_guide_x)
        
        loss = 0.0
        loss = loss_mae
        
        # label_guide_x0 = self.fpn0(x[0])
        # label_guide_x1 = self.fpn1(x[1])
        # label_guide_x2 = self.fpn2(x[2])
        # label_guide_x3 = self.fpn3(x[3])
        # label_guide_x4 = self.fpn4(x[4])
        
        # label_guide_x0 = torch.abs(x[0]).mean(1)
        # label_guide_x1 = torch.abs(x[1]).mean(1)
        # label_guide_x2 = torch.abs(x[2]).mean(1)
        # label_guide_x3 = torch.abs(x[3]).mean(1)
        # label_mask0 =  torch.where(label_guide_x0>0.5, 1.0, 0.0)
        # x[0] = x[0] * label_mask0
        
        # mask0 = nn.functional.interpolate(label_mask0, size=[img.shape[2], img.shape[3]], mode='bilinear', align_corners=False)
        # img = img * mask0
        # save_image(img, '@fuhao/transfer/mask0_img.jpg', normalize=True)
        
        
        # t = label_guide_x0.data.cpu().numpy()
        # # fig = sns.heatmap(data=t[0], vmax=9., vmin=6.5, cmap='viridis') 
        # # fig = sns.heatmap(data=t[0], vmax=4., vmin=3.4, cmap='viridis')
        # fig = sns.heatmap(data=t[0][0], cmap='viridis')  
        # heatmap = fig.get_figure()
        # # heatmap.savefig(output_images_path + name[0], dpi=400)
        # heatmap.savefig('@fuhao/transfer/T.png', dpi=400)
        # heatmap.clear()        

        
        # label_0 = nn.functional.interpolate(Mask_fg, size=[x[0].shape[2], x[0].shape[3]], mode='bilinear', align_corners=False)
        # label_1 = nn.functional.interpolate(Mask_fg, size=[x[1].shape[2], x[1].shape[3]], mode='bilinear', align_corners=False)
        # label_2 = nn.functional.interpolate(Mask_fg, size=[x[2].shape[2], x[2].shape[3]], mode='bilinear', align_corners=False)
        # label_3 = nn.functional.interpolate(Mask_fg, size=[x[3].shape[2], x[3].shape[3]], mode='bilinear', align_corners=False)
        # label_4 = nn.functional.interpolate(Mask_fg, size=[x[4].shape[2], x[4].shape[3]], mode='bilinear', align_corners=False)
        
        
        # loss0 = self.mse_loss(label_guide_x0, label_0)
        # loss1 = self.mse_loss(label_guide_x1, label_1)
        # loss2 = self.mse_loss(label_guide_x2, label_2)
        # loss3 = self.mse_loss(label_guide_x3, label_3)
        # loss4 = self.mse_loss(label_guide_x4, label_4)
        
        # loss = loss0 + loss1 + loss2 + loss3 + loss4
        # loss = loss0

        # t = label_0.data.cpu().numpy()
        # fig = sns.heatmap(data=t[0][0], cmap='viridis')  
        # heatmap = fig.get_figure()
        # heatmap.savefig('@fuhao/transfer/label0.png', dpi=400)
        # heatmap.clear()      
        
        # t = label_mask0.data.cpu().numpy()
        # fig = sns.heatmap(data=t[0][0], cmap='viridis')  
        # heatmap = fig.get_figure()
        # heatmap.savefig('@fuhao/transfer/label_mask0.png', dpi=400)
        # heatmap.clear()       
        
        # label_guide_x = tuple([label_guide_x0, label_guide_x1, label_guide_x2, label_guide_x3, label_guide_x4])
        # label_guide_x = tuple([label_guide_x0, None, None, None, None])


        # mask_img = Mask_fg * img
        # save_image(mask_img, '@fuhao/transfer/mask_img.jpg', normalize=True)
        
        # label_guide_x0 = torch.where(label_guide_x0>0.2, 1.0, 0.)
        # label_guide_x0 = nn.functional.interpolate(label_guide_x0, size=[img.shape[2], img.shape[3]], mode='bilinear', align_corners=False)
        # label_guide_img = label_guide_x0 * img
        # save_image(label_guide_img, '@fuhao/transfer/label_guide_img.jpg', normalize=True)
        
        # x = tuple(x)
        bbox_feat = tuple([None, None, None, None, None])
        # return x, loss, label_guide_x        
        return x, loss, bbox_feat




@DETECTORS.register_module()
class Custom_RetinaNet_v547(Custom_RetinaNet_v545):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(Custom_RetinaNet_v547, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained)
        
        self.mse_loss = torch.nn.MSELoss().cuda()
        
        self.fpn0 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=2, padding=1),
            # nn.Sigmoid()
        )

        self.mae_decoder = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
        self.fg_block = nn.Conv2d(64, 64, 3, 1, 1)
        self.fg_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            # nn.Tanh()
            nn.Sigmoid()
        )
        
        self.bg_block = nn.Conv2d(64, 64, 3, 1, 1)
        self.bg_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            # nn.Tanh()
            nn.Sigmoid()
        )
        # self.en_net = nn.Sequential(
        #     nn.Conv2d(3, 16, 3, 1, 1),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 32, 3, 1, 1),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, 3, 1, 1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 32, 3, 1, 1),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 3, 3, 1, 1),
        #     nn.Tanh()
        # )

        # self.labelnet = ResNet18_FPN(in_channels=1)
        
        # self.masknet = ResNet18_FPN(in_channels=3)

        # self.transfernet = ResNet18_FPN(in_channels=3)
        # self.transfernet.eval()
        # self.convtransfer = nn.Sequential(
        #     nn.Conv2d(256, 3, 3, 1, 1),
        #     nn.Sigmoid()
        # )

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
        
    
    
    def extract_feat_label_guide(self, img, img_metas, gt_bboxes):
        """Directly extract features from the backbone+neck."""

        # x = self.backbone(img)
        # if self.with_neck:
        #     x = self.neck(x)
            
        # bbox_feat = self.masknet(img)
            
        # for i in list(x):
        #     print(i.shape)
        # """
        N, C, H, W = img.shape
        Mask_fg = torch.zeros([N, H, W]).cuda()
        Mask_bg = torch.ones([N, H, W]).cuda()
        wmin,wmax,hmin,hmax = [],[],[],[]
        for i in range(N):
            new_boxxes = torch.ones_like(gt_bboxes[i])
            new_boxxes[:, 0] = gt_bboxes[i][:, 0]/img_metas[i]['img_shape'][1]*W
            new_boxxes[:, 2] = gt_bboxes[i][:, 2]/img_metas[i]['img_shape'][1]*W
            new_boxxes[:, 1] = gt_bboxes[i][:, 1]/img_metas[i]['img_shape'][0]*H
            new_boxxes[:, 3] = gt_bboxes[i][:, 3]/img_metas[i]['img_shape'][0]*H

            wmin.append(torch.floor(new_boxxes[:, 0]).int())
            wmax.append(torch.ceil(new_boxxes[:, 2]).int())
            hmin.append(torch.floor(new_boxxes[:, 1]).int())
            hmax.append(torch.ceil(new_boxxes[:, 3]).int())

            area = 1.0/(hmax[i].view(1,-1)+1-hmin[i].view(1,-1))/(wmax[i].view(1,-1)+1-wmin[i].view(1,-1))
            
            for j in range(len(gt_bboxes[i])):
                # print(hmin[i][j], hmax[i][j], wmin[i][j], wmax[i][j])
                # bbox_h = hmax[i][j] - hmin[i][j] 
                # bbox_w = wmax[i][j] - wmin[i][j] 

                # mask_ratio = 0.75
                # num_patches = bbox_h * bbox_w
                # num_mask = int(mask_ratio * num_patches)

                # mask = torch.hstack([
                #     torch.ones(num_patches - num_mask),
                #     torch.zeros(num_mask),
                # ])
                
                # idx = torch.randperm(num_patches)
                # mask = mask.view(-1)[idx].view((bbox_h, bbox_w))
                Mask_fg[i][hmin[i][j]:hmax[i][j]+1, wmin[i][j]:wmax[i][j]+1] = \
                        torch.maximum(Mask_fg[i][hmin[i][j]:hmax[i][j]+1, wmin[i][j]:wmax[i][j]+1], area[0][j])
                        
            Mask_bg[i] = torch.where(Mask_fg[i]>0, 0, 1)
            if torch.sum(Mask_bg[i]):
                Mask_bg[i] /= torch.sum(Mask_bg[i])
                
            #     Mask_fg[i][hmin[i][j]:hmax[i][j], wmin[i][j]:wmax[i][j]] = 1.0
                
            # Mask_bg[i] = torch.where(Mask_fg[i]>0, 0, 1)
                
        Mask_fg = Mask_fg.unsqueeze(1)
        Mask_bg = Mask_bg.unsqueeze(1)
        # """
        # mask_input = img * Mask_fg
        # with torch.no_grad():
        #     self.transfernet.eval()
        #     img_t = img.detach()
        #     label_guide_x = self.transfernet(img_t)
        
        #     # bbox_feat = list(bbox_feat)
        #     label_guide_x = list(label_guide_x)

        #     T = self.convtransfer(label_guide_x[0])
        #     T = nn.functional.interpolate(T, size=[img.shape[2], img.shape[3]], mode='bilinear', align_corners=False)

        #     # img = img / (1 - T)
        #     img = img * T + img

        input_img = img.detach()
        img_fg = input_img * Mask_fg
        img_bg = input_img * Mask_bg
        # save_image(img_fg, './@fuhao/transfer/img_fg.jpg', normalize=True)
        # save_image(img_bg, './@fuhao/transfer/img_bg.jpg', normalize=True)
        # bbox_h = img.shape[2]
        # bbox_w = img.shape[3]
        # mask_ratio = 0.75
        # num_patches = bbox_h * bbox_w
        # num_mask = int(mask_ratio * num_patches)

        # mask = torch.hstack([
        #     torch.ones(num_patches - num_mask),
        #     torch.zeros(num_mask),
        # ])
        
        # # random.shuffle(mask)
        # # mask = torch.randperm(mask)
        # idx = torch.randperm(num_patches)
        # mask = mask.view(-1)[idx].view((bbox_h, bbox_w))

        # img = img * mask.cuda()
        
        # img = self.en_net(img)

        backbone_x = self.backbone(img)
        if self.with_neck:
            x = self.neck(backbone_x)

        backbone_x = list(backbone_x)
        x = list(x)

        neck_feat0 = x[0].detach()
        # neck_feat0 = neck_feat0.pow(2).mean(1)
        
        # backbone_x0 = backbone_x[0]
        backbone_x0 = self.fpn0(backbone_x[0])
        loss0 = self.mse_loss(backbone_x0, neck_feat0)
        # mae_out = self.mae_decoder(backbone_x[0])
        # feat_fg = self.fg_block(backbone_x[0])
        # feat_bg = self.bg_block(backbone_x[0])
        
        # gen_fg = self.fg_head(feat_fg)
        # gen_bg = self.bg_head(feat_bg)
        # save_image(gen_fg, './@fuhao/transfer/gen_fg.jpg', normalize=True)
        # save_image(gen_bg, './@fuhao/transfer/gen_bg.jpg', normalize=True)
        
        # img_fg = nn.functional.interpolate(Mask_fg, size=[backbone_x[0].shape[2], backbone_x[0].shape[3]], mode='bilinear', align_corners=False)
        # img_bg = nn.functional.interpolate(Mask_bg, size=[backbone_x[0].shape[2], backbone_x[0].shape[3]], mode='bilinear', align_corners=False)
        
        # loss_fg = self.mse_loss(gen_fg, img_fg)
        # loss_bg = self.mse_loss(gen_bg, img_bg)
        
        backbone_x = tuple(backbone_x)
        x = tuple(x)
        # loss0 = self.mse_loss(bbox_feat[0], label_guide_x[0])
        # loss1 = self.mse_loss(bbox_feat[1], label_guide_x[1])
        # loss2 = self.mse_loss(bbox_feat[2], label_guide_x[2])
        # loss3 = self.mse_loss(bbox_feat[3], label_guide_x[3])
        # loss4 = self.mse_loss(bbox_feat[4], label_guide_x[4])
        
        # bbox_feat = tuple(bbox_feat)
        # label_guide_x = tuple(label_guide_x)
        
        loss = 0.0
        # loss = loss_fg + loss_bg
        loss = loss0
        
        # label_guide_x0 = self.fpn0(x[0])
        # label_guide_x1 = self.fpn1(x[1])
        # label_guide_x2 = self.fpn2(x[2])
        # label_guide_x3 = self.fpn3(x[3])
        # label_guide_x4 = self.fpn4(x[4])
        
        # label_guide_x0 = torch.abs(x[0]).mean(1)
        # label_guide_x1 = torch.abs(x[1]).mean(1)
        # label_guide_x2 = torch.abs(x[2]).mean(1)
        # label_guide_x3 = torch.abs(x[3]).mean(1)
        # label_mask0 =  torch.where(label_guide_x0>0.5, 1.0, 0.0)
        # x[0] = x[0] * label_mask0
        
        # mask0 = nn.functional.interpolate(label_mask0, size=[img.shape[2], img.shape[3]], mode='bilinear', align_corners=False)
        # img = img * mask0
        # save_image(img, '@fuhao/transfer/mask0_img.jpg', normalize=True)
        
        
        # t = label_guide_x0.data.cpu().numpy()
        # # fig = sns.heatmap(data=t[0], vmax=9., vmin=6.5, cmap='viridis') 
        # # fig = sns.heatmap(data=t[0], vmax=4., vmin=3.4, cmap='viridis')
        # fig = sns.heatmap(data=t[0][0], cmap='viridis')  
        # heatmap = fig.get_figure()
        # # heatmap.savefig(output_images_path + name[0], dpi=400)
        # heatmap.savefig('@fuhao/transfer/T.png', dpi=400)
        # heatmap.clear()        

        
        # label_0 = nn.functional.interpolate(Mask_fg, size=[x[0].shape[2], x[0].shape[3]], mode='bilinear', align_corners=False)
        # label_1 = nn.functional.interpolate(Mask_fg, size=[x[1].shape[2], x[1].shape[3]], mode='bilinear', align_corners=False)
        # label_2 = nn.functional.interpolate(Mask_fg, size=[x[2].shape[2], x[2].shape[3]], mode='bilinear', align_corners=False)
        # label_3 = nn.functional.interpolate(Mask_fg, size=[x[3].shape[2], x[3].shape[3]], mode='bilinear', align_corners=False)
        # label_4 = nn.functional.interpolate(Mask_fg, size=[x[4].shape[2], x[4].shape[3]], mode='bilinear', align_corners=False)
        
        
        # loss0 = self.mse_loss(label_guide_x0, label_0)
        # loss1 = self.mse_loss(label_guide_x1, label_1)
        # loss2 = self.mse_loss(label_guide_x2, label_2)
        # loss3 = self.mse_loss(label_guide_x3, label_3)
        # loss4 = self.mse_loss(label_guide_x4, label_4)
        
        # loss = loss0 + loss1 + loss2 + loss3 + loss4
        # loss = loss0

        # t = label_0.data.cpu().numpy()
        # fig = sns.heatmap(data=t[0][0], cmap='viridis')  
        # heatmap = fig.get_figure()
        # heatmap.savefig('@fuhao/transfer/label0.png', dpi=400)
        # heatmap.clear()      
        
        # t = label_mask0.data.cpu().numpy()
        # fig = sns.heatmap(data=t[0][0], cmap='viridis')  
        # heatmap = fig.get_figure()
        # heatmap.savefig('@fuhao/transfer/label_mask0.png', dpi=400)
        # heatmap.clear()       
        
        # label_guide_x = tuple([label_guide_x0, label_guide_x1, label_guide_x2, label_guide_x3, label_guide_x4])
        # label_guide_x = tuple([label_guide_x0, None, None, None, None])


        # mask_img = Mask_fg * img
        # save_image(mask_img, '@fuhao/transfer/mask_img.jpg', normalize=True)
        
        # label_guide_x0 = torch.where(label_guide_x0>0.2, 1.0, 0.)
        # label_guide_x0 = nn.functional.interpolate(label_guide_x0, size=[img.shape[2], img.shape[3]], mode='bilinear', align_corners=False)
        # label_guide_img = label_guide_x0 * img
        # save_image(label_guide_img, '@fuhao/transfer/label_guide_img.jpg', normalize=True)
        
        # x = tuple(x)
        bbox_feat = tuple([None, None, None, None, None])
        # return x, loss, label_guide_x        
        return x, loss, bbox_feat



@DETECTORS.register_module()
class Custom_RetinaNet_v548(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(Custom_RetinaNet_v548, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained)
        
        self.mse_loss = torch.nn.MSELoss().cuda()
        
        # self.fpn0 = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.Sigmoid()
        # )

        self.mae_decoder = nn.Sequential(
            nn.Conv2d(256, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
        # self.en_net = nn.Sequential(
        #     nn.Conv2d(3, 16, 3, 1, 1),
        #     nn.ReLU(),
        #     nn.Conv2d(16, 32, 3, 1, 1),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, 3, 1, 1),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 32, 3, 1, 1),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 3, 3, 1, 1),
        #     nn.Tanh()
        # )

        # self.labelnet = ResNet18_FPN(in_channels=1)
        
        # self.masknet = ResNet18_FPN(in_channels=3)

        # self.transfernet = ResNet18_FPN(in_channels=3)
        # self.transfernet.eval()
        # self.convtransfer = nn.Sequential(
        #     nn.Conv2d(256, 3, 3, 1, 1),
        #     nn.Sigmoid()
        # )

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

        # with torch.no_grad():
        #     self.transfernet.eval()
        #     label_x = self.transfernet(img)
        #     label_x = list(label_x)
        #     T = self.convtransfer(label_x[0])
        #     T = nn.functional.interpolate(T, size=[img.shape[2], img.shape[3]], mode='bilinear', align_corners=False)
        #     # img = img / (1 - T)
        #     img = img * T + img
        
        # bbox_h = img.shape[2]
        # bbox_w = img.shape[3]
        # mask_ratio = 0.75
        # num_patches = bbox_h * bbox_w
        # num_mask = int(mask_ratio * num_patches)

        # mask = torch.hstack([
        #     torch.ones(num_patches - num_mask),
        #     torch.zeros(num_mask),
        # ])
        # idx = torch.randperm(num_patches)
        # mask = mask.view(-1)[idx].view((bbox_h, bbox_w))
        
        # img = img * mask.cuda()
        
        # img = self.en_net(img)    
        
        
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        
        # x = list(x)
        # label_guide_x = self.masknet(img)
        # label_guide_x0 = self.fpn0(x[0])
        # label_guide_x1 = self.fpn1(x[1])
        # label_guide_x2 = self.fpn2(x[2])
        # label_guide_x3 = self.fpn3(x[3])
        # label_guide_x4 = self.fpn4(x[4])

        # label_mask0 =  torch.where(label_guide_x0>0.5, 1.0, 0.0)
        # x[0] = x[0] * label_mask0 
        # label_guide_x = tuple([label_guide_x0, label_guide_x1, label_guide_x2, label_guide_x3, label_guide_x4])
        # label_guide_x = tuple([label_guide_x0, None, None, None, None])
        label_guide_x = tuple([None, None, None, None, None])
        
        return x, label_guide_x
    
    
    def extract_feat_label_guide(self, img, img_metas, gt_bboxes):
        """Directly extract features from the backbone+neck."""

        # x = self.backbone(img)
        # if self.with_neck:
        #     x = self.neck(x)
            
        # bbox_feat = self.masknet(img)
            
        # for i in list(x):
        #     print(i.shape)
        # """
        N, C, H, W = img.shape
        Mask_fg = torch.zeros([N, H, W]).cuda()
        wmin,wmax,hmin,hmax = [],[],[],[]
        for i in range(N):
            new_boxxes = torch.ones_like(gt_bboxes[i])
            new_boxxes[:, 0] = gt_bboxes[i][:, 0]/img_metas[i]['img_shape'][1]*W
            new_boxxes[:, 2] = gt_bboxes[i][:, 2]/img_metas[i]['img_shape'][1]*W
            new_boxxes[:, 1] = gt_bboxes[i][:, 1]/img_metas[i]['img_shape'][0]*H
            new_boxxes[:, 3] = gt_bboxes[i][:, 3]/img_metas[i]['img_shape'][0]*H

            wmin.append(torch.floor(new_boxxes[:, 0]).int())
            wmax.append(torch.ceil(new_boxxes[:, 2]).int())
            hmin.append(torch.floor(new_boxxes[:, 1]).int())
            hmax.append(torch.ceil(new_boxxes[:, 3]).int())

            for j in range(len(gt_bboxes[i])):
                # print(hmin[i][j], hmax[i][j], wmin[i][j], wmax[i][j])
                # bbox_h = hmax[i][j] - hmin[i][j] 
                # bbox_w = wmax[i][j] - wmin[i][j] 

                # mask_ratio = 0.75
                # num_patches = bbox_h * bbox_w
                # num_mask = int(mask_ratio * num_patches)

                # mask = torch.hstack([
                #     torch.ones(num_patches - num_mask),
                #     torch.zeros(num_mask),
                # ])
                
                # idx = torch.randperm(num_patches)
                # mask = mask.view(-1)[idx].view((bbox_h, bbox_w))

                Mask_fg[i][hmin[i][j]:hmax[i][j], wmin[i][j]:wmax[i][j]] = 1.0
                
        Mask_fg = Mask_fg.unsqueeze(1)
        # """
        # mask_input = img * Mask_fg
        # with torch.no_grad():
        #     self.transfernet.eval()
        #     img_t = img.detach()
        #     label_guide_x = self.transfernet(img_t)
        
        #     # bbox_feat = list(bbox_feat)
        #     label_guide_x = list(label_guide_x)

        #     T = self.convtransfer(label_guide_x[0])
        #     T = nn.functional.interpolate(T, size=[img.shape[2], img.shape[3]], mode='bilinear', align_corners=False)

        #     # img = img / (1 - T)
        #     img = img * T + img

        input_img = img.detach()
        input_img = input_img * Mask_fg
        # bbox_h = img.shape[2]
        # bbox_w = img.shape[3]
        # mask_ratio = 0.75
        # num_patches = bbox_h * bbox_w
        # num_mask = int(mask_ratio * num_patches)

        # mask = torch.hstack([
        #     torch.ones(num_patches - num_mask),
        #     torch.zeros(num_mask),
        # ])
        
        # # random.shuffle(mask)
        # # mask = torch.randperm(mask)
        # idx = torch.randperm(num_patches)
        # mask = mask.view(-1)[idx].view((bbox_h, bbox_w))

        # img = img * mask.cuda()
        
        # img = self.en_net(img)

        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)

        x = list(x)

        mae_out = self.mae_decoder(x[0])
        input_img = nn.functional.interpolate(input_img, size=[x[0].shape[2], x[0].shape[3]], mode='bilinear', align_corners=False)
        loss_mae = self.mse_loss(mae_out, input_img)
        x = tuple(x)
        # loss0 = self.mse_loss(bbox_feat[0], label_guide_x[0])
        # loss1 = self.mse_loss(bbox_feat[1], label_guide_x[1])
        # loss2 = self.mse_loss(bbox_feat[2], label_guide_x[2])
        # loss3 = self.mse_loss(bbox_feat[3], label_guide_x[3])
        # loss4 = self.mse_loss(bbox_feat[4], label_guide_x[4])
        
        # bbox_feat = tuple(bbox_feat)
        # label_guide_x = tuple(label_guide_x)
        
        loss = 0.0
        loss = loss_mae
        
        # label_guide_x0 = self.fpn0(x[0])
        # label_guide_x1 = self.fpn1(x[1])
        # label_guide_x2 = self.fpn2(x[2])
        # label_guide_x3 = self.fpn3(x[3])
        # label_guide_x4 = self.fpn4(x[4])
        
        # label_guide_x0 = torch.abs(x[0]).mean(1)
        # label_guide_x1 = torch.abs(x[1]).mean(1)
        # label_guide_x2 = torch.abs(x[2]).mean(1)
        # label_guide_x3 = torch.abs(x[3]).mean(1)
        # label_mask0 =  torch.where(label_guide_x0>0.5, 1.0, 0.0)
        # x[0] = x[0] * label_mask0
        
        # mask0 = nn.functional.interpolate(label_mask0, size=[img.shape[2], img.shape[3]], mode='bilinear', align_corners=False)
        # img = img * mask0
        # save_image(img, '@fuhao/transfer/mask0_img.jpg', normalize=True)
        
        
        # t = label_guide_x0.data.cpu().numpy()
        # # fig = sns.heatmap(data=t[0], vmax=9., vmin=6.5, cmap='viridis') 
        # # fig = sns.heatmap(data=t[0], vmax=4., vmin=3.4, cmap='viridis')
        # fig = sns.heatmap(data=t[0][0], cmap='viridis')  
        # heatmap = fig.get_figure()
        # # heatmap.savefig(output_images_path + name[0], dpi=400)
        # heatmap.savefig('@fuhao/transfer/T.png', dpi=400)
        # heatmap.clear()        

        
        # label_0 = nn.functional.interpolate(Mask_fg, size=[x[0].shape[2], x[0].shape[3]], mode='bilinear', align_corners=False)
        # label_1 = nn.functional.interpolate(Mask_fg, size=[x[1].shape[2], x[1].shape[3]], mode='bilinear', align_corners=False)
        # label_2 = nn.functional.interpolate(Mask_fg, size=[x[2].shape[2], x[2].shape[3]], mode='bilinear', align_corners=False)
        # label_3 = nn.functional.interpolate(Mask_fg, size=[x[3].shape[2], x[3].shape[3]], mode='bilinear', align_corners=False)
        # label_4 = nn.functional.interpolate(Mask_fg, size=[x[4].shape[2], x[4].shape[3]], mode='bilinear', align_corners=False)
        
        
        # loss0 = self.mse_loss(label_guide_x0, label_0)
        # loss1 = self.mse_loss(label_guide_x1, label_1)
        # loss2 = self.mse_loss(label_guide_x2, label_2)
        # loss3 = self.mse_loss(label_guide_x3, label_3)
        # loss4 = self.mse_loss(label_guide_x4, label_4)
        
        # loss = loss0 + loss1 + loss2 + loss3 + loss4
        # loss = loss0

        # t = label_0.data.cpu().numpy()
        # fig = sns.heatmap(data=t[0][0], cmap='viridis')  
        # heatmap = fig.get_figure()
        # heatmap.savefig('@fuhao/transfer/label0.png', dpi=400)
        # heatmap.clear()      
        
        # t = label_mask0.data.cpu().numpy()
        # fig = sns.heatmap(data=t[0][0], cmap='viridis')  
        # heatmap = fig.get_figure()
        # heatmap.savefig('@fuhao/transfer/label_mask0.png', dpi=400)
        # heatmap.clear()       
        
        # label_guide_x = tuple([label_guide_x0, label_guide_x1, label_guide_x2, label_guide_x3, label_guide_x4])
        # label_guide_x = tuple([label_guide_x0, None, None, None, None])


        # mask_img = Mask_fg * img
        # save_image(mask_img, '@fuhao/transfer/mask_img.jpg', normalize=True)
        
        # label_guide_x0 = torch.where(label_guide_x0>0.2, 1.0, 0.)
        # label_guide_x0 = nn.functional.interpolate(label_guide_x0, size=[img.shape[2], img.shape[3]], mode='bilinear', align_corners=False)
        # label_guide_img = label_guide_x0 * img
        # save_image(label_guide_img, '@fuhao/transfer/label_guide_img.jpg', normalize=True)
        
        # x = tuple(x)
        bbox_feat = tuple([None, None, None, None, None])
        # return x, loss, label_guide_x        
        return x, loss, bbox_feat

    def extract_feat(self, img, img_metas=None, gt_bboxes=None):
        """Directly extract features from the backbone+neck."""
        x, loss = self.backbone(img, img_metas, gt_bboxes)
        if self.with_neck:
            x = self.neck(x)
        bbox_feat = tuple([None, None, None, None, None])
        return x, loss, bbox_feat

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
        # x, loss_lable_guide, label_guide_x = self.extract_feat_label_guide(img, img_metas, gt_bboxes)
        x, loss_lable_guide, label_guide_x = self.extract_feat(img, img_metas, gt_bboxes)

        
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore, 
                                              label_mask=label_guide_x
                                                )
        
        losses.update({'loss_reconstruct_fg':loss_lable_guide})
        
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
        # x, label_guide_x = self.test_extract_feat_label_guide(img)
        x, loss_lable_guide, label_guide_x = self.extract_feat(img)
        
        outs = self.bbox_head(x, label_guide_x)
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
    
    
# ------------------------------------------------------------------------------------
# 高斯模糊
import torch.nn.functional as F
import cv2
def get_gaussian_kernel(size=3): # 获取高斯kerner 并转为tensor ，size 可以改变模糊程度
    kernel = cv2.getGaussianKernel(size, 0).dot(cv2.getGaussianKernel(size, 0).T)
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    kernel = torch.nn.Parameter(data=kernel, requires_grad=False)
    return kernel

def get_sobel_kernel(im):
    sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')  
    sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
    weight = Variable(torch.from_numpy(sobel_kernel))
    return weight 

def gaussian_blur(x, k, stride=1, padding=0):
    res = []
    x = F.pad(x, (padding, padding, padding, padding), mode='constant', value=0)
    for xx in x.split(1, 1):
        res.append(F.conv2d(xx, k, stride=stride, padding=0))
    return torch.cat(res, 1)

# im = Image.open('./cat.jpg').convert('L')
# # 将图片数据转换为矩阵
# im = np.array(im, dtype='float32')
# # 将图片矩阵转换为pytorch tensor,并适配卷积输入的要求
# im = torch.from_numpy(im.reshape((1, 1, im.shape[0], im.shape[1])))
# gauss_kernel = get_gaussian_kernel(size=9)
# low_gray = gaussian_blur(im, gauss_kernel, padding=0)
# ------------------------------------------------------------------------------------

class JNet(torch.nn.Module):
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

    def forward(self, data, return_conv4=False):
        data = self.conv1(data)
        data = self.conv2(data)
        data = self.conv3(data)
        data = self.conv4(data)
        data1 = self.final(data)
        if return_conv4:
            return data1, data
        else:
            return data1


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


@DETECTORS.register_module()
class Custom_RetinaNet_v549(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(Custom_RetinaNet_v549, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained)
        
        self.mse_loss = torch.nn.MSELoss().cuda()
        
        # self.fpn0 = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.Sigmoid()
        # )

        # self.J_head = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 3, 3, 1, 1),
        #     nn.Tanh()
        # )
        # self.T_head = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 3, 3, 1, 1),
        #     nn.Tanh()
        # )

        # self.J_head = nn.Sequential(
        #     nn.Conv2d(256, 256, 3, 1, 1),
        #     nn.ConvTranspose2d(256 , 128, kernel_size=2, stride=2),
        #     # nn.Conv2d(256, 128, 3, 1, 1),
        #     nn.ReLU(),
        #     # nn.Conv2d(128, 3, 3, 1, 1),
        #     nn.ConvTranspose2d(128 , 64, kernel_size=2, stride=2),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(64 , 3, kernel_size=2, stride=2),
        #     nn.Tanh()
        # ) 
        
        self.t_net = TNet()
        self.j_net = JNet()
        
        self.J_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            # nn.ConvTranspose2d(256 , 128, kernel_size=2, stride=2),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 3, 3, 1, 1),
            # nn.ConvTranspose2d(128 , 64, kernel_size=2, stride=2),
            # nn.ReLU(),
            # nn.ConvTranspose2d(64 , 3, kernel_size=2, stride=2),
            # nn.Tanh()
            nn.Sigmoid()
        )
        
        self.T_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            # nn.ConvTranspose2d(256 , 128, kernel_size=2, stride=2),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 3, 3, 1, 1),
            # nn.ConvTranspose2d(128 , 64, kernel_size=2, stride=2),
            # nn.ReLU(),
            # nn.ConvTranspose2d(64 , 3, kernel_size=2, stride=2),
            # nn.Tanh()
            nn.Sigmoid()
        )
        self.gauss_kernel = get_gaussian_kernel(size=133)
        self.mae_decoder = nn.Sequential(
            nn.Conv2d(256, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
        self.en_net_1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
        )
        
        self.en_net_2 = nn.Sequential(
            nn.ConvTranspose2d(256 , 128, kernel_size=2, stride=2),
            # nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(),
            # nn.Conv2d(128, 3, 3, 1, 1),
            nn.ConvTranspose2d(128 , 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64 , 3, kernel_size=2, stride=2),
            nn.Tanh()
        )
        
        self.ilu_net = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Tanh()
        )

        self.gamma_net = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Tanh()
        )
        
        self.up = nn.Sequential( 
            nn.ConvTranspose2d(3 , 3, kernel_size=2, stride=2),
            nn.ConvTranspose2d(3 , 3, kernel_size=2, stride=2)
        )
        self.mean= torch.tensor([123.675, 116.28, 103.53]).cuda()
        self.std= torch.tensor([58.395, 57.12, 57.375]).cuda()

        self.color_loss = ColorLoss()
        # self.labelnet = ResNet18_FPN(in_channels=1)
        
        # self.masknet = ResNet18_FPN(in_channels=3)

        # self.transfernet = ResNet18_FPN(in_channels=3)
        # self.transfernet.eval()
        # self.convtransfer = nn.Sequential(
        #     nn.Conv2d(256, 3, 3, 1, 1),
        #     nn.Sigmoid()
        # )

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

        # with torch.no_grad():
        #     self.transfernet.eval()
        #     label_x = self.transfernet(img)
        #     label_x = list(label_x)
        #     T = self.convtransfer(label_x[0])
        #     T = nn.functional.interpolate(T, size=[img.shape[2], img.shape[3]], mode='bilinear', align_corners=False)
        #     # img = img / (1 - T)
        #     img = img * T + img
        
        # bbox_h = img.shape[2]
        # bbox_w = img.shape[3]
        # mask_ratio = 0.75
        # num_patches = bbox_h * bbox_w
        # num_mask = int(mask_ratio * num_patches)

        # mask = torch.hstack([
        #     torch.ones(num_patches - num_mask),
        #     torch.zeros(num_mask),
        # ])
        # idx = torch.randperm(num_patches)
        # mask = mask.view(-1)[idx].view((bbox_h, bbox_w))
        
        # img = img * mask.cuda()
        
        # img = self.en_net(img)    
        
        
        # W = img.shape[2] // 4
        # H = img.shape[3] // 4
        # img = nn.functional.interpolate(img, size=[img.shape[2] // 4, img.shape[3] // 4], mode='bilinear', align_corners=True)
        # en_img1 = self.en_net_1(img)
        # en_img2 = self.en_net_2(en_img1)
        
        # # img = self.up(en_img2)   
        # img = en_img2     
        
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        
        # x = list(x)
        # j_out = self.J_head(x[0])
        
        # x = self.backbone(j_out)
        # if self.with_neck:
        #     x = self.neck(x)        
        # x = list(x)
        # x[0] = x[0] + en_img1
        # x = tuple(x)
        
        # x = list(x)
        # label_guide_x = self.masknet(img)
        # label_guide_x0 = self.fpn0(x[0])
        # label_guide_x1 = self.fpn1(x[1])
        # label_guide_x2 = self.fpn2(x[2])
        # label_guide_x3 = self.fpn3(x[3])
        # label_guide_x4 = self.fpn4(x[4])

        # label_mask0 =  torch.where(label_guide_x0>0.5, 1.0, 0.0)
        # x[0] = x[0] * label_mask0 
        # label_guide_x = tuple([label_guide_x0, label_guide_x1, label_guide_x2, label_guide_x3, label_guide_x4])
        # label_guide_x = tuple([label_guide_x0, None, None, None, None])
        label_guide_x = tuple([None, None, None, None, None])
        
        return x, label_guide_x
    
    
    def extract_feat_label_guide(self, img, img_metas, gt_bboxes):
        """Directly extract features from the backbone+neck."""

        # x = self.backbone(img)
        # if self.with_neck:
        #     x = self.neck(x)
            
        # bbox_feat = self.masknet(img)
            
        # for i in list(x):
        #     print(i.shape)
        # """
        
        """ Mask Gen """
        # N, C, H, W = img.shape
        # Mask_fg = torch.zeros([N, H, W]).cuda()
        # wmin,wmax,hmin,hmax = [],[],[],[]
        # for i in range(N):
        #     new_boxxes = torch.ones_like(gt_bboxes[i])
        #     new_boxxes[:, 0] = gt_bboxes[i][:, 0]/img_metas[i]['img_shape'][1]*W
        #     new_boxxes[:, 2] = gt_bboxes[i][:, 2]/img_metas[i]['img_shape'][1]*W
        #     new_boxxes[:, 1] = gt_bboxes[i][:, 1]/img_metas[i]['img_shape'][0]*H
        #     new_boxxes[:, 3] = gt_bboxes[i][:, 3]/img_metas[i]['img_shape'][0]*H

        #     wmin.append(torch.floor(new_boxxes[:, 0]).int())
        #     wmax.append(torch.ceil(new_boxxes[:, 2]).int())
        #     hmin.append(torch.floor(new_boxxes[:, 1]).int())
        #     hmax.append(torch.ceil(new_boxxes[:, 3]).int())

        #     for j in range(len(gt_bboxes[i])):
        #         # print(hmin[i][j], hmax[i][j], wmin[i][j], wmax[i][j])
        #         # bbox_h = hmax[i][j] - hmin[i][j] 
        #         # bbox_w = wmax[i][j] - wmin[i][j] 

        #         # mask_ratio = 0.75
        #         # num_patches = bbox_h * bbox_w
        #         # num_mask = int(mask_ratio * num_patches)

        #         # mask = torch.hstack([
        #         #     torch.ones(num_patches - num_mask),
        #         #     torch.zeros(num_mask),
        #         # ])
                
        #         # idx = torch.randperm(num_patches)
        #         # mask = mask.view(-1)[idx].view((bbox_h, bbox_w))

        #         Mask_fg[i][hmin[i][j]:hmax[i][j], wmin[i][j]:wmax[i][j]] = 1.0
                
        # Mask_fg = Mask_fg.unsqueeze(1)
        """ """
        
        # """
        # mask_input = img * Mask_fg
        # with torch.no_grad():
        #     self.transfernet.eval()
        #     img_t = img.detach()
        #     label_guide_x = self.transfernet(img_t)
        
        #     # bbox_feat = list(bbox_feat)
        #     label_guide_x = list(label_guide_x)

        #     T = self.convtransfer(label_guide_x[0])
        #     T = nn.functional.interpolate(T, size=[img.shape[2], img.shape[3]], mode='bilinear', align_corners=False)

        #     # img = img / (1 - T)
        #     img = img * T + img

        input_img = img.detach()
        H = img.shape[2]
        W = img.shape[3]
        
        # A = get_A(input_img)
        # x = (x - self.mean) / self.std
        
        # input_img = input_img.transpose(1, 3)
        
        # input_img = input_img * self.std + self.mean
        # gamma_factor = random.random()
        # if gamma_factor < 0.1:
        #     gamma_factor = 0.1
        # elif gamma_factor > 1.0:
        #     gamma_factor = 1.0
        # input_img = torch.pow(input_img, 0.5)
        # input_img = (input_img - self.mean) / self.std
        # input_img = input_img.transpose(1, 3)
        # input_img = input_img * Mask_fg
        
        
        # img = nn.functional.interpolate(img, size=[img.shape[2] // 4, img.shape[3] // 4], mode='bilinear', align_corners=True)
        # en_img1 = self.en_net_1(img)
        # en_img2 = self.en_net_2(en_img1)
        
        # img = self.up(en_img2)
        # img = en_img2
        # print(img.shape)
        # bbox_h = img.shape[2]
        # bbox_w = img.shape[3]
        # mask_ratio = 0.75
        # num_patches = bbox_h * bbox_w
        # num_mask = int(mask_ratio * num_patches)

        # mask = torch.hstack([
        #     torch.ones(num_patches - num_mask),
        #     torch.zeros(num_mask),
        # ])
        
        # # random.shuffle(mask)
        # # mask = torch.randperm(mask)
        # idx = torch.randperm(num_patches)
        # mask = mask.view(-1)[idx].view((bbox_h, bbox_w))

        # img = img * mask.cuda()
        
        # img = self.en_net(img)

        x = self.backbone(img)
        if self.with_neck:
            x_neck = self.neck(x)
            
        x_neck = list(x_neck)
        input_img = nn.functional.interpolate(input_img, size=[x_neck[0].shape[2], x_neck[0].shape[3]], mode='bilinear', align_corners=False)
        
        
        input_img = input_img.transpose(1, 3)
        input_img = input_img * self.std + self.mean
        input_img = input_img / 255.
        input_img = input_img.transpose(1, 3)
        
        
        # a_out = get_A(input_img).cuda()
        a_out1 = get_A(input_img[0].unsqueeze(0)).cuda()
        a_out2 = get_A(input_img[1].unsqueeze(0)).cuda()
        a_out = torch.cat([a_out1, a_out2], 0)
        # a_out = gaussian_blur(input_img, self.gauss_kernel, padding=66)
        # j_out = self.J_head(x_neck[0])
        # t_out = self.T_head(x_neck[0])
        j_out = self.j_net(x_neck[0])
        t_out = self.t_net(input_img)
        x_neck = tuple(x_neck)
        # print(j_out.shape, t_out.shape, a_out.shape)
        I_rec = j_out * t_out + (1 - t_out) * a_out
        loss_1 = self.mse_loss(I_rec, input_img)
        # save_image(j_out, './@fuhao/transfer/j_out.jpg', normalize=True)
        # save_image(t_out, './@fuhao/transfer/t_out.jpg', normalize=True)
        # save_image(a_out, './@fuhao/transfer/a_out.jpg', normalize=True)
        # save_image(I_rec, './@fuhao/transfer/I_rec.jpg', normalize=True)
        # save_image(input_img, './@fuhao/transfer/input_img.jpg', normalize=True)
        lam = np.random.beta(1, 1)
        j_out_ = nn.functional.interpolate(j_out, size=[H, W], mode='bilinear', align_corners=False)
        
        j_out_ = j_out_.transpose(1, 3)
        j_out_ = (j_out_ * 255. - self.mean) / self.std
        j_out_ = j_out_.transpose(1, 3)    
    
        
        input_mix = lam * img + (1 - lam) * j_out_

        x = self.backbone(input_mix)
        if self.with_neck:
            x = self.neck(x)

        x = list(x)
        # j_out_mix = self.J_head(x[0])
        j_out_mix = self.j_net(x[0])
        
        # input_mix = lam * input_img + (1 - lam) * j_out
        # j_out_mix = self.j_net(input_mix)
        loss_2 = self.mse_loss(j_out_mix, j_out.detach())

        loss_3 = self.color_loss(j_out)

        total_loss = 1 * loss_1 + 1 * loss_2 + 0.01 * loss_3
        # total_loss = loss_1 + loss_2
        
        loss = total_loss
        
        # x = self.backbone(j_out)
        # if self.with_neck:
        #     x = self.neck(x)

        # x = list(x)
        # # x[0] = x[0] + en_img1
        # neck_feat0 = x[0].detach()
        # loss0 = self.mse_loss(en_img1, neck_feat0)
        # loss1 = self.mse_loss(img, input_img)
        # # mae_out = self.mae_decoder(x[0])
        # # input_img = nn.functional.interpolate(input_img, size=[x[0].shape[2], x[0].shape[3]], mode='bilinear', align_corners=False)
        # # loss_mae = self.mse_loss(mae_out, input_img)
        # x = tuple(x)
        
        # loss0 = self.mse_loss(bbox_feat[0], label_guide_x[0])
        # loss1 = self.mse_loss(bbox_feat[1], label_guide_x[1])
        # loss2 = self.mse_loss(bbox_feat[2], label_guide_x[2])
        # loss3 = self.mse_loss(bbox_feat[3], label_guide_x[3])
        # loss4 = self.mse_loss(bbox_feat[4], label_guide_x[4])
        
        # bbox_feat = tuple(bbox_feat)
        # label_guide_x = tuple(label_guide_x)
        
        # loss = 0.0
        # loss = loss0 + loss1
        
        # label_guide_x0 = self.fpn0(x[0])
        # label_guide_x1 = self.fpn1(x[1])
        # label_guide_x2 = self.fpn2(x[2])
        # label_guide_x3 = self.fpn3(x[3])
        # label_guide_x4 = self.fpn4(x[4])
        
        # label_guide_x0 = torch.abs(x[0]).mean(1)
        # label_guide_x1 = torch.abs(x[1]).mean(1)
        # label_guide_x2 = torch.abs(x[2]).mean(1)
        # label_guide_x3 = torch.abs(x[3]).mean(1)
        # label_mask0 =  torch.where(label_guide_x0>0.5, 1.0, 0.0)
        # x[0] = x[0] * label_mask0
        
        # mask0 = nn.functional.interpolate(label_mask0, size=[img.shape[2], img.shape[3]], mode='bilinear', align_corners=False)
        # img = img * mask0
        # save_image(img, '@fuhao/transfer/mask0_img.jpg', normalize=True)
        
        
        # t = label_guide_x0.data.cpu().numpy()
        # # fig = sns.heatmap(data=t[0], vmax=9., vmin=6.5, cmap='viridis') 
        # # fig = sns.heatmap(data=t[0], vmax=4., vmin=3.4, cmap='viridis')
        # fig = sns.heatmap(data=t[0][0], cmap='viridis')  
        # heatmap = fig.get_figure()
        # # heatmap.savefig(output_images_path + name[0], dpi=400)
        # heatmap.savefig('@fuhao/transfer/T.png', dpi=400)
        # heatmap.clear()        

        
        # label_0 = nn.functional.interpolate(Mask_fg, size=[x[0].shape[2], x[0].shape[3]], mode='bilinear', align_corners=False)
        # label_1 = nn.functional.interpolate(Mask_fg, size=[x[1].shape[2], x[1].shape[3]], mode='bilinear', align_corners=False)
        # label_2 = nn.functional.interpolate(Mask_fg, size=[x[2].shape[2], x[2].shape[3]], mode='bilinear', align_corners=False)
        # label_3 = nn.functional.interpolate(Mask_fg, size=[x[3].shape[2], x[3].shape[3]], mode='bilinear', align_corners=False)
        # label_4 = nn.functional.interpolate(Mask_fg, size=[x[4].shape[2], x[4].shape[3]], mode='bilinear', align_corners=False)
        
        
        # loss0 = self.mse_loss(label_guide_x0, label_0)
        # loss1 = self.mse_loss(label_guide_x1, label_1)
        # loss2 = self.mse_loss(label_guide_x2, label_2)
        # loss3 = self.mse_loss(label_guide_x3, label_3)
        # loss4 = self.mse_loss(label_guide_x4, label_4)
        
        # loss = loss0 + loss1 + loss2 + loss3 + loss4
        # loss = loss0

        # t = label_0.data.cpu().numpy()
        # fig = sns.heatmap(data=t[0][0], cmap='viridis')  
        # heatmap = fig.get_figure()
        # heatmap.savefig('@fuhao/transfer/label0.png', dpi=400)
        # heatmap.clear()      
        
        # t = label_mask0.data.cpu().numpy()
        # fig = sns.heatmap(data=t[0][0], cmap='viridis')  
        # heatmap = fig.get_figure()
        # heatmap.savefig('@fuhao/transfer/label_mask0.png', dpi=400)
        # heatmap.clear()       
        
        # label_guide_x = tuple([label_guide_x0, label_guide_x1, label_guide_x2, label_guide_x3, label_guide_x4])
        # label_guide_x = tuple([label_guide_x0, None, None, None, None])


        # mask_img = Mask_fg * img
        # save_image(mask_img, '@fuhao/transfer/mask_img.jpg', normalize=True)
        
        # label_guide_x0 = torch.where(label_guide_x0>0.2, 1.0, 0.)
        # label_guide_x0 = nn.functional.interpolate(label_guide_x0, size=[img.shape[2], img.shape[3]], mode='bilinear', align_corners=False)
        # label_guide_img = label_guide_x0 * img
        # save_image(label_guide_img, '@fuhao/transfer/label_guide_img.jpg', normalize=True)
        
        # x = tuple(x)
        bbox_feat = tuple([None, None, None, None, None])
        # return x, loss, label_guide_x        
        return x_neck, loss, bbox_feat

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
        
        losses.update({'loss_reconstruct_fg':loss_lable_guide})
        
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
        
        outs = self.bbox_head(x, label_guide_x)
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
class Custom_RetinaNet_v549_1(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(Custom_RetinaNet_v549_1, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained)
        
        self.mse_loss = torch.nn.MSELoss().cuda()
        
        # self.fpn0 = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.Sigmoid()
        # )

        # self.J_head = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 3, 3, 1, 1),
        #     nn.Tanh()
        # )
        # self.T_head = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 3, 3, 1, 1),
        #     nn.Tanh()
        # )

        # self.J_head = nn.Sequential(
        #     nn.Conv2d(256, 256, 3, 1, 1),
        #     nn.ConvTranspose2d(256 , 128, kernel_size=2, stride=2),
        #     # nn.Conv2d(256, 128, 3, 1, 1),
        #     nn.ReLU(),
        #     # nn.Conv2d(128, 3, 3, 1, 1),
        #     nn.ConvTranspose2d(128 , 64, kernel_size=2, stride=2),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(64 , 3, kernel_size=2, stride=2),
        #     nn.Tanh()
        # ) 
        
        self.t_net = TNet()
        self.j_net = JNet()
        
        self.J_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            # nn.ConvTranspose2d(256 , 128, kernel_size=2, stride=2),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 3, 3, 1, 1),
            # nn.ConvTranspose2d(128 , 64, kernel_size=2, stride=2),
            # nn.ReLU(),
            # nn.ConvTranspose2d(64 , 3, kernel_size=2, stride=2),
            # nn.Tanh()
            nn.Sigmoid()
        )
        
        self.T_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            # nn.ConvTranspose2d(256 , 128, kernel_size=2, stride=2),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 3, 3, 1, 1),
            # nn.ConvTranspose2d(128 , 64, kernel_size=2, stride=2),
            # nn.ReLU(),
            # nn.ConvTranspose2d(64 , 3, kernel_size=2, stride=2),
            # nn.Tanh()
            nn.Sigmoid()
        )
        self.gauss_kernel = get_gaussian_kernel(size=133)
        self.mae_decoder = nn.Sequential(
            nn.Conv2d(256, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
        self.en_net_1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
        )
        
        self.en_net_2 = nn.Sequential(
            nn.ConvTranspose2d(256 , 128, kernel_size=2, stride=2),
            # nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(),
            # nn.Conv2d(128, 3, 3, 1, 1),
            nn.ConvTranspose2d(128 , 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64 , 3, kernel_size=2, stride=2),
            nn.Tanh()
        )
        
        self.ilu_net = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Tanh()
        )

        self.gamma_net = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Tanh()
        )
        
        self.up = nn.Sequential( 
            nn.ConvTranspose2d(3 , 3, kernel_size=2, stride=2),
            nn.ConvTranspose2d(3 , 3, kernel_size=2, stride=2)
        )
        self.mean= torch.tensor([123.675, 116.28, 103.53]).cuda()
        self.std= torch.tensor([58.395, 57.12, 57.375]).cuda()

        self.color_loss = ColorLoss()
        # self.labelnet = ResNet18_FPN(in_channels=1)
        
        # self.masknet = ResNet18_FPN(in_channels=3)

        # self.transfernet = ResNet18_FPN(in_channels=3)
        # self.transfernet.eval()
        # self.convtransfer = nn.Sequential(
        #     nn.Conv2d(256, 3, 3, 1, 1),
        #     nn.Sigmoid()
        # )

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

        # with torch.no_grad():
        #     self.transfernet.eval()
        #     label_x = self.transfernet(img)
        #     label_x = list(label_x)
        #     T = self.convtransfer(label_x[0])
        #     T = nn.functional.interpolate(T, size=[img.shape[2], img.shape[3]], mode='bilinear', align_corners=False)
        #     # img = img / (1 - T)
        #     img = img * T + img
        
        # bbox_h = img.shape[2]
        # bbox_w = img.shape[3]
        # mask_ratio = 0.75
        # num_patches = bbox_h * bbox_w
        # num_mask = int(mask_ratio * num_patches)

        # mask = torch.hstack([
        #     torch.ones(num_patches - num_mask),
        #     torch.zeros(num_mask),
        # ])
        # idx = torch.randperm(num_patches)
        # mask = mask.view(-1)[idx].view((bbox_h, bbox_w))
        
        # img = img * mask.cuda()
        
        # img = self.en_net(img)    
        
        
        # W = img.shape[2] // 4
        # H = img.shape[3] // 4
        # img = nn.functional.interpolate(img, size=[img.shape[2] // 4, img.shape[3] // 4], mode='bilinear', align_corners=True)
        # en_img1 = self.en_net_1(img)
        # en_img2 = self.en_net_2(en_img1)
        
        # # img = self.up(en_img2)   
        # img = en_img2     
        
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        
        # x = list(x)
        # j_out = self.J_head(x[0])
        
        # x = self.backbone(j_out)
        # if self.with_neck:
        #     x = self.neck(x)        
        # x = list(x)
        # x[0] = x[0] + en_img1
        # x = tuple(x)
        
        # x = list(x)
        # label_guide_x = self.masknet(img)
        # label_guide_x0 = self.fpn0(x[0])
        # label_guide_x1 = self.fpn1(x[1])
        # label_guide_x2 = self.fpn2(x[2])
        # label_guide_x3 = self.fpn3(x[3])
        # label_guide_x4 = self.fpn4(x[4])

        # label_mask0 =  torch.where(label_guide_x0>0.5, 1.0, 0.0)
        # x[0] = x[0] * label_mask0 
        # label_guide_x = tuple([label_guide_x0, label_guide_x1, label_guide_x2, label_guide_x3, label_guide_x4])
        # label_guide_x = tuple([label_guide_x0, None, None, None, None])
        label_guide_x = tuple([None, None, None, None, None])
        
        return x, label_guide_x
    
    
    def extract_feat_label_guide(self, img, img_metas, gt_bboxes):
        """Directly extract features from the backbone+neck."""

        # x = self.backbone(img)
        # if self.with_neck:
        #     x = self.neck(x)
            
        # bbox_feat = self.masknet(img)
            
        # for i in list(x):
        #     print(i.shape)
        # """
        
        """ Mask Gen """
        # N, C, H, W = img.shape
        # Mask_fg = torch.zeros([N, H, W]).cuda()
        # wmin,wmax,hmin,hmax = [],[],[],[]
        # for i in range(N):
        #     new_boxxes = torch.ones_like(gt_bboxes[i])
        #     new_boxxes[:, 0] = gt_bboxes[i][:, 0]/img_metas[i]['img_shape'][1]*W
        #     new_boxxes[:, 2] = gt_bboxes[i][:, 2]/img_metas[i]['img_shape'][1]*W
        #     new_boxxes[:, 1] = gt_bboxes[i][:, 1]/img_metas[i]['img_shape'][0]*H
        #     new_boxxes[:, 3] = gt_bboxes[i][:, 3]/img_metas[i]['img_shape'][0]*H

        #     wmin.append(torch.floor(new_boxxes[:, 0]).int())
        #     wmax.append(torch.ceil(new_boxxes[:, 2]).int())
        #     hmin.append(torch.floor(new_boxxes[:, 1]).int())
        #     hmax.append(torch.ceil(new_boxxes[:, 3]).int())

        #     for j in range(len(gt_bboxes[i])):
        #         # print(hmin[i][j], hmax[i][j], wmin[i][j], wmax[i][j])
        #         # bbox_h = hmax[i][j] - hmin[i][j] 
        #         # bbox_w = wmax[i][j] - wmin[i][j] 

        #         # mask_ratio = 0.75
        #         # num_patches = bbox_h * bbox_w
        #         # num_mask = int(mask_ratio * num_patches)

        #         # mask = torch.hstack([
        #         #     torch.ones(num_patches - num_mask),
        #         #     torch.zeros(num_mask),
        #         # ])
                
        #         # idx = torch.randperm(num_patches)
        #         # mask = mask.view(-1)[idx].view((bbox_h, bbox_w))

        #         Mask_fg[i][hmin[i][j]:hmax[i][j], wmin[i][j]:wmax[i][j]] = 1.0
                
        # Mask_fg = Mask_fg.unsqueeze(1)
        """ """
        
        # """
        # mask_input = img * Mask_fg
        # with torch.no_grad():
        #     self.transfernet.eval()
        #     img_t = img.detach()
        #     label_guide_x = self.transfernet(img_t)
        
        #     # bbox_feat = list(bbox_feat)
        #     label_guide_x = list(label_guide_x)

        #     T = self.convtransfer(label_guide_x[0])
        #     T = nn.functional.interpolate(T, size=[img.shape[2], img.shape[3]], mode='bilinear', align_corners=False)

        #     # img = img / (1 - T)
        #     img = img * T + img

        input_img = img.detach()
        H = img.shape[2]
        W = img.shape[3]
        
        # A = get_A(input_img)
        # x = (x - self.mean) / self.std
        
        # input_img = input_img.transpose(1, 3)
        
        # input_img = input_img * self.std + self.mean
        # gamma_factor = random.random()
        # if gamma_factor < 0.1:
        #     gamma_factor = 0.1
        # elif gamma_factor > 1.0:
        #     gamma_factor = 1.0
        # input_img = torch.pow(input_img, 0.5)
        # input_img = (input_img - self.mean) / self.std
        # input_img = input_img.transpose(1, 3)
        # input_img = input_img * Mask_fg
        
        
        # img = nn.functional.interpolate(img, size=[img.shape[2] // 4, img.shape[3] // 4], mode='bilinear', align_corners=True)
        # en_img1 = self.en_net_1(img)
        # en_img2 = self.en_net_2(en_img1)
        
        # img = self.up(en_img2)
        # img = en_img2
        # print(img.shape)
        # bbox_h = img.shape[2]
        # bbox_w = img.shape[3]
        # mask_ratio = 0.75
        # num_patches = bbox_h * bbox_w
        # num_mask = int(mask_ratio * num_patches)

        # mask = torch.hstack([
        #     torch.ones(num_patches - num_mask),
        #     torch.zeros(num_mask),
        # ])
        
        # # random.shuffle(mask)
        # # mask = torch.randperm(mask)
        # idx = torch.randperm(num_patches)
        # mask = mask.view(-1)[idx].view((bbox_h, bbox_w))

        # img = img * mask.cuda()
        
        # img = self.en_net(img)

        x = self.backbone(img)
        if self.with_neck:
            x_neck = self.neck(x)
            
        x_neck = list(x_neck)
        input_img = nn.functional.interpolate(input_img, size=[x_neck[0].shape[2], x_neck[0].shape[3]], mode='bilinear', align_corners=False)
        
        
        input_img = input_img.transpose(1, 3)
        input_img = input_img * self.std + self.mean
        input_img = input_img / 255.
        input_img = input_img.transpose(1, 3)
        
        
        # a_out = get_A(input_img).cuda()
        a_out1 = get_A(input_img[0].unsqueeze(0)).cuda()
        a_out2 = get_A(input_img[1].unsqueeze(0)).cuda()
        a_out = torch.cat([a_out1, a_out2], 0)
        # a_out = gaussian_blur(input_img, self.gauss_kernel, padding=66)
        # j_out = self.J_head(x_neck[0])
        # t_out = self.T_head(x_neck[0])
        t_out = self.j_net(x_neck[0])
        j_out = self.t_net(input_img)
        x_neck = tuple(x_neck)
        # print(j_out.shape, t_out.shape, a_out.shape)
        I_rec = j_out * t_out + (1 - t_out) * a_out
        loss_1 = self.mse_loss(I_rec, input_img)
        # save_image(j_out, './@fuhao/transfer/j_out.jpg', normalize=True)
        # save_image(t_out, './@fuhao/transfer/t_out.jpg', normalize=True)
        # save_image(a_out, './@fuhao/transfer/a_out.jpg', normalize=True)
        # save_image(I_rec, './@fuhao/transfer/I_rec.jpg', normalize=True)
        # save_image(input_img, './@fuhao/transfer/input_img.jpg', normalize=True)
        lam = np.random.beta(1, 1)
        # j_out_ = nn.functional.interpolate(j_out, size=[H, W], mode='bilinear', align_corners=False)
        j_out_ = j_out
        
        j_out_ = j_out_.transpose(1, 3)
        j_out_ = (j_out_ * 255. - self.mean) / self.std
        j_out_ = j_out_.transpose(1, 3)    
    
        
        input_mix = lam * input_img + (1 - lam) * j_out_

        # x = self.backbone(input_mix)
        # if self.with_neck:
        #     x = self.neck(x)

        # x = list(x)
        # # j_out_mix = self.J_head(x[0])
        # j_out_mix = self.j_net(x[0])
        
        j_out_mix = self.t_net(input_mix)
        
        # input_mix = lam * input_img + (1 - lam) * j_out
        # j_out_mix = self.j_net(input_mix)
        loss_2 = self.mse_loss(j_out_mix, j_out.detach())

        loss_3 = self.color_loss(j_out)

        total_loss = 1 * loss_1 + 1 * loss_2 + 0.01 * loss_3
        # total_loss = loss_1 + loss_2
        
        loss = total_loss
        
        # x = self.backbone(j_out)
        # if self.with_neck:
        #     x = self.neck(x)

        # x = list(x)
        # # x[0] = x[0] + en_img1
        # neck_feat0 = x[0].detach()
        # loss0 = self.mse_loss(en_img1, neck_feat0)
        # loss1 = self.mse_loss(img, input_img)
        # # mae_out = self.mae_decoder(x[0])
        # # input_img = nn.functional.interpolate(input_img, size=[x[0].shape[2], x[0].shape[3]], mode='bilinear', align_corners=False)
        # # loss_mae = self.mse_loss(mae_out, input_img)
        # x = tuple(x)
        
        # loss0 = self.mse_loss(bbox_feat[0], label_guide_x[0])
        # loss1 = self.mse_loss(bbox_feat[1], label_guide_x[1])
        # loss2 = self.mse_loss(bbox_feat[2], label_guide_x[2])
        # loss3 = self.mse_loss(bbox_feat[3], label_guide_x[3])
        # loss4 = self.mse_loss(bbox_feat[4], label_guide_x[4])
        
        # bbox_feat = tuple(bbox_feat)
        # label_guide_x = tuple(label_guide_x)
        
        # loss = 0.0
        # loss = loss0 + loss1
        
        # label_guide_x0 = self.fpn0(x[0])
        # label_guide_x1 = self.fpn1(x[1])
        # label_guide_x2 = self.fpn2(x[2])
        # label_guide_x3 = self.fpn3(x[3])
        # label_guide_x4 = self.fpn4(x[4])
        
        # label_guide_x0 = torch.abs(x[0]).mean(1)
        # label_guide_x1 = torch.abs(x[1]).mean(1)
        # label_guide_x2 = torch.abs(x[2]).mean(1)
        # label_guide_x3 = torch.abs(x[3]).mean(1)
        # label_mask0 =  torch.where(label_guide_x0>0.5, 1.0, 0.0)
        # x[0] = x[0] * label_mask0
        
        # mask0 = nn.functional.interpolate(label_mask0, size=[img.shape[2], img.shape[3]], mode='bilinear', align_corners=False)
        # img = img * mask0
        # save_image(img, '@fuhao/transfer/mask0_img.jpg', normalize=True)
        
        
        # t = label_guide_x0.data.cpu().numpy()
        # # fig = sns.heatmap(data=t[0], vmax=9., vmin=6.5, cmap='viridis') 
        # # fig = sns.heatmap(data=t[0], vmax=4., vmin=3.4, cmap='viridis')
        # fig = sns.heatmap(data=t[0][0], cmap='viridis')  
        # heatmap = fig.get_figure()
        # # heatmap.savefig(output_images_path + name[0], dpi=400)
        # heatmap.savefig('@fuhao/transfer/T.png', dpi=400)
        # heatmap.clear()        

        
        # label_0 = nn.functional.interpolate(Mask_fg, size=[x[0].shape[2], x[0].shape[3]], mode='bilinear', align_corners=False)
        # label_1 = nn.functional.interpolate(Mask_fg, size=[x[1].shape[2], x[1].shape[3]], mode='bilinear', align_corners=False)
        # label_2 = nn.functional.interpolate(Mask_fg, size=[x[2].shape[2], x[2].shape[3]], mode='bilinear', align_corners=False)
        # label_3 = nn.functional.interpolate(Mask_fg, size=[x[3].shape[2], x[3].shape[3]], mode='bilinear', align_corners=False)
        # label_4 = nn.functional.interpolate(Mask_fg, size=[x[4].shape[2], x[4].shape[3]], mode='bilinear', align_corners=False)
        
        
        # loss0 = self.mse_loss(label_guide_x0, label_0)
        # loss1 = self.mse_loss(label_guide_x1, label_1)
        # loss2 = self.mse_loss(label_guide_x2, label_2)
        # loss3 = self.mse_loss(label_guide_x3, label_3)
        # loss4 = self.mse_loss(label_guide_x4, label_4)
        
        # loss = loss0 + loss1 + loss2 + loss3 + loss4
        # loss = loss0

        # t = label_0.data.cpu().numpy()
        # fig = sns.heatmap(data=t[0][0], cmap='viridis')  
        # heatmap = fig.get_figure()
        # heatmap.savefig('@fuhao/transfer/label0.png', dpi=400)
        # heatmap.clear()      
        
        # t = label_mask0.data.cpu().numpy()
        # fig = sns.heatmap(data=t[0][0], cmap='viridis')  
        # heatmap = fig.get_figure()
        # heatmap.savefig('@fuhao/transfer/label_mask0.png', dpi=400)
        # heatmap.clear()       
        
        # label_guide_x = tuple([label_guide_x0, label_guide_x1, label_guide_x2, label_guide_x3, label_guide_x4])
        # label_guide_x = tuple([label_guide_x0, None, None, None, None])


        # mask_img = Mask_fg * img
        # save_image(mask_img, '@fuhao/transfer/mask_img.jpg', normalize=True)
        
        # label_guide_x0 = torch.where(label_guide_x0>0.2, 1.0, 0.)
        # label_guide_x0 = nn.functional.interpolate(label_guide_x0, size=[img.shape[2], img.shape[3]], mode='bilinear', align_corners=False)
        # label_guide_img = label_guide_x0 * img
        # save_image(label_guide_img, '@fuhao/transfer/label_guide_img.jpg', normalize=True)
        
        # x = tuple(x)
        bbox_feat = tuple([None, None, None, None, None])
        # return x, loss, label_guide_x        
        return x_neck, loss, bbox_feat

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
        
        losses.update({'loss_reconstruct_fg':loss_lable_guide})
        
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
        
        outs = self.bbox_head(x, label_guide_x)
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
class Custom_RetinaNet_v549_2(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(Custom_RetinaNet_v549_2, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained)
        
        self.mse_loss = torch.nn.MSELoss().cuda()
        
        # self.fpn0 = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.Sigmoid()
        # )

        # self.J_head = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 3, 3, 1, 1),
        #     nn.Tanh()
        # )
        # self.T_head = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 3, 3, 1, 1),
        #     nn.Tanh()
        # )

        # self.J_head = nn.Sequential(
        #     nn.Conv2d(256, 256, 3, 1, 1),
        #     nn.ConvTranspose2d(256 , 128, kernel_size=2, stride=2),
        #     # nn.Conv2d(256, 128, 3, 1, 1),
        #     nn.ReLU(),
        #     # nn.Conv2d(128, 3, 3, 1, 1),
        #     nn.ConvTranspose2d(128 , 64, kernel_size=2, stride=2),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(64 , 3, kernel_size=2, stride=2),
        #     nn.Tanh()
        # ) 
        
        self.t_net = TNet()
        self.j_net = JNet()
        
        self.J_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            # nn.ConvTranspose2d(256 , 128, kernel_size=2, stride=2),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 3, 3, 1, 1),
            # nn.ConvTranspose2d(128 , 64, kernel_size=2, stride=2),
            # nn.ReLU(),
            # nn.ConvTranspose2d(64 , 3, kernel_size=2, stride=2),
            # nn.Tanh()
            nn.Sigmoid()
        )
        
        self.T_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256 , 128, kernel_size=2, stride=2),
            # nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 3, 3, 1, 1),
            # nn.ConvTranspose2d(128 , 64, kernel_size=2, stride=2),
            # nn.ReLU(),
            # nn.ConvTranspose2d(64 , 3, kernel_size=2, stride=2),
            # nn.Tanh()
            nn.Sigmoid()
        )
        self.gauss_kernel = get_gaussian_kernel(size=133)
        self.mae_decoder = nn.Sequential(
            nn.Conv2d(256, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
        self.en_net_1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
        )
        
        self.en_net_2 = nn.Sequential(
            nn.ConvTranspose2d(256 , 128, kernel_size=2, stride=2),
            # nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(),
            # nn.Conv2d(128, 3, 3, 1, 1),
            nn.ConvTranspose2d(128 , 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64 , 3, kernel_size=2, stride=2),
            nn.Tanh()
        )
        
        self.ilu_net = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Tanh()
        )

        self.gamma_net = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Tanh()
        )
        
        self.up = nn.Sequential( 
            nn.ConvTranspose2d(3 , 3, kernel_size=2, stride=2),
            nn.ConvTranspose2d(3 , 3, kernel_size=2, stride=2)
        )
        self.mean= torch.tensor([123.675, 116.28, 103.53]).cuda()
        self.std= torch.tensor([58.395, 57.12, 57.375]).cuda()

        self.color_loss = ColorLoss()
        # self.labelnet = ResNet18_FPN(in_channels=1)
        
        # self.masknet = ResNet18_FPN(in_channels=3)

        # self.transfernet = ResNet18_FPN(in_channels=3)
        # self.transfernet.eval()
        # self.convtransfer = nn.Sequential(
        #     nn.Conv2d(256, 3, 3, 1, 1),
        #     nn.Sigmoid()
        # )

        weights_path = '@fuhao/exp/500Adaptive/540mask_retina/547_3fenzhi/checkpoints/RUIE_final.pth'
        print('+---------------------------------------------+')
        print('+ Loading weights into ' + 'uwnet' + ' state dict.. +')
        print('+ Weights path: ', weights_path)
        print('+---------------------------------------------+')
        model_dict = self.j_net.state_dict()
        
        num = 0
        for k in model_dict.keys():
            print('model =>', k)
            num += 1
        print('total num: ', num)
        num = 0

        pretrained_dict = torch.load(weights_path)
        # pretrained_dict = pretrained_dict['state_dict']

        for k in pretrained_dict.keys():
            print('checkpoint =>', k)
            num += 1
        print('total num: ', num)
        num = 0

        momo_dict = {}
        for k, v in pretrained_dict.items(): 
            # if k in model_dict.keys() and cfg.load_checkpoint['load_keys'] in k: # 注意要修改的地方
            if 'image_net' in k:
                k = k.replace('image_net.', '')
                print(k)
            if k in model_dict.keys():
                if pretrained_dict['image_net.' + k].size() == model_dict[k].size():
                    momo_dict.update({k: v})

        # momo_dict = {k:v for 'module.'+k, v in pretrained_dict['state_dict'].items() if k in model_dict.keys() and pretrained_dict['state_dict'][k].shape() == model_dict[k].shape() }
        for k, v in momo_dict.items():
            print('model load => ', k)
            num += 1
        print('total num: ', num)

        model_dict.update(momo_dict)
        self.j_net.load_state_dict(model_dict)
        print('+---------------------------------------------+')
        print('+                 Finished！                  +')
        print('+---------------------------------------------+')
        
    def test_extract_feat_label_guide(self, img):
        """Directly extract features from the backbone+neck."""

        # with torch.no_grad():
        #     self.transfernet.eval()
        #     label_x = self.transfernet(img)
        #     label_x = list(label_x)
        #     T = self.convtransfer(label_x[0])
        #     T = nn.functional.interpolate(T, size=[img.shape[2], img.shape[3]], mode='bilinear', align_corners=False)
        #     # img = img / (1 - T)
        #     img = img * T + img
        
        # bbox_h = img.shape[2]
        # bbox_w = img.shape[3]
        # mask_ratio = 0.75
        # num_patches = bbox_h * bbox_w
        # num_mask = int(mask_ratio * num_patches)

        # mask = torch.hstack([
        #     torch.ones(num_patches - num_mask),
        #     torch.zeros(num_mask),
        # ])
        # idx = torch.randperm(num_patches)
        # mask = mask.view(-1)[idx].view((bbox_h, bbox_w))
        
        # img = img * mask.cuda()
        
        # img = self.en_net(img)    
        
        
        # W = img.shape[2] // 4
        # H = img.shape[3] // 4
        # img = nn.functional.interpolate(img, size=[img.shape[2] // 4, img.shape[3] // 4], mode='bilinear', align_corners=True)
        # en_img1 = self.en_net_1(img)
        # en_img2 = self.en_net_2(en_img1)
        
        # # img = self.up(en_img2)   
        # img = en_img2     
        
        input_img = img.detach()
        H = img.shape[2]
        W = img.shape[3]
        input_img = nn.functional.interpolate(input_img, size=[H//4, W//4], mode='bilinear', align_corners=False)
        input_img = input_img.transpose(1, 3)
        input_img = input_img * self.std + self.mean
        input_img = input_img / 255.
        input_img = input_img.transpose(1, 3)
        j_out, data_conv4 = self.j_net(input_img, return_conv4=True)

        x = self.backbone(img, data_conv4)
        # x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        
        # x = list(x)
        # j_out = self.J_head(x[0])
        
        # x = self.backbone(j_out)
        # if self.with_neck:
        #     x = self.neck(x)        
        # x = list(x)
        # x[0] = x[0] + en_img1
        # x = tuple(x)
        
        # x = list(x)
        # label_guide_x = self.masknet(img)
        # label_guide_x0 = self.fpn0(x[0])
        # label_guide_x1 = self.fpn1(x[1])
        # label_guide_x2 = self.fpn2(x[2])
        # label_guide_x3 = self.fpn3(x[3])
        # label_guide_x4 = self.fpn4(x[4])

        # label_mask0 =  torch.where(label_guide_x0>0.5, 1.0, 0.0)
        # x[0] = x[0] * label_mask0 
        # label_guide_x = tuple([label_guide_x0, label_guide_x1, label_guide_x2, label_guide_x3, label_guide_x4])
        # label_guide_x = tuple([label_guide_x0, None, None, None, None])
        label_guide_x = tuple([None, None, None, None, None])
        
        return x, label_guide_x
    
    
    def extract_feat_label_guide(self, img, img_metas, gt_bboxes):
        """Directly extract features from the backbone+neck."""

        # x = self.backbone(img)
        # if self.with_neck:
        #     x = self.neck(x)
            
        # bbox_feat = self.masknet(img)
            
        # for i in list(x):
        #     print(i.shape)
        # """
        
        """ Mask Gen """
        # N, C, H, W = img.shape
        # Mask_fg = torch.zeros([N, H, W]).cuda()
        # wmin,wmax,hmin,hmax = [],[],[],[]
        # for i in range(N):
        #     new_boxxes = torch.ones_like(gt_bboxes[i])
        #     new_boxxes[:, 0] = gt_bboxes[i][:, 0]/img_metas[i]['img_shape'][1]*W
        #     new_boxxes[:, 2] = gt_bboxes[i][:, 2]/img_metas[i]['img_shape'][1]*W
        #     new_boxxes[:, 1] = gt_bboxes[i][:, 1]/img_metas[i]['img_shape'][0]*H
        #     new_boxxes[:, 3] = gt_bboxes[i][:, 3]/img_metas[i]['img_shape'][0]*H

        #     wmin.append(torch.floor(new_boxxes[:, 0]).int())
        #     wmax.append(torch.ceil(new_boxxes[:, 2]).int())
        #     hmin.append(torch.floor(new_boxxes[:, 1]).int())
        #     hmax.append(torch.ceil(new_boxxes[:, 3]).int())

        #     for j in range(len(gt_bboxes[i])):
        #         # print(hmin[i][j], hmax[i][j], wmin[i][j], wmax[i][j])
        #         # bbox_h = hmax[i][j] - hmin[i][j] 
        #         # bbox_w = wmax[i][j] - wmin[i][j] 

        #         # mask_ratio = 0.75
        #         # num_patches = bbox_h * bbox_w
        #         # num_mask = int(mask_ratio * num_patches)

        #         # mask = torch.hstack([
        #         #     torch.ones(num_patches - num_mask),
        #         #     torch.zeros(num_mask),
        #         # ])
                
        #         # idx = torch.randperm(num_patches)
        #         # mask = mask.view(-1)[idx].view((bbox_h, bbox_w))

        #         Mask_fg[i][hmin[i][j]:hmax[i][j], wmin[i][j]:wmax[i][j]] = 1.0
                
        # Mask_fg = Mask_fg.unsqueeze(1)
        """ """
        
        # """
        # mask_input = img * Mask_fg
        # with torch.no_grad():
        #     self.transfernet.eval()
        #     img_t = img.detach()
        #     label_guide_x = self.transfernet(img_t)
        
        #     # bbox_feat = list(bbox_feat)
        #     label_guide_x = list(label_guide_x)

        #     T = self.convtransfer(label_guide_x[0])
        #     T = nn.functional.interpolate(T, size=[img.shape[2], img.shape[3]], mode='bilinear', align_corners=False)

        #     # img = img / (1 - T)
        #     img = img * T + img

        input_img = img.detach()
        H = img.shape[2]
        W = img.shape[3]
        
        # A = get_A(input_img)
        # x = (x - self.mean) / self.std
        
        # input_img = input_img.transpose(1, 3)
        
        # input_img = input_img * self.std + self.mean
        # gamma_factor = random.random()
        # if gamma_factor < 0.1:
        #     gamma_factor = 0.1
        # elif gamma_factor > 1.0:
        #     gamma_factor = 1.0
        # input_img = torch.pow(input_img, 0.5)
        # input_img = (input_img - self.mean) / self.std
        # input_img = input_img.transpose(1, 3)
        # input_img = input_img * Mask_fg
        
        
        # img = nn.functional.interpolate(img, size=[img.shape[2] // 4, img.shape[3] // 4], mode='bilinear', align_corners=True)
        # en_img1 = self.en_net_1(img)
        # en_img2 = self.en_net_2(en_img1)
        
        # img = self.up(en_img2)
        # img = en_img2
        # print(img.shape)
        # bbox_h = img.shape[2]
        # bbox_w = img.shape[3]
        # mask_ratio = 0.75
        # num_patches = bbox_h * bbox_w
        # num_mask = int(mask_ratio * num_patches)

        # mask = torch.hstack([
        #     torch.ones(num_patches - num_mask),
        #     torch.zeros(num_mask),
        # ])
        
        # # random.shuffle(mask)
        # # mask = torch.randperm(mask)
        # idx = torch.randperm(num_patches)
        # mask = mask.view(-1)[idx].view((bbox_h, bbox_w))

        # img = img * mask.cuda()
        
        # img = self.en_net(img)

        # TNet
        input_img = nn.functional.interpolate(input_img, size=[H//4, W//4], mode='bilinear', align_corners=False)
        input_img = input_img.transpose(1, 3)
        input_img = input_img * self.std + self.mean
        input_img = input_img / 255.
        input_img = input_img.transpose(1, 3)
        j_out, data_conv4 = self.j_net(input_img, return_conv4=True)

        x = self.backbone(img, data_conv4)
        if self.with_neck:
            x_neck = self.neck(x)
            
        x_neck = list(x_neck)
        print('j_out and x_neck: ', j_out.shape, x_neck[0].shape)
        
        # a_out = get_A(input_img).cuda()
        a_out1 = get_A(input_img[0].unsqueeze(0)).cuda()
        a_out2 = get_A(input_img[1].unsqueeze(0)).cuda()
        a_out = torch.cat([a_out1, a_out2], 0)
        # a_out = gaussian_blur(input_img, self.gauss_kernel, padding=66)
        # j_out = self.J_head(x_neck[0])
        t_out = self.T_head(x_neck[0])
        # t_out = self.j_net(x_neck[0])
        # j_out = self.t_net(input_img)
        x_neck = tuple(x_neck)
        # print(j_out.shape, t_out.shape, a_out.shape)
        I_rec = j_out * t_out + (1 - t_out) * a_out
        loss_1 = self.mse_loss(I_rec, input_img)
        # save_image(j_out, './@fuhao/transfer/j_out.jpg', normalize=True)
        # save_image(t_out, './@fuhao/transfer/t_out.jpg', normalize=True)
        # save_image(a_out, './@fuhao/transfer/a_out.jpg', normalize=True)
        # save_image(I_rec, './@fuhao/transfer/I_rec.jpg', normalize=True)
        # save_image(input_img, './@fuhao/transfer/input_img.jpg', normalize=True)
        lam = np.random.beta(1, 1)
        # j_out_ = nn.functional.interpolate(j_out, size=[H, W], mode='bilinear', align_corners=False)
        j_out_ = j_out
        
        j_out_ = j_out_.transpose(1, 3)
        j_out_ = (j_out_ * 255. - self.mean) / self.std
        j_out_ = j_out_.transpose(1, 3)    
    
        
        input_mix = lam * input_img + (1 - lam) * j_out_

        # x = self.backbone(input_mix)
        # if self.with_neck:
        #     x = self.neck(x)

        # x = list(x)
        # # j_out_mix = self.J_head(x[0])
        # j_out_mix = self.j_net(x[0])
        
        j_out_mix = self.j_net(input_mix)
        
        # input_mix = lam * input_img + (1 - lam) * j_out
        # j_out_mix = self.j_net(input_mix)
        loss_2 = self.mse_loss(j_out_mix, j_out.detach())

        loss_3 = self.color_loss(j_out)

        total_loss = 1 * loss_1 + 1 * loss_2 + 0.01 * loss_3
        # total_loss = loss_1 + loss_2
        
        loss = total_loss
        
        # x = self.backbone(j_out)
        # if self.with_neck:
        #     x = self.neck(x)

        # x = list(x)
        # # x[0] = x[0] + en_img1
        # neck_feat0 = x[0].detach()
        # loss0 = self.mse_loss(en_img1, neck_feat0)
        # loss1 = self.mse_loss(img, input_img)
        # # mae_out = self.mae_decoder(x[0])
        # # input_img = nn.functional.interpolate(input_img, size=[x[0].shape[2], x[0].shape[3]], mode='bilinear', align_corners=False)
        # # loss_mae = self.mse_loss(mae_out, input_img)
        # x = tuple(x)
        
        # loss0 = self.mse_loss(bbox_feat[0], label_guide_x[0])
        # loss1 = self.mse_loss(bbox_feat[1], label_guide_x[1])
        # loss2 = self.mse_loss(bbox_feat[2], label_guide_x[2])
        # loss3 = self.mse_loss(bbox_feat[3], label_guide_x[3])
        # loss4 = self.mse_loss(bbox_feat[4], label_guide_x[4])
        
        # bbox_feat = tuple(bbox_feat)
        # label_guide_x = tuple(label_guide_x)
        
        # loss = 0.0
        # loss = loss0 + loss1
        
        # label_guide_x0 = self.fpn0(x[0])
        # label_guide_x1 = self.fpn1(x[1])
        # label_guide_x2 = self.fpn2(x[2])
        # label_guide_x3 = self.fpn3(x[3])
        # label_guide_x4 = self.fpn4(x[4])
        
        # label_guide_x0 = torch.abs(x[0]).mean(1)
        # label_guide_x1 = torch.abs(x[1]).mean(1)
        # label_guide_x2 = torch.abs(x[2]).mean(1)
        # label_guide_x3 = torch.abs(x[3]).mean(1)
        # label_mask0 =  torch.where(label_guide_x0>0.5, 1.0, 0.0)
        # x[0] = x[0] * label_mask0
        
        # mask0 = nn.functional.interpolate(label_mask0, size=[img.shape[2], img.shape[3]], mode='bilinear', align_corners=False)
        # img = img * mask0
        # save_image(img, '@fuhao/transfer/mask0_img.jpg', normalize=True)
        
        
        # t = label_guide_x0.data.cpu().numpy()
        # # fig = sns.heatmap(data=t[0], vmax=9., vmin=6.5, cmap='viridis') 
        # # fig = sns.heatmap(data=t[0], vmax=4., vmin=3.4, cmap='viridis')
        # fig = sns.heatmap(data=t[0][0], cmap='viridis')  
        # heatmap = fig.get_figure()
        # # heatmap.savefig(output_images_path + name[0], dpi=400)
        # heatmap.savefig('@fuhao/transfer/T.png', dpi=400)
        # heatmap.clear()        

        
        # label_0 = nn.functional.interpolate(Mask_fg, size=[x[0].shape[2], x[0].shape[3]], mode='bilinear', align_corners=False)
        # label_1 = nn.functional.interpolate(Mask_fg, size=[x[1].shape[2], x[1].shape[3]], mode='bilinear', align_corners=False)
        # label_2 = nn.functional.interpolate(Mask_fg, size=[x[2].shape[2], x[2].shape[3]], mode='bilinear', align_corners=False)
        # label_3 = nn.functional.interpolate(Mask_fg, size=[x[3].shape[2], x[3].shape[3]], mode='bilinear', align_corners=False)
        # label_4 = nn.functional.interpolate(Mask_fg, size=[x[4].shape[2], x[4].shape[3]], mode='bilinear', align_corners=False)
        
        
        # loss0 = self.mse_loss(label_guide_x0, label_0)
        # loss1 = self.mse_loss(label_guide_x1, label_1)
        # loss2 = self.mse_loss(label_guide_x2, label_2)
        # loss3 = self.mse_loss(label_guide_x3, label_3)
        # loss4 = self.mse_loss(label_guide_x4, label_4)
        
        # loss = loss0 + loss1 + loss2 + loss3 + loss4
        # loss = loss0

        # t = label_0.data.cpu().numpy()
        # fig = sns.heatmap(data=t[0][0], cmap='viridis')  
        # heatmap = fig.get_figure()
        # heatmap.savefig('@fuhao/transfer/label0.png', dpi=400)
        # heatmap.clear()      
        
        # t = label_mask0.data.cpu().numpy()
        # fig = sns.heatmap(data=t[0][0], cmap='viridis')  
        # heatmap = fig.get_figure()
        # heatmap.savefig('@fuhao/transfer/label_mask0.png', dpi=400)
        # heatmap.clear()       
        
        # label_guide_x = tuple([label_guide_x0, label_guide_x1, label_guide_x2, label_guide_x3, label_guide_x4])
        # label_guide_x = tuple([label_guide_x0, None, None, None, None])


        # mask_img = Mask_fg * img
        # save_image(mask_img, '@fuhao/transfer/mask_img.jpg', normalize=True)
        
        # label_guide_x0 = torch.where(label_guide_x0>0.2, 1.0, 0.)
        # label_guide_x0 = nn.functional.interpolate(label_guide_x0, size=[img.shape[2], img.shape[3]], mode='bilinear', align_corners=False)
        # label_guide_img = label_guide_x0 * img
        # save_image(label_guide_img, '@fuhao/transfer/label_guide_img.jpg', normalize=True)
        
        # x = tuple(x)
        bbox_feat = tuple([None, None, None, None, None])
        # return x, loss, label_guide_x        
        return x_neck, loss, bbox_feat

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
        
        losses.update({'loss_reconstruct_fg':loss_lable_guide})
        
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
        
        outs = self.bbox_head(x, label_guide_x)
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
class Custom_RetinaNet_v549_3(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(Custom_RetinaNet_v549_3, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained)
        
        self.mse_loss = torch.nn.MSELoss().cuda()
        
        # self.fpn0 = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.Sigmoid()
        # )

        # self.J_head = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 3, 3, 1, 1),
        #     nn.Tanh()
        # )
        # self.T_head = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 3, 3, 1, 1),
        #     nn.Tanh()
        # )

        # self.J_head = nn.Sequential(
        #     nn.Conv2d(256, 256, 3, 1, 1),
        #     nn.ConvTranspose2d(256 , 128, kernel_size=2, stride=2),
        #     # nn.Conv2d(256, 128, 3, 1, 1),
        #     nn.ReLU(),
        #     # nn.Conv2d(128, 3, 3, 1, 1),
        #     nn.ConvTranspose2d(128 , 64, kernel_size=2, stride=2),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(64 , 3, kernel_size=2, stride=2),
        #     nn.Tanh()
        # ) 
        
        self.j_net = JNet()
        self.t_net = TNet()
        
        self.J_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            # nn.ConvTranspose2d(256 , 128, kernel_size=2, stride=2),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 3, 3, 1, 1),
            # nn.ConvTranspose2d(128 , 64, kernel_size=2, stride=2),
            # nn.ReLU(),
            # nn.ConvTranspose2d(64 , 3, kernel_size=2, stride=2),
            # nn.Tanh()
            nn.Sigmoid()
        )
        
        self.T_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256 , 128, kernel_size=2, stride=2),
            # nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 3, 3, 1, 1),
            # nn.ConvTranspose2d(128 , 64, kernel_size=2, stride=2),
            # nn.ReLU(),
            # nn.ConvTranspose2d(64 , 3, kernel_size=2, stride=2),
            # nn.Tanh()
            nn.Sigmoid()
        )
        self.gauss_kernel = get_gaussian_kernel(size=133)
        self.mae_decoder = nn.Sequential(
            nn.Conv2d(256, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
        self.en_net_1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
        )
        
        self.en_net_2 = nn.Sequential(
            nn.ConvTranspose2d(256 , 128, kernel_size=2, stride=2),
            # nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(),
            # nn.Conv2d(128, 3, 3, 1, 1),
            nn.ConvTranspose2d(128 , 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64 , 3, kernel_size=2, stride=2),
            nn.Tanh()
        )
        
        self.ilu_net = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Tanh()
        )

        self.gamma_net = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Tanh()
        )
        
        self.up = nn.Sequential( 
            nn.ConvTranspose2d(3 , 3, kernel_size=2, stride=2),
            nn.ConvTranspose2d(3 , 3, kernel_size=2, stride=2)
        )
        self.mean= torch.tensor([123.675, 116.28, 103.53]).cuda()
        self.std= torch.tensor([58.395, 57.12, 57.375]).cuda()

        self.color_loss = ColorLoss()
        # self.labelnet = ResNet18_FPN(in_channels=1)
        
        # self.masknet = ResNet18_FPN(in_channels=3)

        # self.transfernet = ResNet18_FPN(in_channels=3)
        # self.transfernet.eval()
        # self.convtransfer = nn.Sequential(
        #     nn.Conv2d(256, 3, 3, 1, 1),
        #     nn.Sigmoid()
        # )

        weights_path = '@fuhao/exp/500Adaptive/540mask_retina/547_3fenzhi/checkpoints/RUIE_final.pth'
        print('+---------------------------------------------+')
        print('+ Loading weights into ' + 'uwnet' + ' state dict.. +')
        print('+ Weights path: ', weights_path)
        print('+---------------------------------------------+')
        model_dict = self.j_net.state_dict()
        
        num = 0
        for k in model_dict.keys():
            print('model =>', k)
            num += 1
        print('total num: ', num)
        num = 0

        pretrained_dict = torch.load(weights_path)
        # pretrained_dict = pretrained_dict['state_dict']

        for k in pretrained_dict.keys():
            print('checkpoint =>', k)
            num += 1
        print('total num: ', num)
        num = 0

        momo_dict = {}
        for k, v in pretrained_dict.items(): 
            # if k in model_dict.keys() and cfg.load_checkpoint['load_keys'] in k: # 注意要修改的地方
            if 'image_net' in k:
                k = k.replace('image_net.', '')
                # print(k)
            if k in model_dict.keys():
                if pretrained_dict['image_net.' + k].size() == model_dict[k].size():
                    momo_dict.update({k: v})

        # momo_dict = {k:v for 'module.'+k, v in pretrained_dict['state_dict'].items() if k in model_dict.keys() and pretrained_dict['state_dict'][k].shape() == model_dict[k].shape() }
        for k, v in momo_dict.items():
            print('model load => ', k)
            num += 1
        print('total num: ', num)

        model_dict.update(momo_dict)
        self.j_net.load_state_dict(model_dict)
        print('+---------------------------------------------+')
        print('+                 Finished！                  +')
        print('+---------------------------------------------+')
        
        weights_path = '@fuhao/exp/500Adaptive/540mask_retina/547_3fenzhi/checkpoints/RUIE_final.pth'
        print('+---------------------------------------------+')
        print('+ Loading weights into ' + 'uwnet' + ' state dict.. +')
        print('+ Weights path: ', weights_path)
        print('+---------------------------------------------+')
        model_dict = self.t_net.state_dict()
        
        num = 0
        for k in model_dict.keys():
            print('model =>', k)
            num += 1
        print('total num: ', num)
        num = 0

        pretrained_dict = torch.load(weights_path)
        # pretrained_dict = pretrained_dict['state_dict']

        for k in pretrained_dict.keys():
            print('checkpoint =>', k)
            num += 1
        print('total num: ', num)
        num = 0

        momo_dict = {}
        for k, v in pretrained_dict.items(): 
            # if k in model_dict.keys() and cfg.load_checkpoint['load_keys'] in k: # 注意要修改的地方
            if 'image_net' in k:
                k = k.replace('image_net.', '')
                # print(k)
            if k in model_dict.keys():
                if pretrained_dict['mask_net.' + k].size() == model_dict[k].size():
                    momo_dict.update({k: v})

        # momo_dict = {k:v for 'module.'+k, v in pretrained_dict['state_dict'].items() if k in model_dict.keys() and pretrained_dict['state_dict'][k].shape() == model_dict[k].shape() }
        for k, v in momo_dict.items():
            print('model load => ', k)
            num += 1
        print('total num: ', num)

        model_dict.update(momo_dict)
        self.t_net.load_state_dict(model_dict)
        print('+---------------------------------------------+')
        print('+                 Finished！                  +')
        print('+---------------------------------------------+')
        
    def test_extract_feat_label_guide(self, img):
        """Directly extract features from the backbone+neck."""

        # with torch.no_grad():
        #     self.transfernet.eval()
        #     label_x = self.transfernet(img)
        #     label_x = list(label_x)
        #     T = self.convtransfer(label_x[0])
        #     T = nn.functional.interpolate(T, size=[img.shape[2], img.shape[3]], mode='bilinear', align_corners=False)
        #     # img = img / (1 - T)
        #     img = img * T + img
        
        # bbox_h = img.shape[2]
        # bbox_w = img.shape[3]
        # mask_ratio = 0.75
        # num_patches = bbox_h * bbox_w
        # num_mask = int(mask_ratio * num_patches)

        # mask = torch.hstack([
        #     torch.ones(num_patches - num_mask),
        #     torch.zeros(num_mask),
        # ])
        # idx = torch.randperm(num_patches)
        # mask = mask.view(-1)[idx].view((bbox_h, bbox_w))
        
        # img = img * mask.cuda()
        
        # img = self.en_net(img)    
        
        
        # W = img.shape[2] // 4
        # H = img.shape[3] // 4
        # img = nn.functional.interpolate(img, size=[img.shape[2] // 4, img.shape[3] // 4], mode='bilinear', align_corners=True)
        # en_img1 = self.en_net_1(img)
        # en_img2 = self.en_net_2(en_img1)
        
        # # img = self.up(en_img2)   
        # img = en_img2     
        self.j_net.eval()
        input_img = img.detach()
        H = img.shape[2]
        W = img.shape[3]
        input_img = nn.functional.interpolate(input_img, size=[H//4, W//4], mode='bilinear', align_corners=False)
        input_img = input_img.transpose(1, 3)
        input_img = input_img * self.std + self.mean
        input_img = input_img / 255.
        input_img = input_img.transpose(1, 3)
        j_out, _ = self.j_net(input_img, return_conv4=True)
        
        j_mid, data_conv4 = self.j_net(j_out, return_conv4=True)

        x = self.backbone(img, data_conv4)
        # x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        
        # x = list(x)
        # j_out = self.J_head(x[0])
        
        # x = self.backbone(j_out)
        # if self.with_neck:
        #     x = self.neck(x)        
        # x = list(x)
        # x[0] = x[0] + en_img1
        # x = tuple(x)
        
        # x = list(x)
        # label_guide_x = self.masknet(img)
        # label_guide_x0 = self.fpn0(x[0])
        # label_guide_x1 = self.fpn1(x[1])
        # label_guide_x2 = self.fpn2(x[2])
        # label_guide_x3 = self.fpn3(x[3])
        # label_guide_x4 = self.fpn4(x[4])

        # label_mask0 =  torch.where(label_guide_x0>0.5, 1.0, 0.0)
        # x[0] = x[0] * label_mask0 
        # label_guide_x = tuple([label_guide_x0, label_guide_x1, label_guide_x2, label_guide_x3, label_guide_x4])
        # label_guide_x = tuple([label_guide_x0, None, None, None, None])
        label_guide_x = tuple([None, None, None, None, None])
        
        return x, label_guide_x
    
    
    def extract_feat_label_guide(self, img, img_metas, gt_bboxes):
        """Directly extract features from the backbone+neck."""

        # x = self.backbone(img)
        # if self.with_neck:
        #     x = self.neck(x)
            
        # bbox_feat = self.masknet(img)
            
        # for i in list(x):
        #     print(i.shape)
        # """
        
        """ Mask Gen """
        # N, C, H, W = img.shape
        # Mask_fg = torch.zeros([N, H, W]).cuda()
        # wmin,wmax,hmin,hmax = [],[],[],[]
        # for i in range(N):
        #     new_boxxes = torch.ones_like(gt_bboxes[i])
        #     new_boxxes[:, 0] = gt_bboxes[i][:, 0]/img_metas[i]['img_shape'][1]*W
        #     new_boxxes[:, 2] = gt_bboxes[i][:, 2]/img_metas[i]['img_shape'][1]*W
        #     new_boxxes[:, 1] = gt_bboxes[i][:, 1]/img_metas[i]['img_shape'][0]*H
        #     new_boxxes[:, 3] = gt_bboxes[i][:, 3]/img_metas[i]['img_shape'][0]*H

        #     wmin.append(torch.floor(new_boxxes[:, 0]).int())
        #     wmax.append(torch.ceil(new_boxxes[:, 2]).int())
        #     hmin.append(torch.floor(new_boxxes[:, 1]).int())
        #     hmax.append(torch.ceil(new_boxxes[:, 3]).int())

        #     for j in range(len(gt_bboxes[i])):
        #         # print(hmin[i][j], hmax[i][j], wmin[i][j], wmax[i][j])
        #         # bbox_h = hmax[i][j] - hmin[i][j] 
        #         # bbox_w = wmax[i][j] - wmin[i][j] 

        #         # mask_ratio = 0.75
        #         # num_patches = bbox_h * bbox_w
        #         # num_mask = int(mask_ratio * num_patches)

        #         # mask = torch.hstack([
        #         #     torch.ones(num_patches - num_mask),
        #         #     torch.zeros(num_mask),
        #         # ])
                
        #         # idx = torch.randperm(num_patches)
        #         # mask = mask.view(-1)[idx].view((bbox_h, bbox_w))

        #         Mask_fg[i][hmin[i][j]:hmax[i][j], wmin[i][j]:wmax[i][j]] = 1.0
                
        # Mask_fg = Mask_fg.unsqueeze(1)
        """ """
        
        # """
        # mask_input = img * Mask_fg
        # with torch.no_grad():
        #     self.transfernet.eval()
        #     img_t = img.detach()
        #     label_guide_x = self.transfernet(img_t)
        
        #     # bbox_feat = list(bbox_feat)
        #     label_guide_x = list(label_guide_x)

        #     T = self.convtransfer(label_guide_x[0])
        #     T = nn.functional.interpolate(T, size=[img.shape[2], img.shape[3]], mode='bilinear', align_corners=False)

        #     # img = img / (1 - T)
        #     img = img * T + img

        input_img = img.detach()
        H = img.shape[2]
        W = img.shape[3]
        
        # A = get_A(input_img)
        # x = (x - self.mean) / self.std
        
        # input_img = input_img.transpose(1, 3)
        
        # input_img = input_img * self.std + self.mean
        # gamma_factor = random.random()
        # if gamma_factor < 0.1:
        #     gamma_factor = 0.1
        # elif gamma_factor > 1.0:
        #     gamma_factor = 1.0
        # input_img = torch.pow(input_img, 0.5)
        # input_img = (input_img - self.mean) / self.std
        # input_img = input_img.transpose(1, 3)
        # input_img = input_img * Mask_fg
        
        
        # img = nn.functional.interpolate(img, size=[img.shape[2] // 4, img.shape[3] // 4], mode='bilinear', align_corners=True)
        # en_img1 = self.en_net_1(img)
        # en_img2 = self.en_net_2(en_img1)
        
        # img = self.up(en_img2)
        # img = en_img2
        # print(img.shape)
        # bbox_h = img.shape[2]
        # bbox_w = img.shape[3]
        # mask_ratio = 0.75
        # num_patches = bbox_h * bbox_w
        # num_mask = int(mask_ratio * num_patches)

        # mask = torch.hstack([
        #     torch.ones(num_patches - num_mask),
        #     torch.zeros(num_mask),
        # ])
        
        # # random.shuffle(mask)
        # # mask = torch.randperm(mask)
        # idx = torch.randperm(num_patches)
        # mask = mask.view(-1)[idx].view((bbox_h, bbox_w))

        # img = img * mask.cuda()
        
        # img = self.en_net(img)

        # JNet
        input_img = nn.functional.interpolate(input_img, size=[H//4, W//4], mode='bilinear', align_corners=False)
        input_img = input_img.transpose(1, 3)
        input_img = input_img * self.std + self.mean
        input_img = input_img / 255.
        input_img = input_img.transpose(1, 3)
        j_out, _ = self.j_net(input_img, return_conv4=True)
        j_mid, data_conv4 = self.j_net(j_out, return_conv4=True)
        
        #TNet
        t_out = self.t_net(input_img)

        x = self.backbone(img, data_conv4)
        if self.with_neck:
            x_neck = self.neck(x)
            
        x_neck = list(x_neck)
        # print('j_out and x_neck: ', j_out.shape, x_neck[0].shape)
        
        # a_out = get_A(input_img).cuda()
        a_out1 = get_A(input_img[0].unsqueeze(0)).cuda()
        a_out2 = get_A(input_img[1].unsqueeze(0)).cuda()
        a_out = torch.cat([a_out1, a_out2], 0)
        # a_out = gaussian_blur(input_img, self.gauss_kernel, padding=66)
        # j_out = self.J_head(x_neck[0])
        
        # j_out_det = self.T_head(x_neck[0])
        # loss_det = self.mse_loss(j_out.detach(), j_out_det)
        
        # t_out = self.j_net(x_neck[0])
        # j_out = self.t_net(input_img)
        x_neck = tuple(x_neck)
        # print(j_out.shape, t_out.shape, a_out.shape)
        I_rec = j_out * t_out + (1 - t_out) * a_out
        loss_1 = self.mse_loss(I_rec, input_img)
        # save_image(j_out, './@fuhao/transfer/j_out.jpg', normalize=True)
        # save_image(t_out, './@fuhao/transfer/t_out.jpg', normalize=True)
        # save_image(a_out, './@fuhao/transfer/a_out.jpg', normalize=True)
        # save_image(I_rec, './@fuhao/transfer/I_rec.jpg', normalize=True)
        # save_image(input_img, './@fuhao/transfer/input_img.jpg', normalize=True)
        lam = np.random.beta(1, 1)
        # j_out_ = nn.functional.interpolate(j_out, size=[H, W], mode='bilinear', align_corners=False)
        j_out_ = j_out
        
        j_out_ = j_out_.transpose(1, 3)
        j_out_ = (j_out_ * 255. - self.mean) / self.std
        j_out_ = j_out_.transpose(1, 3)    
    
        
        input_mix = lam * input_img + (1 - lam) * j_out_

        # x = self.backbone(input_mix)
        # if self.with_neck:
        #     x = self.neck(x)

        # x = list(x)
        # # j_out_mix = self.J_head(x[0])
        # j_out_mix = self.j_net(x[0])
        
        j_out_mix = self.j_net(input_mix)
        
        # input_mix = lam * input_img + (1 - lam) * j_out
        # j_out_mix = self.j_net(input_mix)
        loss_2 = self.mse_loss(j_out_mix, j_out.detach())

        loss_3 = self.color_loss(j_out)

        total_loss = 1 * loss_1 + 1 * loss_2 + 0.01 * loss_3
        # total_loss = loss_1 + loss_2
        
        loss = total_loss
        
        # x = self.backbone(j_out)
        # if self.with_neck:
        #     x = self.neck(x)

        # x = list(x)
        # # x[0] = x[0] + en_img1
        # neck_feat0 = x[0].detach()
        # loss0 = self.mse_loss(en_img1, neck_feat0)
        # loss1 = self.mse_loss(img, input_img)
        # # mae_out = self.mae_decoder(x[0])
        # # input_img = nn.functional.interpolate(input_img, size=[x[0].shape[2], x[0].shape[3]], mode='bilinear', align_corners=False)
        # # loss_mae = self.mse_loss(mae_out, input_img)
        # x = tuple(x)
        
        # loss0 = self.mse_loss(bbox_feat[0], label_guide_x[0])
        # loss1 = self.mse_loss(bbox_feat[1], label_guide_x[1])
        # loss2 = self.mse_loss(bbox_feat[2], label_guide_x[2])
        # loss3 = self.mse_loss(bbox_feat[3], label_guide_x[3])
        # loss4 = self.mse_loss(bbox_feat[4], label_guide_x[4])
        
        # bbox_feat = tuple(bbox_feat)
        # label_guide_x = tuple(label_guide_x)
        
        # loss = 0.0
        # loss = loss0 + loss1
        
        # label_guide_x0 = self.fpn0(x[0])
        # label_guide_x1 = self.fpn1(x[1])
        # label_guide_x2 = self.fpn2(x[2])
        # label_guide_x3 = self.fpn3(x[3])
        # label_guide_x4 = self.fpn4(x[4])
        
        # label_guide_x0 = torch.abs(x[0]).mean(1)
        # label_guide_x1 = torch.abs(x[1]).mean(1)
        # label_guide_x2 = torch.abs(x[2]).mean(1)
        # label_guide_x3 = torch.abs(x[3]).mean(1)
        # label_mask0 =  torch.where(label_guide_x0>0.5, 1.0, 0.0)
        # x[0] = x[0] * label_mask0
        
        # mask0 = nn.functional.interpolate(label_mask0, size=[img.shape[2], img.shape[3]], mode='bilinear', align_corners=False)
        # img = img * mask0
        # save_image(img, '@fuhao/transfer/mask0_img.jpg', normalize=True)
        
        
        # t = label_guide_x0.data.cpu().numpy()
        # # fig = sns.heatmap(data=t[0], vmax=9., vmin=6.5, cmap='viridis') 
        # # fig = sns.heatmap(data=t[0], vmax=4., vmin=3.4, cmap='viridis')
        # fig = sns.heatmap(data=t[0][0], cmap='viridis')  
        # heatmap = fig.get_figure()
        # # heatmap.savefig(output_images_path + name[0], dpi=400)
        # heatmap.savefig('@fuhao/transfer/T.png', dpi=400)
        # heatmap.clear()        

        
        # label_0 = nn.functional.interpolate(Mask_fg, size=[x[0].shape[2], x[0].shape[3]], mode='bilinear', align_corners=False)
        # label_1 = nn.functional.interpolate(Mask_fg, size=[x[1].shape[2], x[1].shape[3]], mode='bilinear', align_corners=False)
        # label_2 = nn.functional.interpolate(Mask_fg, size=[x[2].shape[2], x[2].shape[3]], mode='bilinear', align_corners=False)
        # label_3 = nn.functional.interpolate(Mask_fg, size=[x[3].shape[2], x[3].shape[3]], mode='bilinear', align_corners=False)
        # label_4 = nn.functional.interpolate(Mask_fg, size=[x[4].shape[2], x[4].shape[3]], mode='bilinear', align_corners=False)
        
        
        # loss0 = self.mse_loss(label_guide_x0, label_0)
        # loss1 = self.mse_loss(label_guide_x1, label_1)
        # loss2 = self.mse_loss(label_guide_x2, label_2)
        # loss3 = self.mse_loss(label_guide_x3, label_3)
        # loss4 = self.mse_loss(label_guide_x4, label_4)
        
        # loss = loss0 + loss1 + loss2 + loss3 + loss4
        # loss = loss0

        # t = label_0.data.cpu().numpy()
        # fig = sns.heatmap(data=t[0][0], cmap='viridis')  
        # heatmap = fig.get_figure()
        # heatmap.savefig('@fuhao/transfer/label0.png', dpi=400)
        # heatmap.clear()      
        
        # t = label_mask0.data.cpu().numpy()
        # fig = sns.heatmap(data=t[0][0], cmap='viridis')  
        # heatmap = fig.get_figure()
        # heatmap.savefig('@fuhao/transfer/label_mask0.png', dpi=400)
        # heatmap.clear()       
        
        # label_guide_x = tuple([label_guide_x0, label_guide_x1, label_guide_x2, label_guide_x3, label_guide_x4])
        # label_guide_x = tuple([label_guide_x0, None, None, None, None])


        # mask_img = Mask_fg * img
        # save_image(mask_img, '@fuhao/transfer/mask_img.jpg', normalize=True)
        
        # label_guide_x0 = torch.where(label_guide_x0>0.2, 1.0, 0.)
        # label_guide_x0 = nn.functional.interpolate(label_guide_x0, size=[img.shape[2], img.shape[3]], mode='bilinear', align_corners=False)
        # label_guide_img = label_guide_x0 * img
        # save_image(label_guide_img, '@fuhao/transfer/label_guide_img.jpg', normalize=True)
        
        # x = tuple(x)
        bbox_feat = tuple([None, None, None, None, None])
        # return x, loss, label_guide_x        
        return x_neck, loss, bbox_feat

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
        
        losses.update({'loss_reconstruct_fg':loss_lable_guide})
        
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
        
        outs = self.bbox_head(x, label_guide_x)
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
class Custom_RetinaNet_small_objects(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(Custom_RetinaNet_small_objects, self).__init__(backbone, neck, bbox_head, train_cfg,
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
        # x, loss_lable_guide, label_guide_x = self.extract_feat_label_guide(img, img_metas, gt_bboxes)
        print(img.shape)
        x = list(x)
        for i in x:
            print(i.shape)
        x = tuple(x)
        #  筛选gt_bboxes
        print(gt_bboxes, gt_labels)
        print(gt_bboxes[0].shape, gt_labels[0].shape)
        N, _, _, _ = img.shape
        new_gt_bboxes = []
        new_gt_labels = []
        for i in range(N):
            temp_bboxes = []
            temp_labels = []
        
            for j in range(len(gt_bboxes[i])):
                # temp_bboxes.append(gt_bboxes[i][j].cpu().numpy())
                # temp_labels.append(gt_labels[i][j].cpu().numpy())
                
                
                if gt_bboxes[i][j][2] - gt_bboxes[i][j][0] <= 32. :
                    temp_bboxes.append(gt_bboxes[i][j].cpu().numpy())
                    temp_labels.append(gt_labels[i][j].cpu().numpy())
                #     temp_bboxes.append(gt_bboxes[i][j])
                #     temp_labels.append(gt_labels[i][j])
                #     continue
                elif gt_bboxes[i][j][3] - gt_bboxes[i][j][1] <= 32. :
                    temp_bboxes.append(gt_bboxes[i][j].cpu().numpy())
                    temp_labels.append(gt_labels[i][j].cpu().numpy())
                #     temp_bboxes.append(gt_bboxes[i][j])
                #     temp_labels.append(gt_labels[i][j])
            
            if np.array(temp_bboxes).shape[0] == 0:
                temp_bboxes.append(gt_bboxes[i][0].cpu().numpy())
                temp_labels.append(gt_labels[i][0].cpu().numpy())
            # new_gt_bboxes.append(torch.tensor(temp_bboxes).cuda())
            # new_gt_labels.append(torch.tensor(temp_labels).cuda())
            # temp_bboxes = torch.tensor(temp_bboxes).cuda()
            # temp_labels = torch.tensor(temp_labels).cuda()
            temp_bboxes = torch.from_numpy(np.array(temp_bboxes)).cuda()
            temp_labels = torch.from_numpy(np.array(temp_labels)).cuda()
                
            
            new_gt_bboxes.append(temp_bboxes)
            new_gt_labels.append(temp_labels)
        # new_gt_bboxes = torch.tensor(new_gt_bboxes).cuda()
        # new_gt_labels = torch.tensor(new_gt_labels).cuda()
        print(new_gt_bboxes, new_gt_labels)
        # print(new_gt_bboxes.shape, new_gt_labels.shape)
        # -------------
        
        losses = self.bbox_head.forward_train(x, img_metas, new_gt_bboxes, # 改动了
                                              new_gt_labels, gt_bboxes_ignore,  # 改动了
                                            #   label_mask=label_guide_x
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
        label_guide_x = None
        x = self.extract_feat(img)
        # x, label_guide_x = self.test_extract_feat_label_guide(img)
        
        outs = self.bbox_head(x, label_guide_x)
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
