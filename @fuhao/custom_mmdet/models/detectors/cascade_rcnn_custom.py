# from ..builder import DETECTORS
# from .two_stage import TwoStageDetector
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.two_stage import TwoStageDetector

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import seaborn as sns
sns.set(font_scale=1.5)

@DETECTORS.register_module()
class CascadeRCNN_Custom(TwoStageDetector):
    r"""Implementation of `Cascade R-CNN: Delving into High Quality Object
    Detection <https://arxiv.org/abs/1906.09756>`_"""

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(CascadeRCNN_Custom, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

    def show_result(self, data, result, **kwargs):
        """Show prediction results of the detector.

        Args:
            data (str or np.ndarray): Image filename or loaded image.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).

        Returns:
            np.ndarray: The image with bboxes drawn on it.
        """
        if self.with_mask:
            ms_bbox_result, ms_segm_result = result
            if isinstance(ms_bbox_result, dict):
                result = (ms_bbox_result['ensemble'],
                          ms_segm_result['ensemble'])
        else:
            if isinstance(result, dict):
                result = result['ensemble']
        return super(CascadeRCNN_Custom, self).show_result(data, result, **kwargs)

    
    def gradient(self, img):
        height = img.size(2)
        width = img.size(3)
        # gradient_h = (img[:,:,2:,:]-img[:,:,:height-2,:]).abs()
        # gradient_w = (img[:, :, :, 2:] - img[:, :, :, :width-2]).abs()
        # gradient_h = F.pad(gradient_h, [0, 0, 1, 1], 'replicate')
        # gradient_w = F.pad(gradient_w, [1, 1, 0, 0], 'replicate')
        gradient2_h = (img[:,:,4:,:]-img[:,:,:height-4,:]).abs()
        gradient2_w = (img[:, :, :, 4:] - img[:, :, :, :width-4]).abs()
        gradient2_h = F.pad(gradient2_h, [0, 0, 2, 2], 'replicate')
        gradient2_w = F.pad(gradient2_w, [2, 2, 0, 0], 'replicate')
        return gradient2_h, gradient2_w

        

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
            
        x = list(x)
        for i in range(4):
            img = x[i]
            # lin = torch.mean(img, dim=3, keepdim=True)
            # print(lin.shape)
            mean = torch.mean(torch.mean(img, dim=3, keepdim=True), dim=2, keepdim=True)
            std = torch.std(torch.std(img, dim=3, keepdim=True), dim=2, keepdim=True)
            x[i] = (img - mean) / std 
        x = tuple(x)
        # for i in x:
            # i = i.detach().pow(2).mean(1)
            # i = i.detach().mean(axis=1)
            
            

            # t = i.data.cpu().numpy()
            # fig = sns.heatmap(data=t[0], cmap='viridis')  
            # heatmap = fig.get_figure()
            # heatmap.savefig('@fuhao/transfer/attention.jpg', dpi=400)
            # heatmap.clear()
            
            # # save_image(i, '@fuhao/transfer/attention.jpg', normalize=True)
            # i = i.unsqueeze(1)
            # gradient_h, gradient_w = self.gradient(i)
            # gradient_h = gradient_h.squeeze()
            # gradient_w = gradient_w.squeeze()
            
            # t = gradient_h.data.cpu().numpy()
            # fig = sns.heatmap(data=t[0], cmap='viridis')  
            # heatmap = fig.get_figure()
            # heatmap.savefig('@fuhao/transfer/g_h.jpg', dpi=400)
            # heatmap.clear()

            # t = gradient_w.data.cpu().numpy()
            # fig = sns.heatmap(data=t[0], cmap='viridis')  
            # heatmap = fig.get_figure()
            # heatmap.savefig('@fuhao/transfer/g_w.jpg', dpi=400)
            # heatmap.clear()
            # # save_image(gradient_h, '@fuhao/transfer/g_h.jpg', normalize=True)
            # # save_image(gradient_w, '@fuhao/transfer/g_w.jpg', normalize=True)
            # print('saved!')
            # break
            
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        x = self.extract_feat(img)

        # get origin input shape to onnx dynamic input shape
        if torch.onnx.is_in_onnx_export():
            img_shape = torch._shape_as_tensor(img)[2:]
            img_metas[0]['img_shape_for_onnx'] = img_shape

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)
