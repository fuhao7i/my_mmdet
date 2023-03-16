# from ..builder import DETECTORS
# from .two_stage import TwoStageDetector
from cProfile import label
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.two_stage import TwoStageDetector

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import cv2
import numpy as np
import random
import seaborn as sns
sns.set(font_scale=1.5)
from torchvision import transforms



import os
import torch
import numpy as np
from PIL import Image, ImageFilter
from torchvision.transforms import ToTensor

def normalize01(J):
    mmax = torch.max(torch.max(J, dim=3, keepdim=True).values, dim=2, keepdim=True).values
    mmin = torch.min(torch.min(J, dim=3, keepdim=True).values, dim=2, keepdim=True).values
    J = (J - mmin) / (mmax - mmin)   
    return J

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

def get_A(x):
    x_np = np.clip(torch_to_np(x), 0, 1)
    x_pil = np_to_pil(x_np)
    h, w = x_pil.size
    windows = (h + w) / 2
    A = x_pil.filter(ImageFilter.GaussianBlur(windows))
    A = ToTensor()(A)
    return A.unsqueeze(0)



@DETECTORS.register_module()
class CascadeRCNN_Homology(TwoStageDetector):
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
        super(CascadeRCNN_Homology, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
        
        self.gauss_kernel1 = self.get_gaussian_kernel(size=3).cuda()
        self.gauss_kernel2 = self.get_gaussian_kernel(size=5).cuda()
        self.gauss_kernel3 = self.get_gaussian_kernel(size=7).cuda()
        self.gauss_kernel4 = self.get_gaussian_kernel(size=9).cuda()
        self.gauss_kernel5 = self.get_gaussian_kernel(size=11).cuda()
        
        self.gauss_kernel_list = [[self.gauss_kernel1, 1], [self.gauss_kernel2, 2], [self.gauss_kernel3, 3], [self.gauss_kernel4, 4], [self.gauss_kernel5, 5]]
        
        self.mse_loss = torch.nn.MSELoss().cuda()
        
        self.fpn0 = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.fpn1 = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.fpn2 = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.fpn3 = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )
        

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
        return super(CascadeRCNN_Homology, self).show_result(data, result, **kwargs)

    
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

    
    def get_gaussian_kernel(self, size=3): # 获取高斯kerner 并转为tensor ，size 可以改变模糊程度
        kernel = cv2.getGaussianKernel(size, 0).dot(cv2.getGaussianKernel(size, 0).T)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = torch.nn.Parameter(data=kernel, requires_grad=False)
        return kernel

    def get_sobel_kernel(self, im):
        sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')  
        sobel_kernel = sobel_kernel.reshape((1, 1, 3, 3))
        weight = Variable(torch.from_numpy(sobel_kernel))
        return weight 

    def gaussian_blur(self, x, k, stride=1, padding=0):
        res = []
        x = F.pad(x, (padding, padding, padding, padding), mode='constant', value=0)
        for xx in x.split(1, 1):
            res.append(F.conv2d(xx, k, stride=stride, padding=0))
        return torch.cat(res, 1)
    
    def test_extract_feat_label_guide(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        
        x = list(x)
        
        label_guide_x0 = self.fpn0(x[0])
        label_guide_x1 = self.fpn0(x[1])
        label_guide_x2 = self.fpn0(x[2])
        label_guide_x3 = self.fpn0(x[3])

        label_mask0 =  torch.where(label_guide_x0>0.5, 1.0, 0.0)
        x[0] = x[0] * label_mask0 
        
        return x
    
    
    def extract_feat_label_guide(self, img, img_metas, gt_bboxes):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
            
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

            area = 1.0/(hmax[i].view(1,-1)+1-hmin[i].view(1,-1))/(wmax[i].view(1,-1)+1-wmin[i].view(1,-1))
    

            for j in range(len(gt_bboxes[i])):
                Mask_fg[i][max(hmin[i][j]-20, 0):min(hmax[i][j]+20, H), max(wmin[i][j]-20, 0):min(wmax[i][j]+20, W)] = 1.0
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
        # heatmap.savefig('@fuhao/transfer/T.png', dpi=400)
        # heatmap.clear()
        
        # Mask_fg = Mask_fg.unsqueeze(1)
        # mask_img = Mask_fg * img
        # save_image(mask_img, '@fuhao/transfer/mask_img.jpg', normalize=True)
        # Mask_fg = Mask_fg.squeeze()
        
        x = list(x)

        Mask_fg = Mask_fg.unsqueeze(1)
        
        label_guide_x0 = self.fpn0(x[0])
        label_guide_x1 = self.fpn0(x[1])
        label_guide_x2 = self.fpn0(x[2])
        label_guide_x3 = self.fpn0(x[3])
        # label_guide_x0 = torch.abs(x[0]).mean(1)
        # label_guide_x1 = torch.abs(x[1]).mean(1)
        # label_guide_x2 = torch.abs(x[2]).mean(1)
        # label_guide_x3 = torch.abs(x[3]).mean(1)
        label_mask0 =  torch.where(label_guide_x0>0.5, 1.0, 0.0)
        x[0] = x[0] * label_mask0
        
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

        
        label_0 = nn.functional.interpolate(Mask_fg, size=[x[0].shape[2], x[0].shape[3]], mode='bilinear', align_corners=False)
        label_1 = nn.functional.interpolate(Mask_fg, size=[x[1].shape[2], x[1].shape[3]], mode='bilinear', align_corners=False)
        label_2 = nn.functional.interpolate(Mask_fg, size=[x[2].shape[2], x[2].shape[3]], mode='bilinear', align_corners=False)
        label_3 = nn.functional.interpolate(Mask_fg, size=[x[3].shape[2], x[3].shape[3]], mode='bilinear', align_corners=False)
        
        loss0 = self.mse_loss(label_guide_x0, label_0)
        loss1 = self.mse_loss(label_guide_x1, label_1)
        loss2 = self.mse_loss(label_guide_x2, label_2)
        loss3 = self.mse_loss(label_guide_x3, label_3)
        
        loss = loss0 + loss1 + loss2 + loss3

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

        
        return x, loss
    

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
            
        return x

    def extract_feat_transmission(self, img):
        self.mean= torch.tensor([123.675, 116.28, 103.53]).cuda()
        self.std= torch.tensor([58.395, 57.12, 57.375]).cuda()
        
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
            
        input = img.detach()

        input = normalize01(input)
        
        # x = input * 255.
        # x = x.transpose(1, 3)
        # x = (x - self.mean) / self.std
        # input1 = x.transpose(1, 3)
        # input1 = normalize01(input1)
        
        # x = input1 * 255.
        # x = x.transpose(1, 3)
        # x = (x - self.mean) / self.std
        # input2 = x.transpose(1, 3)
        # input2 = normalize01(input2)
        
        # x = input2 * 255.
        # x = x.transpose(1, 3)
        # x = (x - self.mean) / self.std
        # input3 = x.transpose(1, 3)
        # input3 = normalize01(input3)
        
        # x = input3 * 255.
        # x = x.transpose(1, 3)
        # x = (x - self.mean) / self.std
        # input4 = x.transpose(1, 3)
        # input4 = normalize01(input4)

        # save_image(input, '@fuhao/transfer/input.jpg', normalize=True)
        # save_image(input1, '@fuhao/transfer/input1.jpg', normalize=True)
        # save_image(input2, '@fuhao/transfer/input2.jpg', normalize=True)
        # save_image(input3, '@fuhao/transfer/input3.jpg', normalize=True)
        # save_image(input4, '@fuhao/transfer/input4.jpg', normalize=True)


        A = get_A(input).cuda()
        _, _, h, w = input.shape
        
        x = list(x)
        T = x[0].detach()
        T = nn.functional.interpolate(T, size=[h, w], mode='bilinear', align_corners=False)
        T = T.pow(2).mean(1)
        print(T.shape)
        mmax = torch.max(torch.max(T, dim=2, keepdim=True).values, dim=1, keepdim=True).values
        mmin = torch.min(torch.min(T, dim=2, keepdim=True).values, dim=1, keepdim=True).values
        T = (T - mmin) / (mmax - mmin)    
          
        t = T.data.cpu().numpy()
        # fig = sns.heatmap(data=t[0], vmax=9., vmin=6.5, cmap='viridis') 
        # fig = sns.heatmap(data=t[0], vmax=4., vmin=3.4, cmap='viridis')
        fig = sns.heatmap(data=t[0], cmap='viridis')  
        heatmap = fig.get_figure()
        # heatmap.savefig(output_images_path + name[0], dpi=400)
        heatmap.savefig('@fuhao/transfer/T.png', dpi=400)
        heatmap.clear()
        
        J = (input - (T) * A) / (1 - T)
        # J = torch.log(input * 255. + 1.) - torch.log( (1 - T) * A * 255. + 1.)
        # J = torch.log(input * 255. + 1.) - 0.5 * torch.log((1 - T) * A * 255. + 1.)
        # J = input / (T * A + 0.00001)
        # mean = torch.mean(torch.mean(J, dim=3, keepdim=True), dim=2, keepdim=True)
        # std = torch.std(torch.std(J, dim=3, keepdim=True), dim=2, keepdim=True)
        # J = (J - mean) / std 
        # J = torch.clamp(J, 0, 1)
        mmax = torch.max(torch.max(J, dim=3, keepdim=True).values, dim=2, keepdim=True).values
        mmin = torch.min(torch.min(J, dim=3, keepdim=True).values, dim=2, keepdim=True).values
        J = (J - mmin) / (mmax - mmin)   
        # J = torch.pow(J, 2.)
        
        save_image(img, '@fuhao/transfer/img.jpg', normalize=True)
        save_image(input, '@fuhao/transfer/input.jpg', normalize=True)
        save_image(input1, '@fuhao/transfer/input1.jpg', normalize=True)
        save_image(input2, '@fuhao/transfer/input2.jpg', normalize=True)
        save_image(input3, '@fuhao/transfer/input3.jpg', normalize=True)
        save_image(input4, '@fuhao/transfer/input4.jpg', normalize=True)
        save_image(J, '@fuhao/transfer/J.jpg', normalize=True)
        save_image(A, '@fuhao/transfer/A.jpg', normalize=True)
        
        x = tuple(x)    
        return x

    def extract_feat_custom(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        backbone_x = x
        if self.with_neck:
            x = self.neck(x)
            
        """Homology"""
        ksize = random.randint(0, 4)
        
        _, _, h, w = img.shape
        img = transforms.Resize([h//4, w//4])(img)
        blurred_img = self.gaussian_blur(img, self.gauss_kernel_list[ksize][0], padding=self.gauss_kernel_list[ksize][1])
        blurred_img = transforms.Resize([h, w])(blurred_img)
        blurred = blurred_img.detach()
        # del gauss_kernel

        blurred_x = self.backbone(blurred)
        
        # save_image(img, '@fuhao/transfer/input.jpg', normalize=True)
        # save_image(blurred_img, '@fuhao/transfer/blurred_img.jpg', normalize=True)
   
        # stop_grad_x = x.detach()
        backbone_x = list(backbone_x)
        blurred_x  = list(blurred_x)
        
        homology_loss = 0.0
        for i in range(4):
            stop_grad_x = backbone_x[i].detach()
            # homology_loss += torch.norm(blurred_x[i] - stop_grad_x, 2)
            homology_loss += self.mse_loss(blurred_x[i], stop_grad_x)
            
            # t = stop_grad_x.pow(2).mean(1)
            # t = t.data.cpu().numpy()
            # # fig = sns.heatmap(data=t[0], vmax=9., vmin=6.5, cmap='viridis') 
            # # fig = sns.heatmap(data=t[0], vmax=4., vmin=3.4, cmap='viridis')
            # fig = sns.heatmap(data=t[0], cmap='viridis')  
            # heatmap = fig.get_figure()
            # # heatmap.savefig(output_images_path + name[0], dpi=400)
            # heatmap.savefig('@fuhao/transfer/stop_grad_x_%d.png'%i, dpi=400)
            # heatmap.clear()

            # t = blurred_x[i].pow(2).mean(1)
            # t = t.data.cpu().numpy()
            # # fig = sns.heatmap(data=t[0], vmax=9., vmin=6.5, cmap='viridis') 
            # # fig = sns.heatmap(data=t[0], vmax=4., vmin=3.4, cmap='viridis')
            # fig = sns.heatmap(data=t[0], cmap='viridis')  
            # heatmap = fig.get_figure()
            # # heatmap.savefig(output_images_path + name[0], dpi=400)
            # heatmap.savefig('@fuhao/transfer/blurred_x_%d.png'%i, dpi=400)
            # heatmap.clear()
        
        
        # x = list(x)
        # for i in range(4):
        #     img = x[i]
        #     # lin = torch.mean(img, dim=3, keepdim=True)
        #     # print(lin.shape)
        #     mean = torch.mean(torch.mean(img, dim=3, keepdim=True), dim=2, keepdim=True)
        #     std = torch.std(torch.std(img, dim=3, keepdim=True), dim=2, keepdim=True)
        #     x[i] = (img - mean) / std 
        # x = tuple(x)
        
        
        
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
            
        return x, homology_loss

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
        # x = self.extract_feat_transmission(img)
        x, loss_lable_guide = self.extract_feat_label_guide(img, img_metas, gt_bboxes)

        losses = dict()
        # losses.update({'homology_loss':homology_loss})
        losses.update({'loss_label_guide':loss_lable_guide})

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

        # x = self.extract_feat(img)
        x = self.test_extract_feat_label_guide(img)

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
