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

# ----------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16

import numpy as np

class L_TV(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(L_TV,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]

        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)

        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()
        #self.eps = 1e-6
        self.eps = 0

    def forward(self, x ):

        #b,c,h,w = x.shape

        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2) + self.eps,0.5)


        return k

class L_spa8(nn.Module):
    def __init__(self, patch_size = 4):
        super(L_spa8, self).__init__()
        device = 'cuda:0'
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        # Build conv kernels
        kernel_left = torch.FloatTensor( [[0,0,0],[-1,1,0],[0,0,0]]).to(device).unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).to(device).unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor( [[0,-1,0],[0,1, 0 ],[0,0,0]]).to(device).unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).to(device).unsqueeze(0).unsqueeze(0)
        kernel_upleft = torch.FloatTensor( [[-1,0,0],[0,1,0],[0,0,0]]).to(device).unsqueeze(0).unsqueeze(0)
        kernel_upright = torch.FloatTensor( [[0,0,-1],[0,1,0],[0,0,0]]).to(device).unsqueeze(0).unsqueeze(0)
        kernel_loleft = torch.FloatTensor( [[0,0,0],[0,1,0],[-1,0,0]]).to(device).unsqueeze(0).unsqueeze(0)
        kernel_loright = torch.FloatTensor( [[0,0,0],[0,1,0],[0,0,-1]]).to(device).unsqueeze(0).unsqueeze(0)

        # convert to parameters
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.weight_upleft = nn.Parameter(data=kernel_upleft, requires_grad=False)
        self.weight_upright = nn.Parameter(data=kernel_upright, requires_grad=False)
        self.weight_loleft = nn.Parameter(data=kernel_loleft, requires_grad=False)
        self.weight_loright = nn.Parameter(data=kernel_loright, requires_grad=False)

        # pooling layer
        self.pool = nn.AvgPool2d(patch_size) # default is 4

    def forward(self, org , enhance ):
        #b,c,h,w = org.shape

        org_mean = torch.mean(org,1,keepdim=True)
        enhance_mean = torch.mean(enhance,1,keepdim=True)

        org_pool =  self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        #weight_diff =torch.max(torch.FloatTensor([1]).to(device) + 10000*torch.min(org_pool - torch.FloatTensor([0.3]).to(device),torch.FloatTensor([0]).to(device)),torch.FloatTensor([0.5]).to(device))
        #E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).to(device)) ,enhance_pool-org_pool)


        # Original output
        D_org_letf = F.conv2d(org_pool , self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool , self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool , self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool , self.weight_down, padding=1)
        D_org_upleft = F.conv2d(org_pool , self.weight_upleft , padding=1)
        D_org_upright = F.conv2d(org_pool , self.weight_upright, padding=1)
        D_org_loleft = F.conv2d(org_pool , self.weight_loleft, padding=1)
        D_org_loright = F.conv2d(org_pool , self.weight_loright, padding=1)


        # Enhanced output
        D_enhance_letf = F.conv2d(enhance_pool , self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool , self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool , self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool , self.weight_down, padding=1)
        D_enhance_upleft = F.conv2d(enhance_pool, self.weight_upleft, padding=1)
        D_enhance_upright = F.conv2d(enhance_pool, self.weight_upright, padding=1)
        D_enhance_loleft = F.conv2d(enhance_pool, self.weight_loleft, padding=1)
        D_enhance_loright = F.conv2d(enhance_pool, self.weight_loright, padding=1)

        # Difference
        D_left = torch.pow(D_org_letf - D_enhance_letf,2)
        D_right = torch.pow(D_org_right - D_enhance_right,2)
        D_up = torch.pow(D_org_up - D_enhance_up,2)
        D_down = torch.pow(D_org_down - D_enhance_down,2)
        D_upleft = torch.pow(D_org_upleft - D_enhance_upleft,2)
        D_upright = torch.pow(D_org_upright - D_enhance_upright,2)
        D_loleft = torch.pow(D_org_loleft - D_enhance_loleft,2)
        D_loright = torch.pow(D_org_loright - D_enhance_loright,2)

        # Total difference
        E = (D_left + D_right + D_up +D_down) + 0.5 * (D_upleft + D_upright + D_loleft + D_loright)

        # E = 25*(D_left + D_right + D_up +D_down)

        return E

# ----------------------------------------------
# D
from torch.autograd import Variable
from torchvision import models
import torch
import torch.nn as nn

#使用VGG16的特征提取层+新的全连接层组成新的网络
class MyVgg16(nn.Module):
    def __init__(self):
        super(MyVgg16,self).__init__()
        
        vgg16 = models.vgg16(pretrained=True)
        #获取VGG16的特征提取层
        vgg_f = vgg16.features
        #将vgg16的特征提取层参数冻结，不对其进行更新
        # for param in vgg.parameters():
        #     param.requires_grad_(False)
        
        #预训练的Vgg16的特征提取层
        self.vgg = vgg_f
        #添加新的全连接层
        # self.classifier = nn.Sequential(
        #     nn.Linear(25088,512),
        #     nn.ReLU(),
        #     nn.Dropout(p = 0.5),
        #     nn.Linear(512,256),
        #     nn.ReLU(),
        #     nn.Dropout(p = 0.5),
        #     nn.Linear(256,10),

        # )
        self.classifier = nn.Sequential(
            # nn.Linear(2048, 512),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        self.adaptpool = nn.AdaptiveAvgPool2d((1, 1))

    #定义网络的前向传播
    def forward(self,x):
        x = self.vgg(x)
        x = self.adaptpool(x)
        x = x.view(x.size(0),-1)
        # print(x.shape)
        output = self.classifier(x)
        return output
# ----------------------------------------------


import torch
import torch.nn as nn

class RRDNet(nn.Module):
    def __init__(self):
        super(RRDNet, self).__init__()

        self.reflectance_net = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1)
        )

    def forward(self, input):
        reflectance = torch.sigmoid(self.reflectance_net(input))

        return reflectance
    

    
class LargeKernelNet(nn.Module):
    def __init__(self, kernel=7):
        super(LargeKernelNet, self).__init__()
        
        self.largekernelnet = nn.Sequential(
            nn.Conv2d(3, 3, kernel, 1, kernel // 2)
        )
    def forward(self, input):
        x = torch.sigmoid(self.largekernelnet(input))
        return x

import os
 
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
 
np.random.seed(0)


class MyDataset(Dataset):
	
	def __init__(self, root, size=229, ):
		"""
		Initialize the data producer
		"""
		self._root = root
		self._size = size
		self._num_image = len(os.listdir(root))
		self._img_name = os.listdir(root)

		self.transform_A = transforms.Compose([
			transforms.Resize([112, 112]),
			transforms.ToTensor()
		])
  
        # self.transform_A = transforms.Compose([transforms.Resize([112, 112]), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	def __len__(self):
		return self._num_image
		
	def __getitem__(self, index):
		img = Image.open(os.path.join(self._root, self._img_name[index])).convert('RGB')
		img = self.transform_A(img)
		
		return img


@DETECTORS.register_module()
class RetinaNet_GAN(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 bbox_head_1=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RetinaNet_GAN, self).__init__(backbone, neck, bbox_head, bbox_head_1, train_cfg,
                                        test_cfg, pretrained)
        
        self.voc_dataset = MyDataset('../datasets/coco2017/patched/')
        self.length = len(self.voc_dataset)
        self.net_g = RRDNet()
        self.net_d = MyVgg16()
        self.spa_loss = L_spa8()
        self.color_loss = L_color()
        self.tv_loss = L_TV()
        self.Tensor = torch.cuda.FloatTensor
        self.adversarial_loss = torch.nn.BCELoss()
        
        self.net_g.cuda()
        self.net_d.cuda()
        self.adversarial_loss.cuda()
        self.spa_loss.cuda()
        self.color_loss.cuda()
        self.tv_loss.cuda()

    def extract_feat_test(self, img):
        """Directly extract features from the backbone+neck."""
        
        img = self.net_g(img)
                
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)

        return x

    def extract_feat(self, img, img_metas, gt_bboxes):
        """Directly extract features from the backbone+neck."""
        ori_img = img
        img = self.net_g(ori_img)

        N, C, H, W = img.shape

        Mask_fg = torch.zeros([N, H, W]).cuda()
        Mask_bg = torch.ones([N, H, W]).cuda()
        wmin,wmax,hmin,hmax = [],[],[],[]
        patch_gen = []
        gan_img = nn.functional.interpolate(img, size=[64, 64], mode='bilinear', align_corners=False)
        patch_gen.append(gan_img)
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
                patch = img[i][ :, hmin[i][j]:hmax[i][j], wmin[i][j]:wmax[i][j]]
                # patch = patch.squeeze()
                patch = patch.unsqueeze(0)
                # print(patch.shape)
                patch = nn.functional.interpolate(patch, size=[64, 64], mode='bilinear', align_corners=False)
                patch_gen.append(patch)
                
                
                
                Mask_fg[i][max(hmin[i][j], 0):min(hmax[i][j], H), max(wmin[i][j], 0):min(wmax[i][j], W)] = 1.0
            Mask_bg[i] = torch.where(Mask_fg[i]>0, 0, 1)
        
        g_loss = None
        d_loss = None
        if len(patch_gen) != 0:
            patch_gen = torch.cat(patch_gen, 0)
            save_image(patch_gen[0], '@fuhao/transfer/patch_gen.jpg', normalize=True)
            save_image(img[0], '@fuhao/transfer/img_gen.jpg', normalize=True)
            valid = Variable(self.Tensor(patch_gen.size(0), 1).fill_(1.0), requires_grad=False)
            fake  = Variable(self.Tensor(patch_gen.size(0), 1).fill_(0.0), requires_grad=False)
            # Loss measures generator's ability to fool the discriminator
            g_loss = self.adversarial_loss(self.net_d(patch_gen), valid)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            real_imgs = []
            for i in range(len(patch_gen)):
                index = torch.rand(1)

                index = int(index * self.length)
                pic1 = self.voc_dataset.__getitem__(index)
                pic1 = pic1.unsqueeze(0).cuda()
                real_imgs.append(pic1)
                
            real_imgs = torch.cat(real_imgs, 0)
            save_image(real_imgs[0], '@fuhao/transfer/real_img.jpg', normalize=True)
            # real_imgs.cuda()
            # Measure discriminator's ability to classify real from generated samples
            # print(type(real_imgs), type(valid))
            real_loss = self.adversarial_loss(self.net_d(real_imgs), valid)
            fake_loss = self.adversarial_loss(self.net_d(patch_gen.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
        
        
        spa_loss = self.spa_loss(ori_img, img) * 20
        color_loss = self.color_loss(img) 
        color_loss = 5 * torch.mean(color_loss) * 1000
        # x = self.backbone(img)
        # if self.with_neck:
        #     x = self.neck(x)
        x = img
        return x, g_loss, d_loss, spa_loss, color_loss

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
        
        x, g_loss, d_loss, spa_loss, color_loss = self.extract_feat(img, img_metas, gt_bboxes)
        
        losses = {}
        # losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
        #                                       gt_labels, gt_bboxes_ignore, 
        #                                       label_mask=None
        #                                         )
        if g_loss != None:
            losses.update({'g_loss':g_loss})
            losses.update({'d_loss':d_loss})
            losses.update({'spa_loss':spa_loss})
            losses.update({'color_loss':color_loss})
        else:
            losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                                gt_labels, gt_bboxes_ignore, 
                                                label_mask=None
                                                    )
        
        return losses

    def simple_test(self, img, img_metas, rescale=False):

        x = self.extract_feat_test(img)
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
class RetinaNet_CycleGAN(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 bbox_head_1=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RetinaNet_CycleGAN, self).__init__(backbone, neck, bbox_head, bbox_head_1, train_cfg,
                                        test_cfg, pretrained)
        
        self.voc_dataset = MyDataset('../datasets/coco2017/patched/')
        self.length = len(self.voc_dataset)
        # self.net_g = RRDNet()
        # self.net_d = MyVgg16()
        self.netG_A = RRDNet()
        self.netG_A.cuda()
        self.netG_B = RRDNet()
        self.netG_B.cuda()
        
        self.netD_A = MyVgg16()
        self.netD_A.cuda()
        self.netD_B = MyVgg16()
        self.netD_B.cuda()
        
        self.spa_loss = L_spa8()
        self.color_loss = L_color()
        self.tv_loss = L_TV()
        self.Tensor = torch.cuda.FloatTensor
        self.adversarial_loss = torch.nn.BCELoss()
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionCycle.cuda()
        self.criterionIdt = torch.nn.L1Loss()
        self.criterionIdt.cuda()
        
        # self.net_g.cuda()
        # self.net_d.cuda()
        self.adversarial_loss.cuda()
        self.spa_loss.cuda()
        self.color_loss.cuda()
        self.tv_loss.cuda()
        
        self.iter = 0

    def extract_feat_test(self, img):
        """Directly extract features from the backbone+neck."""
        
        img = self.netG_A(img)
                
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)

        return x

    def extract_feat(self, img, img_metas, gt_bboxes):
        """Directly extract features from the backbone+neck."""
        ori_img = img
        img = self.netG_A(ori_img)

        N, C, H, W = img.shape

        Mask_fg = torch.zeros([N, H, W]).cuda()
        Mask_bg = torch.ones([N, H, W]).cuda()
        wmin,wmax,hmin,hmax = [],[],[],[]
        patch_gen = []
        real_a = []
        # gan_img = nn.functional.interpolate(img, size=[64, 64], mode='bilinear', align_corners=False)
        # patch_gen.append(gan_img)
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
                patch = img[i][ :, hmin[i][j]:hmax[i][j], wmin[i][j]:wmax[i][j]]
                patch = patch.unsqueeze(0)
                patch = nn.functional.interpolate(patch, size=[64, 64], mode='bilinear', align_corners=False)
                patch_gen.append(patch)

                patch = ori_img[i][ :, hmin[i][j]:hmax[i][j], wmin[i][j]:wmax[i][j]]
                patch = patch.unsqueeze(0)
                patch = nn.functional.interpolate(patch, size=[64, 64], mode='bilinear', align_corners=False)
                real_a.append(patch)
                
                
                
                Mask_fg[i][max(hmin[i][j], 0):min(hmax[i][j], H), max(wmin[i][j], 0):min(wmax[i][j], W)] = 1.0
            Mask_bg[i] = torch.where(Mask_fg[i]>0, 0, 1)
        
        g_loss = None
        d_loss = None
        loss = None
        if len(patch_gen) != 0:
            patch_gen = torch.cat(patch_gen, 0)
            self.real_A = torch.cat(real_a, 0)
            self.fake_B = patch_gen
            # save_image(patch_gen[0], '@fuhao/transfer/patch_gen.jpg', normalize=True)
            # save_image(img[0], '@fuhao/transfer/img_gen.jpg', normalize=True)
            valid = Variable(self.Tensor(patch_gen.size(0), 1).fill_(1.0), requires_grad=False)
            fake  = Variable(self.Tensor(patch_gen.size(0), 1).fill_(0.0), requires_grad=False)
            # Loss measures generator's ability to fool the discriminator
            # g_loss = self.adversarial_loss(self.net_d(patch_gen), valid)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            real_imgs = []
            for i in range(len(patch_gen)):
                index = torch.rand(1)

                index = int(index * self.length)
                pic1 = self.voc_dataset.__getitem__(index)
                pic1 = pic1.unsqueeze(0).cuda()
                real_imgs.append(pic1)
                
            real_imgs = torch.cat(real_imgs, 0)
            self.real_B = real_imgs
            # save_image(real_imgs[0], '@fuhao/transfer/real_img.jpg', normalize=True)

            self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
            self.fake_A = self.netG_B(self.real_B)  # G_B(B)
            self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))
            
            if self.iter % 2 == 0:
                
                # 训练生成器
                self.idt_A = self.netG_A(self.real_B)
                self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) 
                # G_B should be identity if real_A is fed: ||G_B(A) - A||
                self.idt_B = self.netG_B(self.real_A)
                self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) 

                self.loss_G_A = self.adversarial_loss(self.netD_A(self.fake_B), valid)
                # GAN loss D_B(G_B(B))
                self.loss_G_B = self.adversarial_loss(self.netD_B(self.fake_A), valid)
                # Forward cycle loss || G_B(G_A(A)) - A||
                self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A)
                # Backward cycle loss || G_A(G_B(B)) - B||
                self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B)
                loss = self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
           
            else:
                
                real_loss = self.adversarial_loss(self.net_D_A(self.real_B), valid)
                fake_loss = self.adversarial_loss(self.net_D_A(self.fake_B.detach()), fake)
                d_loss1 = (real_loss + fake_loss) / 2
                
                real_loss = self.adversarial_loss(self.net_D_B(self.real_A), valid)
                fake_loss = self.adversarial_loss(self.net_D_B(self.fake_A.detach()), fake)
                d_loss2 = (real_loss + fake_loss) / 2
                
                loss = d_loss1 + d_loss2
                
            # real_imgs.cuda()
            # Measure discriminator's ability to classify real from generated samples
            # print(type(real_imgs), type(valid))
            # real_loss = self.adversarial_loss(self.net_d(real_imgs), valid)
            # fake_loss = self.adversarial_loss(self.net_d(patch_gen.detach()), fake)
            # d_loss = (real_loss + fake_loss) / 2
        
        
        # spa_loss = self.spa_loss(ori_img, img) * 20
        # color_loss = self.color_loss(img) 
        # color_loss = 5 * torch.mean(color_loss) * 1000
        # x = self.backbone(img)
        # if self.with_neck:
        #     x = self.neck(x)
        x = img
        return x, loss

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
        
        x, loss = self.extract_feat(img, img_metas, gt_bboxes)
        
        losses = {}
        # losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
        #                                       gt_labels, gt_bboxes_ignore, 
        #                                       label_mask=None
        #                                         )
        if loss != None:
            losses.update({'cyc_loss':loss})

        else:
            losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                                gt_labels, gt_bboxes_ignore, 
                                                label_mask=None
                                                    )
        
        return losses

    def simple_test(self, img, img_metas, rescale=False):

        x = self.extract_feat_test(img)
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
    
    
    

# ======================================================================================================================
# Patch CycleGAN + Detector
# ======================================================================================================================
@DETECTORS.register_module()
class RetinaNet_CycleGAN_22(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 bbox_head_1=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RetinaNet_CycleGAN_22, self).__init__(backbone, neck, bbox_head, bbox_head_1, train_cfg,
                                        test_cfg, pretrained)
        
        self.voc_dataset = MyDataset('../datasets/coco2017/patched/')
        self.length = len(self.voc_dataset)
        # self.net_g = RRDNet()
        # self.net_d = MyVgg16()
        self.netG_A = RRDNet()
        self.netG_A.cuda()
        self.netG_B = RRDNet()
        self.netG_B.cuda()
        
        self.netD_A = MyVgg16()
        self.netD_A.cuda()
        self.netD_B = MyVgg16()
        self.netD_B.cuda()
        
        self.spa_loss = L_spa8()
        self.color_loss = L_color()
        self.tv_loss = L_TV()
        self.Tensor = torch.cuda.FloatTensor
        self.adversarial_loss = torch.nn.BCELoss()
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionCycle.cuda()
        self.criterionIdt = torch.nn.L1Loss()
        self.criterionIdt.cuda()
        
        # self.net_g.cuda()
        # self.net_d.cuda()
        self.adversarial_loss.cuda()
        self.spa_loss.cuda()
        self.color_loss.cuda()
        self.tv_loss.cuda()
        
        self.iter = 0

    def extract_feat_test(self, img):
        """Directly extract features from the backbone+neck."""
        
        img = self.netG_A(img)
                
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)

        return x

    def extract_feat(self, img, img_metas, gt_bboxes):
        """Directly extract features from the backbone+neck."""
        ori_img = img
        img = self.netG_A(ori_img)

        N, C, H, W = img.shape

        Mask_fg = torch.zeros([N, H, W]).cuda()
        Mask_bg = torch.ones([N, H, W]).cuda()
        wmin,wmax,hmin,hmax = [],[],[],[]
        patch_gen = []
        real_a = []
        # gan_img = nn.functional.interpolate(img, size=[64, 64], mode='bilinear', align_corners=False)
        # patch_gen.append(gan_img)
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
                patch = img[i][ :, hmin[i][j]:hmax[i][j], wmin[i][j]:wmax[i][j]]
                patch = patch.unsqueeze(0)
                patch = nn.functional.interpolate(patch, size=[64, 64], mode='bilinear', align_corners=False)
                patch_gen.append(patch)

                patch = ori_img[i][ :, hmin[i][j]:hmax[i][j], wmin[i][j]:wmax[i][j]]
                patch = patch.unsqueeze(0)
                patch = nn.functional.interpolate(patch, size=[64, 64], mode='bilinear', align_corners=False)
                real_a.append(patch)
                
                
                
                Mask_fg[i][max(hmin[i][j], 0):min(hmax[i][j], H), max(wmin[i][j], 0):min(wmax[i][j], W)] = 1.0
            Mask_bg[i] = torch.where(Mask_fg[i]>0, 0, 1)
        
        g_loss = None
        d_loss = None
        loss = None
        if len(patch_gen) != 0:
            patch_gen = torch.cat(patch_gen, 0)
            self.real_A = torch.cat(real_a, 0)
            self.fake_B = patch_gen
            save_image(patch_gen[0], '@fuhao/transfer/patch_gen.jpg', normalize=True)
            save_image(img[0], '@fuhao/transfer/img_gen.jpg', normalize=True)
            valid = Variable(self.Tensor(patch_gen.size(0), 1).fill_(1.0), requires_grad=False)
            fake  = Variable(self.Tensor(patch_gen.size(0), 1).fill_(0.0), requires_grad=False)
            # Loss measures generator's ability to fool the discriminator
            # g_loss = self.adversarial_loss(self.net_d(patch_gen), valid)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            real_imgs = []
            for i in range(len(patch_gen)):
                index = torch.rand(1)

                index = int(index * self.length)
                pic1 = self.voc_dataset.__getitem__(index)
                pic1 = pic1.unsqueeze(0).cuda()
                real_imgs.append(pic1)
                
            real_imgs = torch.cat(real_imgs, 0)
            self.real_B = real_imgs
            # save_image(real_imgs[0], '@fuhao/transfer/real_img.jpg', normalize=True)

            self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
            self.fake_A = self.netG_B(self.real_B)  # G_B(B)
            self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))
            
            if self.iter % 2 == 0:
                
                # 训练生成器
                self.idt_A = self.netG_A(self.real_B)
                self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) 
                # G_B should be identity if real_A is fed: ||G_B(A) - A||
                self.idt_B = self.netG_B(self.real_A)
                self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) 

                self.loss_G_A = self.adversarial_loss(self.netD_A(self.fake_B), valid)
                # GAN loss D_B(G_B(B))
                self.loss_G_B = self.adversarial_loss(self.netD_B(self.fake_A), valid)
                # Forward cycle loss || G_B(G_A(A)) - A||
                self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A)
                # Backward cycle loss || G_A(G_B(B)) - B||
                self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B)
                loss = self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
           
            else:
                
                real_loss = self.adversarial_loss(self.net_D_A(self.real_B), valid)
                fake_loss = self.adversarial_loss(self.net_D_A(self.fake_B.detach()), fake)
                d_loss1 = (real_loss + fake_loss) / 2
                
                real_loss = self.adversarial_loss(self.net_D_B(self.real_A), valid)
                fake_loss = self.adversarial_loss(self.net_D_B(self.fake_A.detach()), fake)
                d_loss2 = (real_loss + fake_loss) / 2
                
                loss = d_loss1 + d_loss2
                
            # real_imgs.cuda()
            # Measure discriminator's ability to classify real from generated samples
            # print(type(real_imgs), type(valid))
            # real_loss = self.adversarial_loss(self.net_d(real_imgs), valid)
            # fake_loss = self.adversarial_loss(self.net_d(patch_gen.detach()), fake)
            # d_loss = (real_loss + fake_loss) / 2
        
        
        # spa_loss = self.spa_loss(ori_img, img) * 20
        # color_loss = self.color_loss(img) 
        # color_loss = 5 * torch.mean(color_loss) * 1000
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)

        return x, loss

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
        
        x, loss = self.extract_feat(img, img_metas, gt_bboxes)
        

        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore, 
                                              label_mask=None
                                                )
        if loss != None:
            losses.update({'cyc_loss':loss})

        
        return losses

    def simple_test(self, img, img_metas, rescale=False):

        x = self.extract_feat_test(img)
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
    




# ======================================================================================================================
# Patch CycleGAN + Detector with Largekernelnet
# ======================================================================================================================
@DETECTORS.register_module()
class RetinaNet_CycleGAN_25(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 kernel=7,
                 bbox_head_1=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RetinaNet_CycleGAN_25, self).__init__(backbone, neck, bbox_head, bbox_head_1, train_cfg,
                                        test_cfg, pretrained)
        
        self.voc_dataset = MyDataset('../datasets/coco2017/patched/')
        self.length = len(self.voc_dataset)
        # self.net_g = RRDNet()
        # self.net_d = MyVgg16()
        self.netG_A = LargeKernelNet(kernel)
        self.netG_A.cuda()
        self.netG_B = LargeKernelNet(kernel)
        self.netG_B.cuda()
        
        self.netD_A = MyVgg16()
        self.netD_A.cuda()
        self.netD_B = MyVgg16()
        self.netD_B.cuda()
        
        self.spa_loss = L_spa8()
        self.color_loss = L_color()
        self.tv_loss = L_TV()
        self.Tensor = torch.cuda.FloatTensor
        self.adversarial_loss = torch.nn.BCELoss()
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionCycle.cuda()
        self.criterionIdt = torch.nn.L1Loss()
        self.criterionIdt.cuda()
        
        # self.net_g.cuda()
        # self.net_d.cuda()
        self.adversarial_loss.cuda()
        self.spa_loss.cuda()
        self.color_loss.cuda()
        self.tv_loss.cuda()
        
        self.iter = 0

    def extract_feat_test(self, img):
        """Directly extract features from the backbone+neck."""
        
        img = self.netG_A(img)
                
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)

        return x

    def extract_feat(self, img, img_metas, gt_bboxes):
        """Directly extract features from the backbone+neck."""
        ori_img = img
        img = self.netG_A(ori_img)

        N, C, H, W = img.shape

        Mask_fg = torch.zeros([N, H, W]).cuda()
        Mask_bg = torch.ones([N, H, W]).cuda()
        wmin,wmax,hmin,hmax = [],[],[],[]
        patch_gen = []
        real_a = []
        # gan_img = nn.functional.interpolate(img, size=[64, 64], mode='bilinear', align_corners=False)
        # patch_gen.append(gan_img)
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
                patch = img[i][ :, hmin[i][j]:hmax[i][j], wmin[i][j]:wmax[i][j]]
                patch = patch.unsqueeze(0)
                patch = nn.functional.interpolate(patch, size=[64, 64], mode='bilinear', align_corners=False)
                patch_gen.append(patch)

                patch = ori_img[i][ :, hmin[i][j]:hmax[i][j], wmin[i][j]:wmax[i][j]]
                patch = patch.unsqueeze(0)
                patch = nn.functional.interpolate(patch, size=[64, 64], mode='bilinear', align_corners=False)
                real_a.append(patch)
                
                
                
                Mask_fg[i][max(hmin[i][j], 0):min(hmax[i][j], H), max(wmin[i][j], 0):min(wmax[i][j], W)] = 1.0
            Mask_bg[i] = torch.where(Mask_fg[i]>0, 0, 1)
        
        g_loss = None
        d_loss = None
        loss = None
        if len(patch_gen) != 0:
            patch_gen = torch.cat(patch_gen, 0)
            self.real_A = torch.cat(real_a, 0)
            self.fake_B = patch_gen
            save_image(patch_gen[0], '@fuhao/transfer/patch_gen.jpg', normalize=True)
            save_image(img[0], '@fuhao/transfer/img_gen.jpg', normalize=True)
            valid = Variable(self.Tensor(patch_gen.size(0), 1).fill_(1.0), requires_grad=False)
            fake  = Variable(self.Tensor(patch_gen.size(0), 1).fill_(0.0), requires_grad=False)
            # Loss measures generator's ability to fool the discriminator
            # g_loss = self.adversarial_loss(self.net_d(patch_gen), valid)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            real_imgs = []
            for i in range(len(patch_gen)):
                index = torch.rand(1)

                index = int(index * self.length)
                pic1 = self.voc_dataset.__getitem__(index)
                pic1 = pic1.unsqueeze(0).cuda()
                real_imgs.append(pic1)
                
            real_imgs = torch.cat(real_imgs, 0)
            self.real_B = real_imgs
            # save_image(real_imgs[0], '@fuhao/transfer/real_img.jpg', normalize=True)

            self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
            self.fake_A = self.netG_B(self.real_B)  # G_B(B)
            self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))
            
            if self.iter % 2 == 0:
                
                # 训练生成器
                self.idt_A = self.netG_A(self.real_B)
                self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) 
                # G_B should be identity if real_A is fed: ||G_B(A) - A||
                self.idt_B = self.netG_B(self.real_A)
                self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) 

                self.loss_G_A = self.adversarial_loss(self.netD_A(self.fake_B), valid)
                # GAN loss D_B(G_B(B))
                self.loss_G_B = self.adversarial_loss(self.netD_B(self.fake_A), valid)
                # Forward cycle loss || G_B(G_A(A)) - A||
                self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A)
                # Backward cycle loss || G_A(G_B(B)) - B||
                self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B)
                loss = self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
           
            else:
                
                real_loss = self.adversarial_loss(self.net_D_A(self.real_B), valid)
                fake_loss = self.adversarial_loss(self.net_D_A(self.fake_B.detach()), fake)
                d_loss1 = (real_loss + fake_loss) / 2
                
                real_loss = self.adversarial_loss(self.net_D_B(self.real_A), valid)
                fake_loss = self.adversarial_loss(self.net_D_B(self.fake_A.detach()), fake)
                d_loss2 = (real_loss + fake_loss) / 2
                
                loss = d_loss1 + d_loss2
                
            # real_imgs.cuda()
            # Measure discriminator's ability to classify real from generated samples
            # print(type(real_imgs), type(valid))
            # real_loss = self.adversarial_loss(self.net_d(real_imgs), valid)
            # fake_loss = self.adversarial_loss(self.net_d(patch_gen.detach()), fake)
            # d_loss = (real_loss + fake_loss) / 2
        
        
        # spa_loss = self.spa_loss(ori_img, img) * 20
        # color_loss = self.color_loss(img) 
        # color_loss = 5 * torch.mean(color_loss) * 1000
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)

        return x, loss

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
        
        x, loss = self.extract_feat(img, img_metas, gt_bboxes)
        

        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore, 
                                              label_mask=None
                                                )
        if loss != None:
            losses.update({'cyc_loss':loss})

        
        return losses

    def simple_test(self, img, img_metas, rescale=False):

        x = self.extract_feat_test(img)
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

# ======================================================================================================================
# 前后背景不分离的版本
# ======================================================================================================================
@DETECTORS.register_module()
class RetinaNet_CycleGAN_24(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 bbox_head_1=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RetinaNet_CycleGAN_24, self).__init__(backbone, neck, bbox_head, bbox_head_1, train_cfg,
                                        test_cfg, pretrained)
        
        self.voc_dataset = MyDataset('../datasets/coco2017/val2017/')
        self.length = len(self.voc_dataset)
        # self.net_g = RRDNet()
        # self.net_d = MyVgg16()
        self.netG_A = RRDNet()
        self.netG_A.cuda()
        self.netG_B = RRDNet()
        self.netG_B.cuda()
        
        self.netD_A = MyVgg16()
        self.netD_A.cuda()
        self.netD_B = MyVgg16()
        self.netD_B.cuda()
        
        self.spa_loss = L_spa8()
        self.color_loss = L_color()
        self.tv_loss = L_TV()
        self.Tensor = torch.cuda.FloatTensor
        self.adversarial_loss = torch.nn.BCELoss()
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionCycle.cuda()
        self.criterionIdt = torch.nn.L1Loss()
        self.criterionIdt.cuda()
        
        # self.net_g.cuda()
        # self.net_d.cuda()
        self.adversarial_loss.cuda()
        self.spa_loss.cuda()
        self.color_loss.cuda()
        self.tv_loss.cuda()
        
        self.iter = 0

    def extract_feat_test(self, img):
        """Directly extract features from the backbone+neck."""
        
        img = self.netG_A(img)
                
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)

        return x

    def extract_feat(self, img, img_metas, gt_bboxes):
        """Directly extract features from the backbone+neck."""
        ori_img = img
        img = self.netG_A(ori_img)

        N, C, H, W = img.shape

        Mask_fg = torch.zeros([N, H, W]).cuda()
        Mask_bg = torch.ones([N, H, W]).cuda()
        wmin,wmax,hmin,hmax = [],[],[],[]
        patch_gen = []
        real_a = []
        # gan_img = nn.functional.interpolate(img, size=[64, 64], mode='bilinear', align_corners=False)
        # patch_gen.append(gan_img)
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
                patch = img[i][ :, hmin[i][j]:hmax[i][j], wmin[i][j]:wmax[i][j]]
                patch = patch.unsqueeze(0)
                patch = nn.functional.interpolate(patch, size=[64, 64], mode='bilinear', align_corners=False)
                patch_gen.append(patch)

                patch = ori_img[i][ :, hmin[i][j]:hmax[i][j], wmin[i][j]:wmax[i][j]]
                patch = patch.unsqueeze(0)
                patch = nn.functional.interpolate(patch, size=[64, 64], mode='bilinear', align_corners=False)
                real_a.append(patch)
                
                
                
                Mask_fg[i][max(hmin[i][j], 0):min(hmax[i][j], H), max(wmin[i][j], 0):min(wmax[i][j], W)] = 1.0
            Mask_bg[i] = torch.where(Mask_fg[i]>0, 0, 1)
        
        g_loss = None
        d_loss = None
        loss = None
        if len(patch_gen) != 0:
            patch_gen = torch.cat(patch_gen, 0)
            self.real_A = torch.cat(real_a, 0)
            self.fake_B = patch_gen
            # save_image(patch_gen[0], '@fuhao/transfer/patch_gen.jpg', normalize=True)
            # save_image(img[0], '@fuhao/transfer/img_gen.jpg', normalize=True)
            valid = Variable(self.Tensor(patch_gen.size(0), 1).fill_(1.0), requires_grad=False)
            fake  = Variable(self.Tensor(patch_gen.size(0), 1).fill_(0.0), requires_grad=False)
            # Loss measures generator's ability to fool the discriminator
            # g_loss = self.adversarial_loss(self.net_d(patch_gen), valid)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            real_imgs = []
            for i in range(len(patch_gen)):
                index = torch.rand(1)

                index = int(index * self.length)
                pic1 = self.voc_dataset.__getitem__(index)
                pic1 = pic1.unsqueeze(0).cuda()
                real_imgs.append(pic1)
                
            real_imgs = torch.cat(real_imgs, 0)
            self.real_B = real_imgs
            # save_image(real_imgs[0], '@fuhao/transfer/real_img.jpg', normalize=True)

            self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
            self.fake_A = self.netG_B(self.real_B)  # G_B(B)
            self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))
            
            if self.iter % 2 == 0:
                
                # 训练生成器
                self.idt_A = self.netG_A(self.real_B)
                self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) 
                # G_B should be identity if real_A is fed: ||G_B(A) - A||
                self.idt_B = self.netG_B(self.real_A)
                self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) 

                self.loss_G_A = self.adversarial_loss(self.netD_A(self.fake_B), valid)
                # GAN loss D_B(G_B(B))
                self.loss_G_B = self.adversarial_loss(self.netD_B(self.fake_A), valid)
                # Forward cycle loss || G_B(G_A(A)) - A||
                self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A)
                # Backward cycle loss || G_A(G_B(B)) - B||
                self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B)
                loss = self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
           
            else:
                
                real_loss = self.adversarial_loss(self.net_D_A(self.real_B), valid)
                fake_loss = self.adversarial_loss(self.net_D_A(self.fake_B.detach()), fake)
                d_loss1 = (real_loss + fake_loss) / 2
                
                real_loss = self.adversarial_loss(self.net_D_B(self.real_A), valid)
                fake_loss = self.adversarial_loss(self.net_D_B(self.fake_A.detach()), fake)
                d_loss2 = (real_loss + fake_loss) / 2
                
                loss = d_loss1 + d_loss2
                
            # real_imgs.cuda()
            # Measure discriminator's ability to classify real from generated samples
            # print(type(real_imgs), type(valid))
            # real_loss = self.adversarial_loss(self.net_d(real_imgs), valid)
            # fake_loss = self.adversarial_loss(self.net_d(patch_gen.detach()), fake)
            # d_loss = (real_loss + fake_loss) / 2
        
        
        # spa_loss = self.spa_loss(ori_img, img) * 20
        # color_loss = self.color_loss(img) 
        # color_loss = 5 * torch.mean(color_loss) * 1000
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)

        return x, loss

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
        
        x, loss = self.extract_feat(img, img_metas, gt_bboxes)
        

        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore, 
                                              label_mask=None
                                                )
        if loss != None:
            losses.update({'cyc_loss':loss})

        
        return losses

    def simple_test(self, img, img_metas, rescale=False):

        x = self.extract_feat_test(img)
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
    
    

# ======================================================================================================================
# Patch CycleGAN + Contrastive learning
# ======================================================================================================================
from .tools.DASR.MoCo import MoCo
@DETECTORS.register_module()
class RetinaNet_CycleGAN_CL(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 bbox_head_1=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RetinaNet_CycleGAN_CL, self).__init__(backbone, neck, bbox_head, bbox_head_1, train_cfg,
                                        test_cfg, pretrained)
        
        self.voc_dataset = MyDataset('../datasets/coco2017/patched/')
        self.length = len(self.voc_dataset)
        # self.net_g = RRDNet()
        # self.net_d = MyVgg16()
        self.netG_A = RRDNet()
        self.netG_A.cuda()
        self.netG_B = RRDNet()
        self.netG_B.cuda()
        self.netE = MoCo()
        self.netE.cuda()
        
        self.netD_A = MyVgg16()
        self.netD_A.cuda()
        self.netD_B = MyVgg16()
        self.netD_B.cuda()
        
        self.spa_loss = L_spa8()
        self.color_loss = L_color()
        self.tv_loss = L_TV()
        self.Tensor = torch.cuda.FloatTensor
        self.adversarial_loss = torch.nn.BCELoss()
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionCycle.cuda()
        self.criterionIdt = torch.nn.L1Loss()
        self.criterionIdt.cuda()
        
        self.contrast_loss = torch.nn.CrossEntropyLoss().cuda()
        
        # self.net_g.cuda()
        # self.net_d.cuda()
        self.adversarial_loss.cuda()
        self.spa_loss.cuda()
        self.color_loss.cuda()
        self.tv_loss.cuda()
        
        self.iter = 0

    def extract_feat_test(self, img):
        """Directly extract features from the backbone+neck."""
        
        img = self.netG_A(img)
                
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)

        return x

    def extract_feat(self, img, img_metas, gt_bboxes):
        """Directly extract features from the backbone+neck."""
        ori_img = img
        img = self.netG_A(ori_img)

        N, C, H, W = img.shape

        Mask_fg = torch.zeros([N, H, W]).cuda()
        Mask_bg = torch.ones([N, H, W]).cuda()
        wmin,wmax,hmin,hmax = [],[],[],[]
        patch_gen = []
        real_a = []
        # gan_img = nn.functional.interpolate(img, size=[64, 64], mode='bilinear', align_corners=False)
        # patch_gen.append(gan_img)
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
                patch = img[i][ :, hmin[i][j]:hmax[i][j], wmin[i][j]:wmax[i][j]]
                patch = patch.unsqueeze(0)
                patch = nn.functional.interpolate(patch, size=[64, 64], mode='bilinear', align_corners=False)
                patch_gen.append(patch)

                patch = ori_img[i][ :, hmin[i][j]:hmax[i][j], wmin[i][j]:wmax[i][j]]
                patch = patch.unsqueeze(0)
                patch = nn.functional.interpolate(patch, size=[64, 64], mode='bilinear', align_corners=False)
                real_a.append(patch)
                
                
                
                Mask_fg[i][max(hmin[i][j], 0):min(hmax[i][j], H), max(wmin[i][j], 0):min(wmax[i][j], W)] = 1.0
            Mask_bg[i] = torch.where(Mask_fg[i]>0, 0, 1)
        
        g_loss = None
        d_loss = None
        loss = None
        if len(patch_gen) != 0:
            patch_gen = torch.cat(patch_gen, 0)
            self.real_A = torch.cat(real_a, 0)
            self.fake_B = patch_gen
            save_image(patch_gen[0], '@fuhao/transfer/patch_gen.jpg', normalize=True)
            save_image(img[0], '@fuhao/transfer/img_gen.jpg', normalize=True)
            valid = Variable(self.Tensor(patch_gen.size(0), 1).fill_(1.0), requires_grad=False)
            fake  = Variable(self.Tensor(patch_gen.size(0), 1).fill_(0.0), requires_grad=False)
            # Loss measures generator's ability to fool the discriminator
            # g_loss = self.adversarial_loss(self.net_d(patch_gen), valid)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            real_imgs = []
            for i in range(len(patch_gen)):
                index = torch.rand(1)

                index = int(index * self.length)
                pic1 = self.voc_dataset.__getitem__(index)
                pic1 = pic1.unsqueeze(0).cuda()
                real_imgs.append(pic1)
                
            real_imgs = torch.cat(real_imgs, 0)
            self.real_B = real_imgs
            # save_image(real_imgs[0], '@fuhao/transfer/real_img.jpg', normalize=True)

            self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
            self.fake_A = self.netG_B(self.real_B)  # G_B(B)
            self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))
            
            if self.iter % 2 == 0:
                
                # 训练生成器
                self.idt_A = self.netG_A(self.real_B)
                self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) 
                # G_B should be identity if real_A is fed: ||G_B(A) - A||
                self.idt_B = self.netG_B(self.real_A)
                self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) 

                self.loss_G_A = self.adversarial_loss(self.netD_A(self.fake_B), valid)
                # GAN loss D_B(G_B(B))
                self.loss_G_B = self.adversarial_loss(self.netD_B(self.fake_A), valid)
                # Forward cycle loss || G_B(G_A(A)) - A||
                self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A)
                # Backward cycle loss || G_A(G_B(B)) - B||
                self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B)
                
                # ==============================
                # contrastive loss
                # ==============================
                
                _, output, target = self.netE(im_q=self.fake_B, im_k1=self.real_B, im_k2=self.real_A)
                self.loss_constrast = self.contrast_loss(output, target)

                
                loss = self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_constrast * 0.01
           
            else:
                
                real_loss = self.adversarial_loss(self.net_D_A(self.real_B), valid)
                fake_loss = self.adversarial_loss(self.net_D_A(self.fake_B.detach()), fake)
                d_loss1 = (real_loss + fake_loss) / 2
                
                real_loss = self.adversarial_loss(self.net_D_B(self.real_A), valid)
                fake_loss = self.adversarial_loss(self.net_D_B(self.fake_A.detach()), fake)
                d_loss2 = (real_loss + fake_loss) / 2
                
                loss = d_loss1 + d_loss2
                
            # real_imgs.cuda()
            # Measure discriminator's ability to classify real from generated samples
            # print(type(real_imgs), type(valid))
            # real_loss = self.adversarial_loss(self.net_d(real_imgs), valid)
            # fake_loss = self.adversarial_loss(self.net_d(patch_gen.detach()), fake)
            # d_loss = (real_loss + fake_loss) / 2
        
        
        # spa_loss = self.spa_loss(ori_img, img) * 20
        # color_loss = self.color_loss(img) 
        # color_loss = 5 * torch.mean(color_loss) * 1000
        # x = self.backbone(img)
        # if self.with_neck:
        #     x = self.neck(x)
        x = img

        return x, loss

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
        
        x, loss = self.extract_feat(img, img_metas, gt_bboxes)
        
        losses = {}
        # losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
        #                                       gt_labels, gt_bboxes_ignore, 
        #                                       label_mask=None
        #                                         )
        if loss != None:
            losses.update({'cyc_loss':loss})

        
        return losses

    def simple_test(self, img, img_metas, rescale=False):

        x = self.extract_feat_test(img)
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
    
    
# ======================================================================================================================
# Patch CycleGAN  Double Contrastive learning 
# ======================================================================================================================
from .tools.DASR.MoCo import MoCo
@DETECTORS.register_module()
class RetinaNet_CycleGAN_DCL(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 bbox_head_1=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RetinaNet_CycleGAN_DCL, self).__init__(backbone, neck, bbox_head, bbox_head_1, train_cfg,
                                        test_cfg, pretrained)
        
        self.voc_dataset = MyDataset('../datasets/coco2017/patched/')
        self.length = len(self.voc_dataset)
        # self.net_g = RRDNet()
        # self.net_d = MyVgg16()
        self.netG_A = RRDNet()
        self.netG_A.cuda()
        self.netG_B = RRDNet()
        self.netG_B.cuda()
        self.netE = MoCo()
        self.netE.cuda()
        
        self.netF = MoCo()
        self.netF.cuda()
        
        self.netD_A = MyVgg16()
        self.netD_A.cuda()
        self.netD_B = MyVgg16()
        self.netD_B.cuda()
        
        self.spa_loss = L_spa8()
        self.color_loss = L_color()
        self.tv_loss = L_TV()
        self.Tensor = torch.cuda.FloatTensor
        self.adversarial_loss = torch.nn.BCELoss()
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionCycle.cuda()
        self.criterionIdt = torch.nn.L1Loss()
        self.criterionIdt.cuda()
        
        self.contrast_loss = torch.nn.CrossEntropyLoss().cuda()
        
        # self.net_g.cuda()
        # self.net_d.cuda()
        self.adversarial_loss.cuda()
        self.spa_loss.cuda()
        self.color_loss.cuda()
        self.tv_loss.cuda()
        
        self.iter = 0

    def extract_feat_test(self, img):
        """Directly extract features from the backbone+neck."""
        
        img = self.netG_A(img)
                
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)

        return x

    def extract_feat(self, img, img_metas, gt_bboxes):
        """Directly extract features from the backbone+neck."""
        ori_img = img
        img = self.netG_A(ori_img)

        N, C, H, W = img.shape

        Mask_fg = torch.zeros([N, H, W]).cuda()
        Mask_bg = torch.ones([N, H, W]).cuda()
        wmin,wmax,hmin,hmax = [],[],[],[]
        patch_gen = []
        real_a = []
        # gan_img = nn.functional.interpolate(img, size=[64, 64], mode='bilinear', align_corners=False)
        # patch_gen.append(gan_img)
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
                patch = img[i][ :, hmin[i][j]:hmax[i][j], wmin[i][j]:wmax[i][j]]
                patch = patch.unsqueeze(0)
                patch = nn.functional.interpolate(patch, size=[64, 64], mode='bilinear', align_corners=False)
                patch_gen.append(patch)

                patch = ori_img[i][ :, hmin[i][j]:hmax[i][j], wmin[i][j]:wmax[i][j]]
                patch = patch.unsqueeze(0)
                patch = nn.functional.interpolate(patch, size=[64, 64], mode='bilinear', align_corners=False)
                real_a.append(patch)
                
                
                
                Mask_fg[i][max(hmin[i][j], 0):min(hmax[i][j], H), max(wmin[i][j], 0):min(wmax[i][j], W)] = 1.0
            Mask_bg[i] = torch.where(Mask_fg[i]>0, 0, 1)
        
        g_loss = None
        d_loss = None
        loss = None
        if len(patch_gen) != 0:
            patch_gen = torch.cat(patch_gen, 0)
            self.real_A = torch.cat(real_a, 0)
            self.fake_B = patch_gen
            save_image(patch_gen[0], '@fuhao/transfer/patch_gen.jpg', normalize=True)
            save_image(img[0], '@fuhao/transfer/img_gen.jpg', normalize=True)
            valid = Variable(self.Tensor(patch_gen.size(0), 1).fill_(1.0), requires_grad=False)
            fake  = Variable(self.Tensor(patch_gen.size(0), 1).fill_(0.0), requires_grad=False)
            # Loss measures generator's ability to fool the discriminator
            # g_loss = self.adversarial_loss(self.net_d(patch_gen), valid)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            real_imgs = []
            for i in range(len(patch_gen)):
                index = torch.rand(1)

                index = int(index * self.length)
                pic1 = self.voc_dataset.__getitem__(index)
                pic1 = pic1.unsqueeze(0).cuda()
                real_imgs.append(pic1)
                
            real_imgs = torch.cat(real_imgs, 0)
            self.real_B = real_imgs
            # save_image(real_imgs[0], '@fuhao/transfer/real_img.jpg', normalize=True)

            self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
            self.fake_A = self.netG_B(self.real_B)  # G_B(B)
            self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))
            
            if self.iter % 2 == 0:
                
                # 训练生成器
                self.idt_A = self.netG_A(self.real_B)
                self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) 
                # G_B should be identity if real_A is fed: ||G_B(A) - A||
                self.idt_B = self.netG_B(self.real_A)
                self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) 

                self.loss_G_A = self.adversarial_loss(self.netD_A(self.fake_B), valid)
                # GAN loss D_B(G_B(B))
                self.loss_G_B = self.adversarial_loss(self.netD_B(self.fake_A), valid)
                # Forward cycle loss || G_B(G_A(A)) - A||
                self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A)
                # Backward cycle loss || G_A(G_B(B)) - B||
                self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B)
                
                # ==============================
                # patch contrastive loss
                # ==============================
                
                _, output, target = self.netE(im_q=self.fake_B, im_k1=self.real_B, im_k2=self.real_A)
                self.loss_constrast = self.contrast_loss(output, target)
                
                # ==============================
                # img contrastive loss
                # ==============================
                self.gen_img = nn.functional.interpolate(img, size=[64, 64], mode='bilinear', align_corners=False)
                self.ori_img = nn.functional.interpolate(ori_img, size=[64, 64], mode='bilinear', align_corners=False)
                _, output, target = self.netF(im_q=self.gen_img, im_k1=self.real_B, im_k2=self.ori_img)
                self.loss_constrast_img = self.contrast_loss(output, target)

                
                loss = self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_constrast * 0.01
           
            else:
                
                real_loss = self.adversarial_loss(self.net_D_A(self.real_B), valid)
                fake_loss = self.adversarial_loss(self.net_D_A(self.fake_B.detach()), fake)
                d_loss1 = (real_loss + fake_loss) / 2
                
                real_loss = self.adversarial_loss(self.net_D_B(self.real_A), valid)
                fake_loss = self.adversarial_loss(self.net_D_B(self.fake_A.detach()), fake)
                d_loss2 = (real_loss + fake_loss) / 2
                
                loss = d_loss1 + d_loss2
                
            # real_imgs.cuda()
            # Measure discriminator's ability to classify real from generated samples
            # print(type(real_imgs), type(valid))
            # real_loss = self.adversarial_loss(self.net_d(real_imgs), valid)
            # fake_loss = self.adversarial_loss(self.net_d(patch_gen.detach()), fake)
            # d_loss = (real_loss + fake_loss) / 2
        
        
        # spa_loss = self.spa_loss(ori_img, img) * 20
        # color_loss = self.color_loss(img) 
        # color_loss = 5 * torch.mean(color_loss) * 1000
        # x = self.backbone(img)
        # if self.with_neck:
        #     x = self.neck(x)
        x = img

        return x, loss

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
        
        x, loss = self.extract_feat(img, img_metas, gt_bboxes)
        
        losses = {}
        # losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
        #                                       gt_labels, gt_bboxes_ignore, 
        #                                       label_mask=None
        #                                         )
        if loss != None:
            losses.update({'cyc_loss':loss})

        
        return losses

    def simple_test(self, img, img_metas, rescale=False):

        x = self.extract_feat_test(img)
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
    

# ======================================================================================================================
# Patch CycleGAN + Detector + 整幅图像 CycleGAN
# ======================================================================================================================
@DETECTORS.register_module()
class RetinaNet_CycleGAN_26(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_detector=True,
                 bbox_head_1=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RetinaNet_CycleGAN_26, self).__init__(backbone, neck, bbox_head, bbox_head_1, train_cfg,
                                        test_cfg, pretrained)
        
        self.train_detector = train_detector
        
        self.voc_dataset = MyDataset('../datasets/coco2017/patched/')
        self.length = len(self.voc_dataset)
        
        self.voc_dataset_global = MyDataset('../datasets/coco2017/val2017/')
        self.length_global = len(self.voc_dataset_global)
        # self.net_g = RRDNet()
        # self.net_d = MyVgg16()
        self.netG_A = RRDNet()
        self.netG_A.cuda()
        self.netG_B = RRDNet()
        self.netG_B.cuda()
        
        self.netD_A = MyVgg16()
        self.netD_A.cuda()
        self.netD_B = MyVgg16()
        self.netD_B.cuda()
        
        self.spa_loss = L_spa8()
        self.color_loss = L_color()
        self.tv_loss = L_TV()
        self.Tensor = torch.cuda.FloatTensor
        self.adversarial_loss = torch.nn.BCELoss()
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionCycle.cuda()
        self.criterionIdt = torch.nn.L1Loss()
        self.criterionIdt.cuda()
        
        # self.net_g.cuda()
        # self.net_d.cuda()
        self.adversarial_loss.cuda()
        self.spa_loss.cuda()
        self.color_loss.cuda()
        self.tv_loss.cuda()
        
        self.iter = 0
        self.train_D = 1
        
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def extract_feat_test(self, img):
        """Directly extract features from the backbone+neck."""
        
        img = self.netG_A(img)
                
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)

        return x

    def extract_feat(self, img, img_metas, gt_bboxes):
        """Directly extract features from the backbone+neck."""
        ori_img = img
        img = self.netG_A(ori_img)

        N, C, H, W = img.shape

        Mask_fg = torch.zeros([N, H, W]).cuda()
        Mask_bg = torch.ones([N, H, W]).cuda()
        wmin,wmax,hmin,hmax = [],[],[],[]
        patch_gen = []
        real_a = []
        # gan_img = nn.functional.interpolate(img, size=[64, 64], mode='bilinear', align_corners=False)
        # patch_gen.append(gan_img)
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
                patch = img[i][ :, hmin[i][j]:hmax[i][j], wmin[i][j]:wmax[i][j]]
                patch = patch.unsqueeze(0)
                patch = nn.functional.interpolate(patch, size=[64, 64], mode='bilinear', align_corners=False)
                patch_gen.append(patch)

                patch = ori_img[i][ :, hmin[i][j]:hmax[i][j], wmin[i][j]:wmax[i][j]]
                patch = patch.unsqueeze(0)
                patch = nn.functional.interpolate(patch, size=[64, 64], mode='bilinear', align_corners=False)
                real_a.append(patch)
                
                
                
                Mask_fg[i][max(hmin[i][j], 0):min(hmax[i][j], H), max(wmin[i][j], 0):min(wmax[i][j], W)] = 1.0
            Mask_bg[i] = torch.where(Mask_fg[i]>0, 0, 1)
        
        g_loss = None
        d_loss = None
        loss = None
        if len(patch_gen) != 0:
            patch_gen = torch.cat(patch_gen, 0)
            self.real_A = torch.cat(real_a, 0)
            self.fake_B = patch_gen
            # save_image(patch_gen[0], '@fuhao/transfer/patch_gen.jpg', normalize=True)
            # save_image(img[0], '@fuhao/transfer/img_gen.jpg', normalize=True)
            # valid = Variable(self.Tensor(patch_gen.size(0), 1).fill_(1.0), requires_grad=False)
            # fake  = Variable(self.Tensor(patch_gen.size(0), 1).fill_(0.0), requires_grad=False)    
            # valid = Variable(torch.tensor(patch_gen.size(0), 1).fill_(1.0), requires_grad=False)
            # fake  = Variable(torch.tensor(patch_gen.size(0), 1).fill_(0.0), requires_grad=False)  
            valid = torch.ones([patch_gen.shape[0], 1]).cuda()
            fake = torch.ones([patch_gen.shape[0], 1]).cuda()
            
            # Loss measures generator's ability to fool the discriminator
            # g_loss = self.adversarial_loss(self.net_d(patch_gen), valid)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            real_imgs = []
            for i in range(len(patch_gen)):
                index = torch.rand(1)

                index = int(index * self.length)
                pic1 = self.voc_dataset.__getitem__(index)
                pic1 = pic1.unsqueeze(0).cuda()
                real_imgs.append(pic1)
                
            real_imgs = torch.cat(real_imgs, 0)
            self.real_B = real_imgs
            # save_image(real_imgs[0], '@fuhao/transfer/real_img.jpg', normalize=True)

            self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
            self.fake_A = self.netG_B(self.real_B)  # G_B(B)
            self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))
            
            # ========================================================
            # 整幅图像

            # valid_g = Variable(self.Tensor(ori_img.size(0), 1).fill_(1.0), requires_grad=False)
            # fake_g  = Variable(self.Tensor(ori_img.size(0), 1).fill_(0.0), requires_grad=False)   

            # real_imgs = []
            # for i in range(len(ori_img)):
            #     index = torch.rand(1)

            #     index = int(index * self.length_global)
            #     pic1 = self.voc_dataset_global.__getitem__(index)
            #     pic1 = pic1.unsqueeze(0).cuda()
            #     real_imgs.append(pic1)
            # real_imgs = torch.cat(real_imgs, 0)
            # self.real_B_global = real_imgs
            # self.real_A_global = ori_img
            # self.fake_B_global = img
            # self.rec_A_global = self.netG_B(self.fake_B_global)
            # self.fake_A_global = self.netG_B(self.real_B_global)
            # self.rec_B_global = self.netG_A(self.fake_A_global)
            # ========================================================
            
            # print(self.iter)
            if self.iter % 25 != 0:
            # if self.train_D == 0:
            # if True:
                
                # 训练生成器
                # ========================================================
                # 训练生成器的时候，要把判别器冻结
                self.set_requires_grad([self.netD_A, self.netD_B], False)
                # ========================================================
                
                self.idt_A = self.netG_A(self.real_B)
                self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) 
                # G_B should be identity if real_A is fed: ||G_B(A) - A||
                self.idt_B = self.netG_B(self.real_A)
                self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) 

                self.loss_G_A = self.adversarial_loss(self.netD_A(self.fake_B), valid)
                # GAN loss D_B(G_B(B))
                self.loss_G_B = self.adversarial_loss(self.netD_B(self.fake_A), valid)
                # Forward cycle loss || G_B(G_A(A)) - A||
                self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A)
                # Backward cycle loss || G_A(G_B(B)) - B||
                self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B)
                
                # ========================================================
                # 训练整幅图像
                # self.idt_A_g = self.netG_A(self.real_B_global)
                # self.loss_idt_A_g = self.criterionIdt(self.idt_A_g, self.real_B_global) 
                # # G_B should be identity if real_A is fed: ||G_B(A) - A||
                # self.idt_B_g = self.netG_B(self.real_A_global)
                # self.loss_idt_B_g = self.criterionIdt(self.idt_B_g, self.real_A_global) 

                # self.loss_G_A_g = self.adversarial_loss(self.netD_A(self.fake_B_global), valid_g)
                # # GAN loss D_B(G_B(B))
                # self.loss_G_B_g = self.adversarial_loss(self.netD_B(self.fake_A_global), valid_g)
                # # Forward cycle loss || G_B(G_A(A)) - A||
                # self.loss_cycle_A_g = self.criterionCycle(self.rec_A_global, self.real_A_global)
                # # Backward cycle loss || G_A(G_B(B)) - B||
                # self.loss_cycle_B_g = self.criterionCycle(self.rec_B_global, self.real_B_global)
                # ========================================================
                # loss_G = self.loss_cycle_A * 10. + self.loss_cycle_B * 10. + self.loss_idt_A * 5. + self.loss_idt_B * 5.
                # loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A * 10. + self.loss_cycle_B * 10.
                loss_G = self.loss_G_A.mean() + self.loss_G_B.mean() + self.loss_cycle_A  + self.loss_cycle_B  + self.loss_idt_A  + self.loss_idt_B 
                    # self.loss_G_A_g + self.loss_G_B_g + self.loss_cycle_A_g * 10. + self.loss_cycle_B_g * 10. + self.loss_idt_A_g * 5. + self.loss_idt_B_g * 5.
                
                self.iter += 1
                if self.iter % 5 == 0 :
                    self.train_D = 1
                # print('netG => ', self.loss_G_A, self.loss_G_B, self.loss_cycle_A, self.loss_cycle_B)
                loss = loss_G
                # print('G loss => ', loss)
                # ========================================================
                # 解冻
                self.set_requires_grad([self.netD_A, self.netD_B], True)
                # ========================================================
            else:

                real_loss = self.adversarial_loss(self.netD_A(self.real_B), valid)
                fake_loss = self.adversarial_loss(self.netD_A(self.fake_B.detach()), fake)
                d_loss1 = (real_loss.mean() + fake_loss.mean()) / 2
                
                real_loss = self.adversarial_loss(self.netD_B(self.real_A), valid)
                fake_loss = self.adversarial_loss(self.netD_B(self.fake_A.detach()), fake)
                d_loss2 = (real_loss.mean() + fake_loss.mean()) / 2
                
                # ========================================================
                # 训练整幅图像
                # real_loss_g = self.adversarial_loss(self.netD_A(self.real_B_global), valid_g)
                # fake_loss_g = self.adversarial_loss(self.netD_A(self.fake_B_global.detach()), fake_g)
                # d_loss1_g = (real_loss_g + fake_loss_g) / 2
                
                # real_loss_g = self.adversarial_loss(self.netD_B(self.real_A_global), valid_g)
                # fake_loss_g = self.adversarial_loss(self.netD_B(self.fake_A_global.detach()), fake_g)
                # d_loss2_g = (real_loss_g + fake_loss_g) / 2
                # # ========================================================
                
                loss_D = d_loss1 + d_loss2 
                    # d_loss1_g + d_loss2_g
                
                self.iter += 1
                if self.iter % 5 == 0 :
                    self.train_D = 0
                loss = loss_D
                print('netD + ', loss)
                # print('d loss => ', loss)
            # real_imgs.cuda()
            # Measure discriminator's ability to classify real from generated samples
            # print(type(real_imgs), type(valid))
            # real_loss = self.adversarial_loss(self.net_d(real_imgs), valid)
            # fake_loss = self.adversarial_loss(self.net_d(patch_gen.detach()), fake)
            # d_loss = (real_loss + fake_loss) / 2
        
        
        # spa_loss = self.spa_loss(ori_img, img) * 20
        # color_loss = self.color_loss(img) 
        # color_loss = 5 * torch.mean(color_loss) * 1000
        if self.train_detector:
            x = self.backbone(img)
            if self.with_neck:
                x = self.neck(x)
            return x, loss
        else:
            return None, loss

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
        
        x, loss = self.extract_feat(img, img_metas, gt_bboxes)
        
        if self.train_detector:
            losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                                gt_labels, gt_bboxes_ignore, 
                                                label_mask=None
                                                    )
        else:
            losses = {}
        if loss != None:
            losses.update({'cyc_loss':loss})
            # losses.update({'cyc_loss_D':loss_D})

        
        return losses

    def simple_test(self, img, img_metas, rescale=False):

        x = self.extract_feat_test(img)
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
    
    
# ======================================================================================================================
# CycleGAN + Detector （整幅图像）
# ======================================================================================================================

from .tools.GAN.Cycle_networks import define_G
@DETECTORS.register_module()
class RetinaNet_OriCycleGAN(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_detector=True,
                 bbox_head_1=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RetinaNet_OriCycleGAN, self).__init__(backbone, neck, bbox_head, bbox_head_1, train_cfg,
                                        test_cfg, pretrained)
        
        self.train_detector = train_detector
        
        self.voc_dataset = MyDataset('../datasets/coco2017/patched/')
        self.length = len(self.voc_dataset)
        
        self.voc_dataset_global = MyDataset('../datasets/coco2017/val2017/')
        self.length_global = len(self.voc_dataset_global)
        # self.net_g = RRDNet()
        # self.net_d = MyVgg16()
        self.netG_A = RRDNet()
        # self.netG_A = define_G(3, 3, 64, 'resnet_9blocks', 'instance',
                                        # False, 'normal', 0.02, [0])
        self.netG_A.cuda()
        # self.netG_B = define_G(3, 3, 64, 'resnet_9blocks', 'instance',
                                        # False, 'normal', 0.02, [0])
        self.netG_B = RRDNet()
        self.netG_B.cuda()
        
        self.netD_A = MyVgg16()
        self.netD_A.cuda()
        self.netD_B = MyVgg16()
        self.netD_B.cuda()
        
        self.spa_loss = L_spa8()
        self.color_loss = L_color()
        self.tv_loss = L_TV()
        self.Tensor = torch.cuda.FloatTensor
        self.adversarial_loss = torch.nn.BCELoss()
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionCycle.cuda()
        self.criterionIdt = torch.nn.L1Loss()
        self.criterionIdt.cuda()
        
        # self.net_g.cuda()
        # self.net_d.cuda()
        self.adversarial_loss.cuda()
        self.spa_loss.cuda()
        self.color_loss.cuda()
        self.tv_loss.cuda()
        
        self.iter = 0
        self.train_D = 1
        
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def extract_feat_test(self, img):
        """Directly extract features from the backbone+neck."""
        
        img = self.netG_A(img)
                
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)

        return x

    def extract_feat(self, img, img_metas, gt_bboxes):
        """Directly extract features from the backbone+neck."""
        
        self.real_A = img
        self.fake_B = self.netG_A(self.real_A)

        N, C, H, W = img.shape
        
        g_loss = None
        d_loss = None
        loss = None
        if len(self.real_A) != 0:

            real_label = torch.tensor(1.0).expand(self.real_A.shape[0], 1).cuda()
            fake_label = torch.tensor(1.0).expand(self.real_A.shape[0], 1).cuda()
            
            # ========================================================
            # 整幅图像

            real_imgs = []
            for i in range(len(self.real_A)):
                index = torch.rand(1)
                index = int(index * self.length_global)
                pic1 = self.voc_dataset_global.__getitem__(index)
                pic1 = pic1.unsqueeze(0).cuda()
                real_imgs.append(pic1)
            real_imgs = torch.cat(real_imgs, 0)
            self.real_B = real_imgs
            self.fake_A = self.netG_B(self.real_B)
            
            self.rec_A = self.netG_B(self.fake_B)
            self.rec_B = self.netG_A(self.fake_A)
            
            # save_image(self.fake_A[0], '@fuhao/transfer/fake_A.jpg', normalize=True)
            # save_image(self.fake_B[0], '@fuhao/transfer/fake_B.jpg', normalize=True)
            # save_image(self.rec_A[0], '@fuhao/transfer/rec_A.jpg', normalize=True)
            # save_image(self.rec_B[0], '@fuhao/transfer/rec_B.jpg', normalize=True)
            
            # save_image(self.real_A[0], '@fuhao/transfer/real_A.jpg', normalize=True)
            # save_image(self.real_B[0], '@fuhao/transfer/real_B.jpg', normalize=True)
            # ========================================================
        
            # if self.iter % 20 != 0 and self.iter > 100:
            # if True:
            if self.iter % 25 != 0:
                # print(self.real_A.max(), self.real_A.min(), self.real_B.max(), self.real_B.min(), self.fake_B.max(), self.fake_B.min())
                # print()
                # 训练生成器
                # ========================================================
                # 训练生成器的时候，要把判别器冻结
                self.set_requires_grad([self.netD_A, self.netD_B], False)
                # # ========================================================
                
                # ========================================================
                # 训练整幅图像
                self.idt_A = self.netG_A(self.real_B)
                self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) 
                # G_B should be identity if real_A is fed: ||G_B(A) - A||
                self.idt_B = self.netG_B(self.real_A)
                self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) 

                self.loss_G_A = self.adversarial_loss(self.netD_A(self.fake_B), real_label)
                # GAN loss D_B(G_B(B))
                self.loss_G_B = self.adversarial_loss(self.netD_B(self.fake_A), real_label)
                # Forward cycle loss || G_B(G_A(A)) - A||
                self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A)
                # Backward cycle loss || G_A(G_B(B)) - B||
                self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B)
                
                # ========================================================

                loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A * 10.  + self.loss_cycle_B * 10.  + self.loss_idt_A * 5.  + self.loss_idt_B * 5. 
                # loss_G = self.loss_cycle_A * 10. + self.loss_cycle_B * 10. + self.loss_idt_A + self.idt_B
                loss = loss_G
                self.iter += 1
                # ========================================================
                # 解冻
                self.set_requires_grad([self.netD_A, self.netD_B], True)
                # ========================================================
            else:
                
                # ========================================================
                # 训练整幅图像
                gen_preds_v = self.netD_A(self.real_B)
                real_B_loss = self.adversarial_loss(gen_preds_v, real_label)
                # gen_preds_f = self.netD_A(self.fake_B.detach())
                # fake_loss_g = self.adversarial_loss(gen_preds_f, fake_label)
                # d_loss1 = (real_loss_g + fake_loss_g) / 2
                
                preds_v = self.netD_B(self.real_A)
                real_A_loss = self.adversarial_loss(preds_v, real_label)
                # preds_f = self.netD_B(self.fake_A.detach())
                # fake_loss_g = self.adversarial_loss(preds_f, fake_label)
                # d_loss2 = (real_loss_g + fake_loss_g) / 2
                # # ========================================================
                
                loss_D = real_B_loss + real_A_loss
                loss = loss_D
                self.iter += 1

                # print('netD + ', loss_D.data, 'real_A:', preds_v.data, 'fake_A:', preds_f.data, 'real_B:', gen_preds_v, 'fake_B:', gen_preds_f)

        
        # loss = loss_G + loss_D

        if self.train_detector:
            x = self.backbone(self.fake_B)
            if self.with_neck:
                x = self.neck(x)
            return x, loss
        else:
            return None, loss

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
        
        x, loss = self.extract_feat(img, img_metas, gt_bboxes)
        
        if self.train_detector:
            losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                                gt_labels, gt_bboxes_ignore, 
                                                label_mask=None
                                                    )
        else:
            losses = {}
        if loss != None:
            losses.update({'cyc_loss':loss})
            # losses.update({'cyc_loss_D':loss_D})

        
        return losses

    def simple_test(self, img, img_metas, rescale=False):

        x = self.extract_feat_test(img)
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
class RetinaNet_OriGAN(SingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_detector=True,
                 bbox_head_1=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(RetinaNet_OriGAN, self).__init__(backbone, neck, bbox_head, bbox_head_1, train_cfg,
                                        test_cfg, pretrained)
        
        self.train_detector = train_detector
        
        self.voc_dataset = MyDataset('../datasets/coco2017/patched/')
        self.length = len(self.voc_dataset)
        
        self.voc_dataset_global = MyDataset('../datasets/coco2017/val2017/')
        self.length_global = len(self.voc_dataset_global)
        # self.net_g = RRDNet()
        # self.net_d = MyVgg16()
        self.netG_A = RRDNet()
        self.netG_A.cuda()
        self.netG_B = RRDNet()
        self.netG_B.cuda()
        
        self.netD_A = MyVgg16()
        self.netD_A.cuda()
        self.netD_B = MyVgg16()
        self.netD_B.cuda()
        
        self.spa_loss = L_spa8()
        self.color_loss = L_color()
        self.tv_loss = L_TV()
        self.Tensor = torch.cuda.FloatTensor
        self.adversarial_loss = torch.nn.BCELoss()
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionCycle.cuda()
        self.criterionIdt = torch.nn.L1Loss()
        self.criterionIdt.cuda()
        
        # self.net_g.cuda()
        # self.net_d.cuda()
        self.adversarial_loss.cuda()
        self.spa_loss.cuda()
        self.color_loss.cuda()
        self.tv_loss.cuda()
        
        self.iter = 0
        self.train_D = 1
        
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def extract_feat_test(self, img):
        """Directly extract features from the backbone+neck."""
        
        img = self.netG_A(img)
                
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)

        return x

    def extract_feat(self, img, img_metas, gt_bboxes):
        """Directly extract features from the backbone+neck."""
        ori_img = img
        img = self.netG_A(ori_img)

        N, C, H, W = img.shape
        
        g_loss = None
        d_loss = None
        loss = None
        if len(ori_img) != 0:
            
            self.real_A = ori_img
            self.fake_B = img
            # save_image(patch_gen[0], '@fuhao/transfer/patch_gen.jpg', normalize=True)
            save_image(img[0], '@fuhao/transfer/img_gen.jpg', normalize=True)

            valid = torch.ones([ori_img.shape[0], 1]).cuda()
            fake = torch.zeros([ori_img.shape[0], 1]).cuda() 

            real_imgs = []
            for i in range(len(ori_img)):
                index = torch.rand(1)

                index = int(index * self.length_global)
                pic1 = self.voc_dataset_global.__getitem__(index)
                pic1 = pic1.unsqueeze(0).cuda()
                real_imgs.append(pic1)
            real_imgs = torch.cat(real_imgs, 0)
            self.real_B_global = real_imgs
            self.real_A_global = ori_img
            self.fake_B_global = img

            if True:
                
                # 训练生成器
                # ========================================================
                # 训练生成器的时候，要把判别器冻结
                self.set_requires_grad([self.netD_A, self.netD_B], False)
                # # ========================================================
                
                # ========================================================
                # 训练整幅图像
                self.loss_G_A_g = self.adversarial_loss(self.netD_A(self.fake_B_global), valid)
                # ========================================================

                loss_G = self.loss_G_A_g.mean() 

                # ========================================================
                # 解冻
                self.set_requires_grad([self.netD_A, self.netD_B], True)
                # ========================================================
                
                # ========================================================
                # 训练整幅图像
                
                preds_v = self.netD_B(self.real_B_global)
                real_loss_g = self.adversarial_loss(preds_v, valid)
                preds_f = self.netD_B(self.fake_B_global.detach())
                fake_loss_g = self.adversarial_loss(preds_f, fake)
                d_loss = (real_loss_g + fake_loss_g) / 2
                # # ========================================================
                
                loss_D = d_loss.mean()
                
                loss = loss_D
                print('netD + ', loss.data, 'pred_v:', preds_v.data, 'pred_f:', preds_f.data)
                # print('d loss => ', loss)
            # real_imgs.cuda()
            # Measure discriminator's ability to classify real from generated samples
            # print(type(real_imgs), type(valid))
            # real_loss = self.adversarial_loss(self.net_d(real_imgs), valid)
            # fake_loss = self.adversarial_loss(self.net_d(patch_gen.detach()), fake)
            # d_loss = (real_loss + fake_loss) / 2
        
        loss = loss_G + loss_D
        # spa_loss = self.spa_loss(ori_img, img) * 20
        # color_loss = self.color_loss(img) 
        # color_loss = 5 * torch.mean(color_loss) * 1000
        if self.train_detector:
            x = self.backbone(img)
            if self.with_neck:
                x = self.neck(x)
            return x, loss
        else:
            return None, loss

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
        
        x, loss = self.extract_feat(img, img_metas, gt_bboxes)
        
        if self.train_detector:
            losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                                gt_labels, gt_bboxes_ignore, 
                                                label_mask=None
                                                    )
        else:
            losses = {}
        if loss != None:
            losses.update({'cyc_loss':loss})
            # losses.update({'cyc_loss_D':loss_D})

        
        return losses

    def simple_test(self, img, img_metas, rescale=False):

        x = self.extract_feat_test(img)
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