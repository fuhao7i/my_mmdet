from locale import ABDAY_3, normalize
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.builder import BACKBONES
from mmdet.models.utils import ResLayer
from mmdet.models.backbones.resnet import ResNet

import sys
sys.path.append('@fuhao/')
from custom_mmdet.models.backbones.resnet.resnet18 import ResNet18_FPN, ResNet18_FPN_Cls, ResNet50_FPN
# from custom_mmdet.models.backbones.stem.scene_mining import *

import torch.nn.functional as F
import seaborn as sns
sns.set(font_scale=1.5)

from torchvision.utils import save_image
import numpy as np
import cv2
import time






def color_c(C1, C2, C3, c1, c2, c3, a1, a2):
    
    # C1_c = C1 + c1 * (c3 - c1)*(255-c1) * C3
    # C1_c = np.array(C1_c.reshape(C3.shape), np.uint8)

    # C2_c = C2 + c2 * (c3 - c2)*(255-c2) * C3
    # C2_c = np.array(C2_c.reshape(C3.shape), np.uint8)

    # C3 = np.array(C3, np.uint8)

    # print(C1.shape, a1.shape, c3.shape, c1.shape, C3.shape)
    C1_c = C1 + a1 * (c3 - c1)*(1-c1) * C3
    # C1_c = np.array(C1_c.reshape(C3.shape), np.uint8)

    C2_c = C2 + a2 * (c3 - c2)*(1-c2) * C3
    # C2_c = np.array(C2_c.reshape(C3.shape), np.uint8)

    # C3 = np.array(C3, np.uint8)

    # C1_c = C3 * 0.8
    # C1_c = np.array(C1_c.reshape(C3.shape), np.uint8)
    # C2_c = C3 * 0.9
    # C2_c = np.array(C2_c.reshape(C3.shape), np.uint8)

    # C3 = np.array(C3, np.uint8)

    return C1_c, C2_c, C3






@BACKBONES.register_module()
class ResNet_v511(ResNet):
    
    def __init__(self,
                 depth,
                 conv_cfg=None,
                 **kwargs
                ):
        super(ResNet_v511, self).__init__(depth, **kwargs)
        

        self.trans = ResNet18_FPN_Cls(cls_nums= 5)

        self.align = nn.Conv2d(256, 64, 1, 1, 0)

        self.mask = nn.Conv2d(256, 1, 1, 1, 0)

        self.conv_cfg = conv_cfg

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        input_cls = x

        trans_outs, trans_cls = self.trans(input_cls)
        # print('trans_cls shape: ', trans_cls.shape)

        input = input_cls.detach()
        """1. color compensation"""
        R = input[:, 0, :, :]
        G = input[:, 1, :, :]
        B = input[:, 2, :, :]

        Irm = R.mean(1).mean(1)
        Igm = G.mean(1).mean(1)
        Ibm = B.mean(1).mean(1)

        for i in range(input.shape[0]):
            Ic = max([Irm[i], Igm[i], Ibm[i]])
            Ic_index = [Irm[i], Igm[i], Ibm[i]].index(Ic)
            
            a1 = self.sigmoid(trans_cls[i][0])
            a2 = self.sigmoid(trans_cls[i][1])

            if Ic_index == 0:

                input[i][1] = input[i][1] + a1 * (Irm[i] - Igm[i])*(1-Igm[i]) * input[i][0]
                input[i][2] = input[i][2] + a2 * (Irm[i] - Ibm[i])*(1-Ibm[i]) * input[i][0]

                # color_c(input[i][1], input[i][2], input[i][0], Igm[i], Ibm[i], Irm[i], a1, a2)
                # G[i], B[i], R[i] = color_c(G[i], B[i], R[i], Igm[i], Ibm[i], Irm[i], a1, a2)
                # img = cv2.merge([R[i], G[i], B[i]])
                # show(img, "color_compensation by R channel")

            elif Ic_index == 1:
                input[i][0] = input[i][0] + a1 * (Igm[i] - Irm[i])*(1-Irm[i]) * input[i][1]
                input[i][2] = input[i][2] + a2 * (Igm[i] - Ibm[i])*(1-Ibm[i]) * input[i][1]

                # color_c(input[i][0], input[i][2], input[i][1], Igm[i], Ibm[i], Irm[i], a1, a2)
                # R, B, G = color_c(R, B, G, Irm, Ibm, Igm, a1, a2)
                # img = cv2.merge([R, G, B])
                # show(img, "color_compensation by G channel")
     
            else:
                input[i][0] = input[i][0] + a1 * (Ibm[i] - Irm[i])*(1-Irm[i]) * input[i][2]
                input[i][1] = input[i][1] + a2 * (Ibm[i] - Igm[i])*(1-Igm[i]) * input[i][2]
        
        # save_image(input, "@fuhao/transfer/CC.jpg" , normalize=True)
        """2. white balance"""
        # trans_cls[:, 2:5] = torch.abs(trans_cls[:, 2:5])
        # trans_cls[:, 2:5] = trans_cls[:, 2:5] + 1.0

        # for i in range(input.shape[0]):
        #     input[i, 0, :, :] = input[i, 0, :, :] * trans_cls[i, 2]
        #     input[i, 1, :, :] = input[i, 1, :, :] * trans_cls[i, 3]
        #     input[i, 2, :, :] = input[i, 2, :, :] * trans_cls[i, 4]
        # print('trans cls => ', trans_cls)
        # save_image(input, "@fuhao/transfer/WB.jpg" , normalize=True)
        """3. enhancement + denoising"""

        """4. gamma correction"""

        """5. sharpen"""

        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)

        # neck = self.c2(data['feat'])
        # """Combine Method."""
        # align = self.align(mix2[0])
        # x = self.mixop(x, align)

        # # x = neck
        # """Over"""
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)


# -----------------------------------------
# MSRCR算法
# def get_gauss_kernel(sigma,dim=2):
#     '''1D gaussian function: G(x)=1/(sqrt{2π}σ)exp{-(x-μ)²/2σ²}. Herein, μ:=0, after 
#        normalizing the 1D kernel, we can get 2D kernel version by 
#        matmul(1D_kernel',1D_kernel), having same sigma in both directions. Note that 
#        if you want to blur one image with a 2-D gaussian filter, you should separate 
#        it into two steps(i.e. separate the 2-D filter into two 1-D filter, one column 
#        filter, one row filter): 1) blur image with first column filter, 2) blur the 
#        result image of 1) with the second row filter. Analyse the time complexity: if 
#        m&n is the shape of image, p&q is the size of 2-D filter, bluring image with 
#        2-D filter takes O(mnpq), but two-step method takes O(pmn+qmn)'''
#     ksize=int(np.floor(sigma*6)/2)*2+1 #kernel size("3-σ"法则) refer to 
#     #https://github.com/upcAutoLang/MSRCR-Restoration/blob/master/src/MSRCR.cpp
#     k_1D=np.arange(ksize)-ksize//2
#     k_1D=np.exp(-k_1D**2/(2*sigma**2))
#     k_1D=k_1D/np.sum(k_1D)
#     if dim==1:
#         return k_1D
#     elif dim==2:
#         return k_1D[:,None].dot(k_1D.reshape(1,-1))

# def gauss_blur(img,sigma):
#     '''suitable for 1 or 3 channel image'''
#     row_filter=get_gauss_kernel(sigma,1)
#     t=cv2.filter2D(img,-1,row_filter[...,None])
#     return cv2.filter2D(t,-1,row_filter.reshape(1,-1))

def simplest_color_balance(img_msrcr,s1,s2):
    '''see section 3.1 in “Simplest Color Balance”(doi: 10.5201/ipol.2011.llmps-scb). 
    Only suitable for 1-channel image'''
    img = img_msrcr
    sort_img = np.array(img.detach().cpu())
    sort_img = np.sort(sort_img, None)
    N=sort_img.size
    # N=img_msrcr.size(0) * img_msrcr.size(1)
    # print(sort_img.size())
    # print(img_msrcr.size(0) * img_msrcr.size(1))
    # print(sort_img)
    Vmin=sort_img[int(N*s1)]
    Vmax=sort_img[int(N*(1-s2))-1]
    img_msrcr = torch.clamp(img_msrcr, Vmin, Vmax)
    # img_msrcr[img_msrcr<Vmin]= Vmin
    # img_msrcr[img_msrcr>Vmax]= Vmax
    return (img_msrcr-Vmin)*255/(Vmax-Vmin)


# From https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/3
import torch
import math
import torch.nn as nn

def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,kernel_size=kernel_size, groups=channels, bias=False, padding=kernel_size//2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False
    
    return gaussian_filter

def gauss_blur(img, sigma, ksize=None):
    if ksize == None:
        ksize=int(np.floor(sigma*6)/2)*2+1
    blur_layer = get_gaussian_kernel(kernel_size=ksize, sigma=sigma).cuda()
    img = img.transpose(0, 2).unsqueeze(0)
    blured_img = blur_layer(img)
    blured_img = blured_img.squeeze(0).transpose(0, 2)
    return blured_img

def gauss_blur_v1(img, ksize=None, blur_layer=None):

    # img = img.transpose(0, 2).unsqueeze(0)
    blured_img = blur_layer(img)
    # blured_img = blured_img.squeeze(0).transpose(0, 2)
    return blured_img


def MultiScaleRetinex(img,sigmas=[15,80,250],blur_layer0=None, blur_layer1=None, blur_layer2=None, weights=None,flag=False):
    '''equal to func retinex_MSR, just remove the outer for-loop. Practice has proven 
       that when MSR used in MSRCR or Gimp, we should add stretch step, otherwise the 
       result color may be dim. But it's up to you, if you select to neglect stretch, 
       set flag as False, have fun'''

    st = time.time()
    # if weights==None:
    #     weights=torch.ones(len(sigmas))/len(sigmas)
    # elif not abs(sum(weights)-1)<0.00001:
    #     raise ValueError('sum of weights must be 1!')

    weights = torch.tensor([0.33, 0.33, 0.33]).cuda()
    # print(img.shape)
    # r=torch.zeros(img.shape).cuda()
    
    st1 = time.time()
    print('h2', st1 - st)
    # r=torch.zeros(img.shape,dtype='torch.float')
    # img=img.astype('float')

    # for i,sigma in enumerate(sigmas):
    #     r+=(torch.log(img+1)-torch.log(gauss_blur(img,sigma)+1))*weights[i]
    f1 = time.time()
    r = (torch.log(img+1)-torch.log(gauss_blur_v1(img,blur_layer=blur_layer0)+1))*weights[0]
    r += (torch.log(img+1)-torch.log(gauss_blur_v1(img,blur_layer=blur_layer1)+1))*weights[1]
    r += (torch.log(img+1)-torch.log(gauss_blur_v1(img,blur_layer=blur_layer2)+1))*weights[2]
    f2 = time.time()
    print('MSR: ', f2 - f1)
    # if flag:
    #     mmin=torch.min(r,dim=0,keepdims=True).values
    #     mmin=torch.min(mmin,dim=1,keepdims=True).values
    #     # print(mmin)
    #     # print(mmin.shape)
    #     mmax=torch.max(r,dim=0,keepdims=True).values
    #     mmax=torch.max(mmax,dim=1,keepdims=True).values
    #     r=(r-mmin)/(mmax-mmin)*255 #maybe indispensable when used in MSRCR or Gimp, make pic vibrant
        # r=r.astype('float')
    return r

def MSR(img,sigmas=[15,80,250],blur_layer0=None, blur_layer1=None, blur_layer2=None, weights=None,flag=False):
    '''equal to func retinex_MSR, just remove the outer for-loop. Practice has proven 
       that when MSR used in MSRCR or Gimp, we should add stretch step, otherwise the 
       result color may be dim. But it's up to you, if you select to neglect stretch, 
       set flag as False, have fun'''

    r1 = (torch.log(img+1)-torch.log(gauss_blur_v1(img,blur_layer=blur_layer0)+1))
    r2 = (torch.log(img+1)-torch.log(gauss_blur_v1(img,blur_layer=blur_layer1)+1))
    r3 = (torch.log(img+1)-torch.log(gauss_blur_v1(img,blur_layer=blur_layer2)+1))

    # if flag:
    #     mmin=torch.min(r,dim=0,keepdims=True).values
    #     mmin=torch.min(mmin,dim=1,keepdims=True).values
    #     # print(mmin)
    #     # print(mmin.shape)
    #     mmax=torch.max(r,dim=0,keepdims=True).values
    #     mmax=torch.max(mmax,dim=1,keepdims=True).values
    #     r=(r-mmin)/(mmax-mmin)*255 #maybe indispensable when used in MSRCR or Gimp, make pic vibrant
        # r=r.astype('float')
    return r1, r2, r3

def retinex_MSRCR(img,sigmas=[12,80,250],s1=0.01,s2=0.01, blur_layer0=None, blur_layer1=None, blur_layer2=None):
    '''r=βlog(αI')MSR, I'=I/∑I, I is one channel of image, ∑I is the sum of all channels, 
       C:=βlog(αI') is named as color recovery factor. Last we improve previously used 
       linear stretch: MSRCR:=r, r=G[MSRCR-b], then doing linear stretch. In practice, it 
       doesn't work well, so we take another measure: Simplest Color Balance'''

    img = img.squeeze(0)
    
    alpha=125
    img=img + 1. #
    csum_log=torch.log(torch.sum(img,axis=2))
    st = time.time()
    msr=MultiScaleRetinex(img-1,sigmas, blur_layer0, blur_layer1, blur_layer2) #-1
    f1 = time.time()
    print('msr', f1 - st)
    # r=(torch.log(alpha*img)-csum_log[...,None])*msr
    f2 = time.time()
    print('h1', f2-f1)
    #beta=46;G=192;b=-30;r=G*(beta*r-b) #deprecated
    #mmin,mmax=np.min(r),np.max(r)
    #stretch=(r-mmin)/(mmax-mmin)*255 #linear stretch is unsatisfactory

    # st = time.time()
    # for i in range(r.shape[-1]):
    #     r[...,i]=simplest_color_balance(r[...,i],0.01,0.01)
    # f1 = time.time()
    # print('simplest color balance: ', f1 - st, 's')

    # mmin=torch.min(r,dim=0,keepdims=True).values
    # mmin=torch.min(mmin,dim=1,keepdims=True).values
    # # print(mmin)
    # # print(mmin.shape)
    # mmax=torch.max(r,dim=0,keepdims=True).values
    # mmax=torch.max(mmax,dim=1,keepdims=True).values
    # print(mmin, mmax)
    # r=(r-mmin)/(mmax-mmin)*255 #maybe indispensable when used in MSRCR or Gimp, make pic vibrant

    img = r.unsqueeze(0)
    return img

# -----------------------------------------





# @BACKBONES.register_module()
# class ResNet_v512(ResNet):
    
#     def __init__(self,
#                  depth,
#                  **kwargs
#                 ):
#         super(ResNet_v512, self).__init__(depth, **kwargs)
#         self.mean= torch.tensor([123.675, 116.28, 103.53]).cuda()
#         self.std= torch.tensor([58.395, 57.12, 57.375]).cuda()

#         sigmas=[12,80,250]
#         ksize0=int(np.floor(sigmas[0]*6)/2)*2+1
#         self.blur_layer0 = get_gaussian_kernel(kernel_size=ksize0, sigma=sigmas[0]).cuda()

#         ksize1=int(np.floor(sigmas[1]*6)/2)*2+1
#         self.blur_layer1 = get_gaussian_kernel(kernel_size=ksize1, sigma=sigmas[1]).cuda()

#         ksize2=int(np.floor(sigmas[2]*6)/2)*2+1
#         self.blur_layer2 = get_gaussian_kernel(kernel_size=ksize2, sigma=sigmas[2]).cuda()

#     def forward(self, x):
        
#         # print(x)
#         # x : b, c, w, h
#         # 1. MSRCR算法
#         x = x.transpose(1, 3)
#         x = (x - self.mean) / self.std
#         # x: b, h, w, c
#         # x = MultiScaleRetinex(x)
#         st = time.time()
#         # for i in range(x.shape[0]):
#             # x[i, ...] = retinex_MSRCR(x[i, ...], blur_layer0=self.blur_layer0, blur_layer1=self.blur_layer1, blur_layer2=self.blur_layer2)

#         sigmas=[12,80,250]
#         x[0] = x[0].squeeze(0)
#         alpha=125
#         x[0]=x[0] + 1. #
#         csum_log=torch.log(torch.sum(x[0],axis=2))
#         st = time.time()
#         msr=MultiScaleRetinex(x[0]-1,sigmas, self.blur_layer0, self.blur_layer1, self.blur_layer2) #-1
#         f1 = time.time()
#         print('msr', f1 - st)
#         r=(torch.log(alpha*x[0])-csum_log[...,None])*msr
#         x[0] = r.unsqueeze(0)

#         # x[1] = x[1].squeeze(0)
#         # alpha=125
#         # x[1]=x[1] + 1. #
#         # csum_log=torch.log(torch.sum(x[1],axis=2))
#         # st = time.time()
#         # msr=MultiScaleRetinex(x[1]-1,sigmas, self.blur_layer0, self.blur_layer1, self.blur_layer2) #-1
#         # f1 = time.time()
#         # print('msr', f1 - st)
#         # r=(torch.log(alpha*x[1])-csum_log[...,None])*msr
#         # x[1] = r.unsqueeze(0)



#         # x = x.transpose(1, 3)
#         f1 = time.time()
#         print('MSRCR: ', f1 - st)
#         # raise
#         # 2. 归一化到[-1, 1]
#         # print(x.max(), x.min())
#         # x = x.transpose(1, 3)
#         # x = (x - self.mean) / self.std
#         x = x.transpose(1, 3)

#         # from torchvision.utils import save_image
#         # save_image(x, "@fuhao/transfer/MSRCR_input_norm.png" , normalize=True)
#         # save_image(x, "@fuhao/transfer/MSRCR_input.png" , normalize=False)


#         """Forward function."""
#         if self.deep_stem:
#             x = self.stem(x)
#         else:
#             x = self.conv1(x)
#             x = self.norm1(x)
#             x = self.relu(x)
#         x = self.maxpool(x)

#         # x = self.partition(x)
#         # x = self.c1(x)

#         outs = []
#         for i, layer_name in enumerate(self.res_layers):
#             res_layer = getattr(self, layer_name)
#             x = res_layer(x)
#             if i in self.out_indices:
#                 outs.append(x)
#         return tuple(outs)


@BACKBONES.register_module()
class ResNet_v512(ResNet):
    
    def __init__(self,
                 depth,
                 conv_cfg=None,
                 **kwargs
                ):
        super(ResNet_v512, self).__init__(depth, **kwargs)
        

        self.trans = ResNet18_FPN_Cls(cls_nums= 3)

        self.conv_cfg = conv_cfg

        self.mean= torch.tensor([123.675, 116.28, 103.53]).cuda()
        self.std= torch.tensor([58.395, 57.12, 57.375]).cuda()

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

        self.sigmas=[12,80,250]
        ksize0=int(np.floor(self.sigmas[0]*6)/2)*2+1
        self.blur_layer0 = get_gaussian_kernel(kernel_size=ksize0, sigma=self.sigmas[0])

        ksize1=int(np.floor(self.sigmas[1]*6)/2)*2+1
        self.blur_layer1 = get_gaussian_kernel(kernel_size=ksize1, sigma=self.sigmas[1])

        ksize2=int(np.floor(self.sigmas[2]*6)/2)*2+1
        self.blur_layer2 = get_gaussian_kernel(kernel_size=ksize2, sigma=self.sigmas[2])

        self.sigmas=[12,80,250]
        ksize0=int(np.floor(self.sigmas[0]*6)/2)*2+1
        self.blur_layer0 = get_gaussian_kernel(kernel_size=3, sigma=2).cuda()

        ksize1=int(np.floor(self.sigmas[1]*6)/2)*2+1
        self.blur_layer1 = get_gaussian_kernel(kernel_size=13, sigma=4).cuda()

        ksize2=int(np.floor(self.sigmas[2]*6)/2)*2+1
        self.blur_layer2 = get_gaussian_kernel(kernel_size=25, sigma=8).cuda()




    def forward(self, x):

        # st = time.time()
        input_cls = x.clone()

        input_cls = input_cls.transpose(1, 3)
        input_cls = (input_cls - self.mean) / self.std
        input_cls = input_cls.transpose(1, 3)

        trans_outs, trans_cls = self.trans(input_cls)
        # print('trans_cls shape: ', trans_cls.shape)

        """1. MSR"""
        trans_cls = self.softmax(trans_cls)
        # print(trans_cls, trans_cls.shape, trans_cls[..., 0], trans_cls[..., 0].shape)
        # print(torch.max(x), torch.min(x))

        with torch.no_grad():
            r0, r1, r2 = MSR(x,self.sigmas, self.blur_layer0, self.blur_layer1, self.blur_layer2)
        # print(r0.shape)
        # print(torch.max(r0))
        x = trans_cls[..., 0].unsqueeze(1).unsqueeze(1).unsqueeze(1)  * r0 + trans_cls[..., 1].unsqueeze(1).unsqueeze(1).unsqueeze(1) * r1 + trans_cls[..., 2].unsqueeze(1).unsqueeze(1).unsqueeze(1)  * r2
        # print(torch.min(x), torch.max(x))
        x = x.transpose(1, 3)
        x = (x - self.mean) / self.std
        x = x.transpose(1, 3)

        # ed = time.time()
        # print('used time:', ed - st, 's', torch.max(input_cls), torch.max(x), torch.min(x))
        # print(torch.min(x), torch.max(x))
        """2. white balance"""

        """3. enhancement + denoising"""

        """4. gamma correction"""

        """5. sharpen"""

        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)

        # """Over"""
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

# -----------------------------------------------
# simple net

class k3_net(nn.Module):
    
    def __init__(self):
        super(k3_net, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1),

        )

    def forward(self, x):
        x = self.net(x)
        x = torch.sigmoid(x)
        return x
    
class k5_net(nn.Module):
    
    def __init__(self):
        super(k5_net, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 5, 1, 2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.ReLU(),
            nn.Conv2d(64, 32, 5, 1, 2),
            nn.ReLU(),
            nn.Conv2d(32, 3, 5, 1, 2),

        )

    def forward(self, x):
        x = self.net(x)
        x = torch.sigmoid(x)
        return x

class k7_net(nn.Module):
    
    def __init__(self):
        super(k7_net, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 7, 1, 3),
            nn.ReLU(),
            nn.Conv2d(16, 32, 7, 1, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7, 1, 3),
            nn.ReLU(),
            nn.Conv2d(64, 32, 7, 1, 3),
            nn.ReLU(),
            nn.Conv2d(32, 3, 7, 1, 3),

        )

    def forward(self, x):
        x = self.net(x)
        x = torch.sigmoid(x)
        return x
    

class noise_net(nn.Module):
    
    def __init__(self):
        super(noise_net, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1),

        )

    def forward(self, x):
        x = self.net(x)
        x = torch.tanh(x)
        return x
# -----------------------------------------------






@BACKBONES.register_module()
class ResNet_v513(ResNet):
    
    def __init__(self,
                 depth,
                 conv_cfg=None,
                 **kwargs
                ):
        super(ResNet_v513, self).__init__(depth, **kwargs)
        

        self.trans = ResNet18_FPN_Cls(cls_nums= 3)

        self.conv_cfg = conv_cfg

        self.mean= torch.tensor([123.675, 116.28, 103.53]).cuda()
        self.std= torch.tensor([58.395, 57.12, 57.375]).cuda()

        self.k3 = k3_net()
        self.k5 = k5_net()
        self.k7 = k7_net()
        self.noisy = k3_net()
        

    def forward(self, x):
        # save_image(x, "@fuhao/transfer/input.jpg" , normalize=True) 
        _, _, w, h = x.shape
        # x = x / 255.

        # x ~ [0, 1]
        # k3_img = self.k3(x) + 1
        # k5_img = self.k5(x) + 1
        # k7_img = self.k7(x) + 1
        # noisy = self.noisy(x)
        
        # x = torch.log(torch.sigmoid(x - noisy) + 1)
        
        # x = (x - torch.log(k3_img)) * 0.333 + (x - torch.log(k5_img)) * 0.333 + (x - torch.log(k7_img)) * 0.333
        # x = x * 255.
        # x = x.transpose(1, 3)
        # x = (x - self.mean) / self.std
        # x = x.transpose(1, 3)

        """2. """
        x = x / 255.
        x_mini = x.clone()
        x_mini = nn.functional.interpolate(x_mini, scale_factor=0.25, mode='bilinear', align_corners=False)
        # print(x_mini.shape, x.shape)
        
        k3_img = self.k3(x_mini) + 1
        k5_img = self.k5(x_mini) + 1
        k7_img = self.k7(x_mini) + 1
        noisy = self.noisy(x)
        
        x = torch.log(torch.sigmoid(x - noisy) + 1)
        
        k3_img = nn.functional.interpolate(k3_img, size=[w, h], mode='bilinear', align_corners=False)
        k5_img = nn.functional.interpolate(k5_img, size=[w, h], mode='bilinear', align_corners=False)
        k7_img = nn.functional.interpolate(k7_img, size=[w, h], mode='bilinear', align_corners=False)
        
        x = (x - torch.log(k3_img)) * 0.333 + (x - torch.log(k5_img)) * 0.333 + (x - torch.log(k7_img)) * 0.333
        x = x * 255.
        
        x = x.transpose(1, 3)
        x = (x - self.mean) / self.std
        x = x.transpose(1, 3)
        
        """3. """
        # x_mini = x.clone()
        # x_mini = x_mini / 255.
        # x_mini = nn.functional.interpolate(x_mini, scale_factor=0.25, mode='bilinear', align_corners=False)
        # # print(x_mini.shape, x.shape)
        
        # k3_img = self.k3(x_mini) * 255. + 1.
        # k5_img = self.k5(x_mini) * 255. + 1.
        # k7_img = self.k7(x_mini) * 255. + 1.
        # noisy = self.noisy(x) * 255.
        
        # x = torch.log(torch.sigmoid(x - noisy) + 1)
        
        # k3_img = nn.functional.interpolate(k3_img, size=[w, h], mode='bilinear', align_corners=False)
        # k5_img = nn.functional.interpolate(k5_img, size=[w, h], mode='bilinear', align_corners=False)
        # k7_img = nn.functional.interpolate(k7_img, size=[w, h], mode='bilinear', align_corners=False)
        
        # x = (x - torch.log(k3_img)) * 0.333 + (x - torch.log(k5_img)) * 0.333 + (x - torch.log(k7_img)) * 0.333
        
        # x = x.transpose(1, 3)
        # x = (x - self.mean) / self.std
        # x = x.transpose(1, 3)
        
        # save_image(x, "@fuhao/transfer/CC.jpg" , normalize=True)     
        # raise
        """1. MSR"""

        """2. white balance"""

        """3. enhancement + denoising"""

        """4. gamma correction"""

        """5. sharpen"""

        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)

        # """Over"""
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)




@BACKBONES.register_module()
class ResNet_v514(ResNet):
    
    def __init__(self,
                 depth,
                 conv_cfg=None,
                 **kwargs
                ):
        super(ResNet_v514, self).__init__(depth, **kwargs)
        

        self.trans = ResNet18_FPN_Cls(cls_nums= 3)

        self.conv_cfg = conv_cfg

        self.mean= torch.tensor([123.675, 116.28, 103.53]).cuda()
        self.std= torch.tensor([58.395, 57.12, 57.375]).cuda()

        self.k3 = k3_net()
        self.k5 = k5_net()
        self.k7 = k7_net()
        # self.noise = noise_net()
        self.noisy = noise_net()
                

    def forward(self, x):
        # save_image(x, "@fuhao/transfer/input.jpg" , normalize=True) 
        _, _, w, h = x.shape
        # x = x / 255.

        # x ~ [0, 1]
        # k3_img = self.k3(x) + 1
        # k5_img = self.k5(x) + 1
        # k7_img = self.k7(x) + 1
        # noisy = self.noisy(x)
        
        # x = torch.log(torch.sigmoid(x - noisy) + 1)
        
        # x = (x - torch.log(k3_img)) * 0.333 + (x - torch.log(k5_img)) * 0.333 + (x - torch.log(k7_img)) * 0.333
        # x = x * 255.
        # x = x.transpose(1, 3)
        # x = (x - self.mean) / self.std
        # x = x.transpose(1, 3)

        """2. """
        # # 归一化到 [0, 1]
        # # x = x / 255.
        # mmax = torch.max(torch.max(x, dim=3, keepdim=True).values, dim=2, keepdim=True).values
        # mmin = torch.min(torch.min(x, dim=3, keepdim=True).values, dim=2, keepdim=True).values
        # x = (x - mmin) / (mmax - mmin)
        
        # x_mini = x.clone()
        # x_mini = nn.functional.interpolate(x_mini, scale_factor=0.25, mode='bilinear', align_corners=False)
        # # print(x_mini.shape, x.shape)
        
        # k3_img = self.k3(x_mini) + 1
        # k5_img = self.k5(x_mini) + 1
        # k7_img = self.k7(x_mini) + 1
        # noisy = self.noisy(x)
        # print(torch.min(k3_img), torch.min(k5_img), torch.min(k7_img))
        # x = torch.log(torch.sigmoid(x - noisy) + 1)
        
        # k3_img = nn.functional.interpolate(k3_img, size=[w, h], mode='bilinear', align_corners=False)
        # k5_img = nn.functional.interpolate(k5_img, size=[w, h], mode='bilinear', align_corners=False)
        # k7_img = nn.functional.interpolate(k7_img, size=[w, h], mode='bilinear', align_corners=False)
        
        # x = (x - torch.log(k3_img)) * 0.333 + (x - torch.log(k5_img)) * 0.333 + (x - torch.log(k7_img)) * 0.333

        # # 归一化到 [0, 1]
        # # x = x.transpose(1, 3)
        # mmax = torch.max(torch.max(x, dim=3, keepdim=True).values, dim=2, keepdim=True).values
        # mmin = torch.min(torch.min(x, dim=3, keepdim=True).values, dim=2, keepdim=True).values
        # x = (x - mmin) / (mmax - mmin)
        # # x = x.transpose(1, 3)      

        # x = x * 255.
        
        # # gamma correction
        # gamma = 1.4
        # # gc = torch.pow(x / 255., gamma)
        # # save_image(gc, "@fuhao/transfer/gamma_correction.jpg" , normalize=True)
        
        # x = torch.pow(x / 255., gamma) * 255
        # # save_image(x, "@fuhao/transfer/gamma_correction.jpg" , normalize=True)
        
        # x = x.transpose(1, 3)
        # x = (x - self.mean) / self.std
        # x = x.transpose(1, 3)
        
        """3. """
        # x_mini = x.clone()
        # x_mini = x_mini / 255.
        # x_mini = nn.functional.interpolate(x_mini, scale_factor=0.25, mode='bilinear', align_corners=False)
        # # print(x_mini.shape, x.shape)
        
        # k3_img = self.k3(x_mini) * 255. + 1.
        # k5_img = self.k5(x_mini) * 255. + 1.
        # k7_img = self.k7(x_mini) * 255. + 1.
        # noise = self.noisy(x) * 255.
        
        # x = torch.log( torch.sigmoid((x - noise) / 255.) * 255. + 1.)
        
        # k3_img = nn.functional.interpolate(k3_img, size=[w, h], mode='bilinear', align_corners=False)
        # k5_img = nn.functional.interpolate(k5_img, size=[w, h], mode='bilinear', align_corners=False)
        # k7_img = nn.functional.interpolate(k7_img, size=[w, h], mode='bilinear', align_corners=False)
        
        # x = (x - torch.log(k3_img)) * 0.333 + (x - torch.log(k5_img)) * 0.333 + (x - torch.log(k7_img)) * 0.333
        
        # x = x.transpose(1, 3)
        # x = (x - self.mean) / self.std
        # x = x.transpose(1, 3)
        mmax = torch.max(torch.max(x, dim=3, keepdim=True).values, dim=2, keepdim=True).values
        # mmin = torch.min(torch.min(x, dim=3, keepdim=True).values, dim=2, keepdim=True).values
        # x = (x - mmin) / (mmax - mmin)
        x = x / mmax
        
        x = torch.log(x + 1.)
        gamma = 1.4
        x = torch.pow(x, gamma)
        
        mmax = torch.max(torch.max(x, dim=3, keepdim=True).values, dim=2, keepdim=True).values
        mmin = torch.min(torch.min(x, dim=3, keepdim=True).values, dim=2, keepdim=True).values
        # x = (x - mmin) / (mmax - mmin) * 255. 
        
        x = x / mmax * 255.

        x = x.transpose(1, 3)
        x = (x - self.mean) / self.std
        x = x.transpose(1, 3)
        # save_image(x, "@fuhao/transfer/CC.jpg" , normalize=True)

        # raise
        """1. MSR"""

        """2. white balance"""

        """3. enhancement + denoising"""

        """4. gamma correction"""

        """5. sharpen"""

        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)

        # """Over"""
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)



class UWNet(nn.Module):
    def __init__(self, is_noise=False):
        super(UWNet, self).__init__()
        
        self.k3_net = nn.Sequential(
            nn.Conv2d(3, 3, 3, 1, 1, groups=3)
        )
        
        self.is_noise = is_noise
        if self.is_noise:
            self.noise_net = nn.Sequential(
                nn.Conv2d(3, 16, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(64, 32, 3, 1, 1),
                nn.ReLU(),
                nn.Conv2d(32, 3, 3, 1, 1),
                nn.ReLU()
            )

    def forward(self, input):
        k3_img = torch.sigmoid(self.k3_net(input))
        
        if self.is_noise:
            noise = self.noise_net(input)
            return k3_img, noise
        return k3_img


@BACKBONES.register_module()
class ResNet_v522(ResNet):
    
    def __init__(self,
                 depth,
                 conv_cfg=None,
                 **kwargs
                ):
        super(ResNet_v522, self).__init__(depth, **kwargs)
        self.conv_cfg = conv_cfg

        self.mean= torch.tensor([123.675, 116.28, 103.53]).cuda()
        self.std= torch.tensor([58.395, 57.12, 57.375]).cuda()

        self.net = UWNet()
        weights_path = '@fuhao/exp/500Adaptive/520/522color_correction/checkpoints/epoch_10000.pth'


        print('+---------------------------------------------+')
        print('+ Loading weights into ' + 'uwnet' + ' state dict.. +')
        print('+ Weights path: ', weights_path)
        print('+---------------------------------------------+')
        model_dict = self.net.state_dict()
        
        num = 0
        for k in model_dict.keys():
            print('model =>', k)
            num += 1
        print('total num: ', num)
        num = 0

        pretrained_dict = torch.load(weights_path)

        for k in pretrained_dict.keys():
            print('checkpoint =>', k)
            num += 1
        print('total num: ', num)
        num = 0

        momo_dict = {}
        for k, v in pretrained_dict.items(): 
            # if k in model_dict.keys() and cfg.load_checkpoint['load_keys'] in k: # 注意要修改的地方
            if k in model_dict.keys():
                if pretrained_dict[k].size() == model_dict[k].size():
                    momo_dict.update({k: v})

        # momo_dict = {k:v for 'module.'+k, v in pretrained_dict['state_dict'].items() if k in model_dict.keys() and pretrained_dict['state_dict'][k].shape() == model_dict[k].shape() }
        for k, v in momo_dict.items():
            print('model load => ', k)
            num += 1
        print('total num: ', num)

        model_dict.update(momo_dict)
        self.net.load_state_dict(model_dict)
        print('+---------------------------------------------+')
        print('+                 Finished！                  +')
        print('+---------------------------------------------+')
                

    def forward(self, x):

        _, _, w, h = x.shape

        k3_img = self.net(x)  # [1, c, h, w]
    
        x = torch.log(x * 255. + 1.) - torch.log(k3_img * 255.+ 1.)
        
        x = x.transpose(1, 3)
        x = (x - self.mean) / self.std
        x = x.transpose(1, 3)

        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)

        # """Over"""
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

@BACKBONES.register_module()
class ResNet_v523(ResNet):
    
    def __init__(self,
                 depth,
                 conv_cfg=None,
                 **kwargs
                ):
        super(ResNet_v523, self).__init__(depth, **kwargs)
        self.conv_cfg = conv_cfg

        self.mean= torch.tensor([123.675, 116.28, 103.53]).cuda()
        self.std= torch.tensor([58.395, 57.12, 57.375]).cuda()

        self.net = UWNet(is_noise=False)
        # self.net.eval()
        weights_path = '@fuhao/exp/500Adaptive/520/522color_correction/checkpoints/epoch_10000.pth'


        print('+---------------------------------------------+')
        print('+ Loading weights into ' + 'uwnet' + ' state dict.. +')
        print('+ Weights path: ', weights_path)
        print('+---------------------------------------------+')
        model_dict = self.net.state_dict()
        
        num = 0
        for k in model_dict.keys():
            print('model =>', k)
            num += 1
        print('total num: ', num)
        num = 0

        pretrained_dict = torch.load(weights_path)

        for k in pretrained_dict.keys():
            print('checkpoint =>', k)
            num += 1
        print('total num: ', num)
        num = 0

        momo_dict = {}
        for k, v in pretrained_dict.items(): 
            # if k in model_dict.keys() and cfg.load_checkpoint['load_keys'] in k: # 注意要修改的地方
            if k in model_dict.keys():
                if pretrained_dict[k].size() == model_dict[k].size():
                    momo_dict.update({k: v})

        # momo_dict = {k:v for 'module.'+k, v in pretrained_dict['state_dict'].items() if k in model_dict.keys() and pretrained_dict['state_dict'][k].shape() == model_dict[k].shape() }
        for k, v in momo_dict.items():
            print('model load => ', k)
            num += 1
        print('total num: ', num)

        model_dict.update(momo_dict)
        self.net.load_state_dict(model_dict)
        print('+---------------------------------------------+')
        print('+                 Finished！                  +')
        print('+---------------------------------------------+')
                

    def forward(self, x):
        # print('ori_x:', torch.max(x).data, torch.min(x).data, torch.mean(x).data)
        # mmax = torch.max(torch.max(x, dim=3, keepdim=True).values, dim=2, keepdim=True).values
        # mmin = torch.min(torch.min(x, dim=3, keepdim=True).values, dim=2, keepdim=True).values
        # x = (x - mmin) / (mmax - mmin)
        x = x / 255.
        
        # save_image(x, '@fuhao/transfer/input.jpg', normalize=True)
        _, _, w, h = x.shape

        k3_img = self.net(x)  # [1, c, h, w]

        x = torch.log(x * 255. + 1.) - torch.log(k3_img * 255.+ 1.)
        # x = torch.exp(x)
        # print('log_x:', torch.max(x).data, torch.min(x).data, torch.mean(x).data)
        # x = x - noise * 255.
        
        # x = x / 255.
        
        
        mmax = torch.max(torch.max(x, dim=3, keepdim=True).values, dim=2, keepdim=True).values
        mmin = torch.min(torch.min(x, dim=3, keepdim=True).values, dim=2, keepdim=True).values
        x = (x - mmin) / (mmax - mmin)
        
        x = torch.pow(x, 6.) * 255.
        # print('norm_x:', torch.max(x).data, torch.min(x).data, torch.mean(x).data)

        
        x = x.transpose(1, 3)
        x = (x - self.mean) / self.std
        x = x.transpose(1, 3)
        # save_image(x, '@fuhao/transfer/cc_deblur.jpg', normalize=True)

        """1. MSR"""

        """2. white balance"""

        """3. enhancement + denoising"""

        """4. gamma correction"""

        """5. sharpen"""

        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)

        # """Over"""
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

# ------------------------------------------------------------------------------------
# 高斯模糊
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


@BACKBONES.register_module()
class ResNet_v524(ResNet):
    
    def __init__(self,
                 depth,
                 conv_cfg=None,
                 **kwargs
                ):
        super(ResNet_v524, self).__init__(depth, **kwargs)
        self.conv_cfg = conv_cfg

        self.mean= torch.tensor([123.675, 116.28, 103.53]).cuda()
        self.std= torch.tensor([58.395, 57.12, 57.375]).cuda()

        self.net = UWNet(is_noise=False)
        self.aod = UWNet(is_noise=True)
        # self.net.eval()
        weights_path = '@fuhao/exp/500Adaptive/520/522color_correction/checkpoints/epoch_10000.pth'


        print('+---------------------------------------------+')
        print('+ Loading weights into ' + 'uwnet' + ' state dict.. +')
        print('+ Weights path: ', weights_path)
        print('+---------------------------------------------+')
        model_dict = self.net.state_dict()
        
        num = 0
        for k in model_dict.keys():
            print('model =>', k)
            num += 1
        print('total num: ', num)
        num = 0

        pretrained_dict = torch.load(weights_path)

        for k in pretrained_dict.keys():
            print('checkpoint =>', k)
            num += 1
        print('total num: ', num)
        num = 0

        momo_dict = {}
        for k, v in pretrained_dict.items(): 
            # if k in model_dict.keys() and cfg.load_checkpoint['load_keys'] in k: # 注意要修改的地方
            if k in model_dict.keys():
                if pretrained_dict[k].size() == model_dict[k].size():
                    momo_dict.update({k: v})

        # momo_dict = {k:v for 'module.'+k, v in pretrained_dict['state_dict'].items() if k in model_dict.keys() and pretrained_dict['state_dict'][k].shape() == model_dict[k].shape() }
        for k, v in momo_dict.items():
            print('model load => ', k)
            num += 1
        print('total num: ', num)

        model_dict.update(momo_dict)
        self.net.load_state_dict(model_dict)
        print('+---------------------------------------------+')
        print('+                 Finished！                  +')
        print('+---------------------------------------------+')
                

    def forward(self, x):
        # print('ori_x:', torch.max(x).data, torch.min(x).data, torch.mean(x).data)
        # mmax = torch.max(torch.max(x, dim=3, keepdim=True).values, dim=2, keepdim=True).values
        # mmin = torch.min(torch.min(x, dim=3, keepdim=True).values, dim=2, keepdim=True).values
        # x = (x - mmin) / (mmax - mmin)
        x = x / 255.
        
        # save_image(x, '@fuhao/transfer/input.jpg', normalize=True)
        _, _, w, h = x.shape

        k3_img = self.net(x)  # [1, c, h, w]
        # save_image(k3_img, '@fuhao/transfer/k3_img.jpg', normalize=True)
        x = torch.log(x * 255. + 1.) - torch.log(k3_img * 255.+ 1.)
        # x = torch.exp(x)
        # print('log_x:', torch.max(x).data, torch.min(x).data, torch.mean(x).data)
        # x = x - noise * 255.
        
        # x = x / 255.
        
        
        mmax = torch.max(torch.max(x, dim=3, keepdim=True).values, dim=2, keepdim=True).values
        mmin = torch.min(torch.min(x, dim=3, keepdim=True).values, dim=2, keepdim=True).values
        x = (x - mmin) / (mmax - mmin)
        
        # x = torch.pow(x, 6.) * 255.
        x = torch.pow(x, 6.)
        # save_image(x, '@fuhao/transfer/cc.jpg', normalize=True)
        # _, noise = self.aod(x)
        # save_image(noise, '@fuhao/transfer/aod.jpg', normalize=True)
        img = x.detach()
        noise = self.net(img)
        # x = ( x * noise - noise + 1. )
        x = x + x - noise
        
        mmax = torch.max(torch.max(x, dim=3, keepdim=True).values, dim=2, keepdim=True).values
        mmin = torch.min(torch.min(x, dim=3, keepdim=True).values, dim=2, keepdim=True).values
        x = (x - mmin) / (mmax - mmin) * 255.
        
        # print('norm_x:', torch.max(x).data, torch.min(x).data, torch.mean(x).data)

        
        x = x.transpose(1, 3)
        x = (x - self.mean) / self.std
        x = x.transpose(1, 3)
        # save_image(x, '@fuhao/transfer/cc_deblur.jpg', normalize=True)

        """1. MSR"""

        """2. white balance"""

        """3. enhancement + denoising"""

        """4. gamma correction"""

        """5. sharpen"""

        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)

        # """Over"""
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
    
@BACKBONES.register_module()
class ResNet_v527(ResNet):
    
    def __init__(self,
                 depth,
                 conv_cfg=None,
                 **kwargs
                ):
        super(ResNet_v527, self).__init__(depth, **kwargs)
        self.conv_cfg = conv_cfg

        self.mean= torch.tensor([123.675, 116.28, 103.53]).cuda()
        self.std= torch.tensor([58.395, 57.12, 57.375]).cuda()

        
        self.masknet = ResNet50_FPN()
        weights_path = '@fuhao/exp/500Adaptive/520/529label_mask/checkpoints/529_only_guide_x0_4580_epoch_12.pth'
        # weights_path = '@fuhao/exp/500Adaptive/520/526homology/checkpoints/label_guide2_4670_epoch_9.pth'
        # weights_path = '@fuhao/exp/500Adaptive/520/526homology/checkpoints/test_label_guide_4470_epoch_12.pth'

        print('+---------------------------------------------+')
        print('+ Loading weights into ' + 'masknet' + ' state dict.. +')
        print('+ Weights path: ', weights_path)
        print('+---------------------------------------------+')
        model_dict = self.masknet.state_dict()
        
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
            if k in model_dict.keys():
                if pretrained_dict[k].size() == model_dict[k].size():
                    momo_dict.update({k: v})

        # momo_dict = {k:v for 'module.'+k, v in pretrained_dict['state_dict'].items() if k in model_dict.keys() and pretrained_dict['state_dict'][k].shape() == model_dict[k].shape() }
        for k, v in momo_dict.items():
            print('model load => ', k)
            num += 1
        print('total num: ', num)

        model_dict.update(momo_dict)
        self.masknet.load_state_dict(model_dict)
        print('+---------------------------------------------+')
        print('+                 Finished！                  +')
        print('+---------------------------------------------+')
                

    def forward(self, x):
        
        input = x.detach()
        N, C, H, W = input.shape
        with torch.no_grad():
            self.masknet.eval()
            mask_fpn, mask = self.masknet(input)
            # mask0 = nn.functional.interpolate(mask[0], size=[H, W], mode='bilinear', align_corners=False)
            # mask1 = nn.functional.interpolate(mask[1], size=[H, W], mode='bilinear', align_corners=False)
            # mask2 = nn.functional.interpolate(mask[2], size=[H, W], mode='bilinear', align_corners=False)
            # mask3 = nn.functional.interpolate(mask[3], size=[H, W], mode='bilinear', align_corners=False)
            mask0 = torch.where(mask[0]>0.2, 1.0, 0.0)
            mask0 = nn.functional.interpolate(mask0, size=[H, W], mode='bilinear', align_corners=False)
            # x0 = x * mask0
            # x1 = x * mask1
            # x2 = x * mask2
            # x3 = x * mask3
            x = x * mask0
        # save_image(x, '@fuhao/transfer/mask0_img.jpg', normalize=True)
        # save_image(x1, '@fuhao/transfer/mask1_img.jpg', normalize=True)
        # save_image(x2, '@fuhao/transfer/mask2_img.jpg', normalize=True)
        # save_image(x3, '@fuhao/transfer/mask3_img.jpg', normalize=True)
        
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)

        # """Over"""
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                # if i == 0:
                #     C, N, H, W = x.shape
                #     mask = nn.functional.interpolate(mask0, size=[H, W], mode='bilinear', align_corners=False)
                #     x = x * mask
                
                outs.append(x)
        return tuple(outs)


@BACKBONES.register_module()
class ResNet_v528(ResNet):
    
    def __init__(self,
                 depth,
                 conv_cfg=None,
                 **kwargs
                ):
        super(ResNet_v528, self).__init__(depth, **kwargs)
        self.conv_cfg = conv_cfg

        self.mean= torch.tensor([123.675, 116.28, 103.53]).cuda()
        self.std= torch.tensor([58.395, 57.12, 57.375]).cuda()

        
        # self.masknet = ResNet50_FPN()
        # # weights_path = '@fuhao/exp/500Adaptive/520/526homology/checkpoints/label_guide2_4670_epoch_9.pth'
        # weights_path = '@fuhao/exp/500Adaptive/520/526homology/checkpoints/test_label_guide_4470_epoch_12.pth'

        # print('+---------------------------------------------+')
        # print('+ Loading weights into ' + 'masknet' + ' state dict.. +')
        # print('+ Weights path: ', weights_path)
        # print('+---------------------------------------------+')
        # model_dict = self.masknet.state_dict()
        
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
        # self.masknet.load_state_dict(model_dict)
        # print('+---------------------------------------------+')
        # print('+                 Finished！                  +')
        # print('+---------------------------------------------+')
                

    def forward(self, x):
        
        # input = x.detach()
        # N, C, H, W = input.shape
        # self.masknet.eval()
        # with torch.no_grad():
        #     mask_fpn, mask = self.masknet(input)
        #     # mask0 = nn.functional.interpolate(mask[0], size=[H, W], mode='bilinear', align_corners=False)
        #     # mask1 = nn.functional.interpolate(mask[1], size=[H, W], mode='bilinear', align_corners=False)
        #     # mask2 = nn.functional.interpolate(mask[2], size=[H, W], mode='bilinear', align_corners=False)
        #     # mask3 = nn.functional.interpolate(mask[3], size=[H, W], mode='bilinear', align_corners=False)
        #     mask0 = torch.where(mask[0]>0.1, 1.0, 0.0)
        #     mask0 = nn.functional.interpolate(mask0, size=[H, W], mode='bilinear', align_corners=False)
        #     # x0 = x * mask0
        #     # x1 = x * mask1
        #     # x2 = x * mask2
        #     # x3 = x * mask3
        # x = x * mask0
        # # save_image(x, '@fuhao/transfer/mask0_img.jpg', normalize=True)
        # # save_image(x1, '@fuhao/transfer/mask1_img.jpg', normalize=True)
        # # save_image(x2, '@fuhao/transfer/mask2_img.jpg', normalize=True)
        # # save_image(x3, '@fuhao/transfer/mask3_img.jpg', normalize=True)
        
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)

        # """Over"""
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:        
                t = x.detach().pow(2).mean(1)
                t = t.data.cpu().numpy()
                # fig = sns.heatmap(data=t[0], vmax=9., vmin=6.5, cmap='viridis') 
                # fig = sns.heatmap(data=t[0], vmax=4., vmin=3.4, cmap='viridis')
                fig = sns.heatmap(data=t[0].squeeze(), cmap='viridis')  
                heatmap = fig.get_figure()
                # heatmap.savefig(output_images_path + name[0], dpi=400)
                heatmap.savefig('@fuhao/transfer/feat_t%d.png'%i, dpi=400)
                heatmap.clear()
                
                outs.append(x)
                
        return tuple(outs)
    
    
@BACKBONES.register_module()
class ResNet_v548(ResNet):
    
    def __init__(self,
                 depth,
                 conv_cfg=None,
                 **kwargs
                ):
        super(ResNet_v548, self).__init__(depth, **kwargs)
        self.conv_cfg = conv_cfg

        self.mean= torch.tensor([123.675, 116.28, 103.53]).cuda()
        self.std= torch.tensor([58.395, 57.12, 57.375]).cuda()

        self.mse_loss = torch.nn.MSELoss().cuda()
        
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
        
        # self.masknet = ResNet50_FPN()
        # # weights_path = '@fuhao/exp/500Adaptive/520/526homology/checkpoints/label_guide2_4670_epoch_9.pth'
        # weights_path = '@fuhao/exp/500Adaptive/520/526homology/checkpoints/test_label_guide_4470_epoch_12.pth'

        # print('+---------------------------------------------+')
        # print('+ Loading weights into ' + 'masknet' + ' state dict.. +')
        # print('+ Weights path: ', weights_path)
        # print('+---------------------------------------------+')
        # model_dict = self.masknet.state_dict()
        
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
        # self.masknet.load_state_dict(model_dict)
        # print('+---------------------------------------------+')
        # print('+                 Finished！                  +')
        # print('+---------------------------------------------+')
                

    def forward(self, x, img_metas=None, gt_bboxes=None):
        if gt_bboxes != None:
            img = x
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

                for j in range(len(gt_bboxes[i])):

                    Mask_fg[i][hmin[i][j]:hmax[i][j], wmin[i][j]:wmax[i][j]] = 1.0
                    
                Mask_bg[i] = torch.where(Mask_fg[i]>0, 0, 1)
                    
            Mask_fg = Mask_fg.unsqueeze(1)
            Mask_bg = Mask_bg.unsqueeze(1)
            input_img = img.detach()
            img_fg = input_img * Mask_fg
            img_bg = input_img * Mask_bg
            
        loss = 0.0
        
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)

        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:     
                
                if i == 0:

                    feat_fg = self.fg_block(x)
                    feat_bg = self.bg_block(x)
                    
                    gen_fg = self.fg_head(feat_fg)
                    gen_bg = self.bg_head(feat_bg)
                    
                    if gt_bboxes != None:
                        img_fg = nn.functional.interpolate(img_fg, size=[x.shape[2], x.shape[3]], mode='bilinear', align_corners=False)
                        img_bg = nn.functional.interpolate(img_bg, size=[x.shape[2], x.shape[3]], mode='bilinear', align_corners=False)
                        input_img = nn.functional.interpolate(input_img, size=[x.shape[2], x.shape[3]], mode='bilinear', align_corners=False)
                        
                        loss_fg = self.mse_loss(gen_fg, img_fg)
                        loss_bg = self.mse_loss(gen_bg, img_bg)  
                        loss_whole = self.mse_loss(gen_fg + gen_bg, input_img) 
                        loss = loss_fg + loss_bg + loss_whole
                    
                    # x = x + feat_fg - feat_bg   
                    x = feat_fg
                # t = x.detach().pow(2).mean(1)
                # t = t.data.cpu().numpy()
                # # fig = sns.heatmap(data=t[0], vmax=9., vmin=6.5, cmap='viridis') 
                # # fig = sns.heatmap(data=t[0], vmax=4., vmin=3.4, cmap='viridis')
                # fig = sns.heatmap(data=t[0].squeeze(), cmap='viridis')  
                # heatmap = fig.get_figure()
                # # heatmap.savefig(output_images_path + name[0], dpi=400)
                # heatmap.savefig('@fuhao/transfer/feat_t%d.png'%i, dpi=400)
                # heatmap.clear()
                
                outs.append(x)
                
        return tuple(outs), loss
    
@BACKBONES.register_module()
class ResNet_v549_2(ResNet):
    
    def __init__(self,
                 depth,
                 conv_cfg=None,
                 **kwargs
                ):
        super(ResNet_v549_2, self).__init__(depth, **kwargs)
        self.conv_cfg = conv_cfg

        self.mean= torch.tensor([123.675, 116.28, 103.53]).cuda()
        self.std= torch.tensor([58.395, 57.12, 57.375]).cuda()

        self.mse_loss = torch.nn.MSELoss().cuda()
        
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
        
        # self.masknet = ResNet50_FPN()
        # # weights_path = '@fuhao/exp/500Adaptive/520/526homology/checkpoints/label_guide2_4670_epoch_9.pth'
        # weights_path = '@fuhao/exp/500Adaptive/520/526homology/checkpoints/test_label_guide_4470_epoch_12.pth'

        # print('+---------------------------------------------+')
        # print('+ Loading weights into ' + 'masknet' + ' state dict.. +')
        # print('+ Weights path: ', weights_path)
        # print('+---------------------------------------------+')
        # model_dict = self.masknet.state_dict()
        
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
        # self.masknet.load_state_dict(model_dict)
        # print('+---------------------------------------------+')
        # print('+                 Finished！                  +')
        # print('+---------------------------------------------+')
                

    def forward(self, x, feat=None):
        
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        
        x = x + feat
        
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:     
                
                # if i == 0:
                #     # print(x.shape, feat.shape)
                #     x = x + feat
                
                outs.append(x)
                
        return tuple(outs)


class JNet(torch.nn.Module):
    def __init__(self, num=256):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(64, num, 3, 1, 0),
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
            torch.nn.Conv2d(num, 256, 1, 1, 0),
            # torch.nn.Sigmoid()
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


@BACKBONES.register_module()
class ResNet_v549_3(ResNet):
    
    def __init__(self,
                 depth,
                 conv_cfg=None,
                 **kwargs
                ):
        super(ResNet_v549_3, self).__init__(depth, **kwargs)
        self.conv_cfg = conv_cfg

        self.mean= torch.tensor([123.675, 116.28, 103.53]).cuda()
        self.std= torch.tensor([58.395, 57.12, 57.375]).cuda()

        self.mse_loss = torch.nn.MSELoss().cuda()
        
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
        
        self.j_net = JNet()
        
        # self.masknet = ResNet50_FPN()
        # # weights_path = '@fuhao/exp/500Adaptive/520/526homology/checkpoints/label_guide2_4670_epoch_9.pth'
        # weights_path = '@fuhao/exp/500Adaptive/520/526homology/checkpoints/test_label_guide_4470_epoch_12.pth'

        # print('+---------------------------------------------+')
        # print('+ Loading weights into ' + 'masknet' + ' state dict.. +')
        # print('+ Weights path: ', weights_path)
        # print('+---------------------------------------------+')
        # model_dict = self.masknet.state_dict()
        
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
        # self.masknet.load_state_dict(model_dict)
        # print('+---------------------------------------------+')
        # print('+                 Finished！                  +')
        # print('+---------------------------------------------+')
                

    def forward(self, x, feat=None):
        
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        
        # x = self.j_net(x)
        outs = []
        outs.append(x)
        # return tuple(outs)
    
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:     
                outs.append(x)
                return tuple(outs)
                
        return tuple(outs)
   
import torch
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
        # self.final = torch.nn.Sequential(
        #     torch.nn.Conv2d(num, 3, 1, 1, 0),
        #     torch.nn.Sigmoid()
        # )

    def forward(self, data):
        data = self.conv1(data)
        data = self.conv2(data)
        data = self.conv3(data)
        data = self.conv4(data)
        # data = self.final(data)

        return data
 

@BACKBONES.register_module()
class ResNet_return_stem(ResNet):
    
    def __init__(self,
                 depth,
                 conv_cfg=None,
                 **kwargs
                ):
        super(ResNet_return_stem, self).__init__(depth, **kwargs)
        self.conv_cfg = conv_cfg
        
        self.j_net = JNet()
        # self.j_net = nn.Sequential(
        #     nn.Conv2d(3, 64, 3, 1, 1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(True)
        # )
        
        self.stem = nn.Sequential(
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.stem1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        self.stem2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        
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

    def forward(self, x, feat=None):
        
        j_out = self.j_net(x)
        
        x = self.stem(j_out)
        stem0 = x
        # if self.deep_stem:
        #     x = self.stem(x)
        # else:
        #     x = self.conv1(x)
        #     x = self.norm1(x)
        #     x = self.relu(x)
        # x = self.stem0(x)
        # stem0 = x
        # x = self.stem1(x)
        # stem1 = x
        # x = self.stem2(x)
        # stem2 = x
        
        x = self.maxpool(x)
        
        outs = []
    
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:     
                outs.append(x)
                
        return tuple(outs), stem0, j_out


@BACKBONES.register_module()
class ResNet_return_stem1(ResNet):
    
    def __init__(self,
                 depth,
                 conv_cfg=None,
                 **kwargs
                ):
        super(ResNet_return_stem1, self).__init__(depth, **kwargs)
        self.conv_cfg = conv_cfg
        

    def forward(self, x, feat=None):
        
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        
        stem = x
        
        x = self.maxpool(x)
        
        outs = []
    
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:     
                outs.append(x)
                
        return tuple(outs), stem

if __name__ == "__main__":
    # model = ResNet18()
    model = ResNet_v512(50).cuda()
    imgs = torch.randn((2, 3, 224, 224)).cuda()
    outs= model(imgs)
    # print(cls)
    for i in outs:
        print(i.shape)