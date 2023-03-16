from tkinter.tix import InputOnly
import torch.nn as nn
import torch.nn.functional as F
import torch
from mmdet.models.detectors.base import BaseDetector
from mmdet.models import build_detector
from mmcv.runner import  load_checkpoint, _load_checkpoint, load_state_dict
from ..builder import DISTILLER,build_distill_loss
from collections import OrderedDict

from mmdet.utils import get_root_logger
logger = get_root_logger(log_level='INFO')

import cv2
import numpy as np
from torchvision.utils import save_image
import seaborn as sns
sns.set(font_scale=1.5)
import random



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



@DISTILLER.register_module()
class Custom_DetectionDistiller(BaseDetector):
    """Base distiller for detectors.

    It typically consists of teacher_model and student_model.
    """
    def __init__(self,
                 teacher_cfg,
                 student_cfg,
                 distill_cfg=None,
                 teacher_pretrained=None,
                 student_pretrained=None,
                 distill_loss_pretrained=None,
                 init_student=False):

        super(Custom_DetectionDistiller, self).__init__()
        
        self.teacher = build_detector(teacher_cfg.model,
                                        train_cfg=teacher_cfg.get('train_cfg'),
                                        test_cfg=teacher_cfg.get('test_cfg'))
        self.init_weights_teacher(teacher_pretrained)
        self.teacher.eval()

        self.student= build_detector(student_cfg.model,
                                        train_cfg=student_cfg.get('train_cfg'),
                                        test_cfg=student_cfg.get('test_cfg'))
        
        
        
        self.mse_loss = torch.nn.MSELoss().cuda()
        self.gauss_kernel = get_gaussian_kernel(size=31).cuda()
        

        if init_student:
            t_checkpoint = _load_checkpoint(teacher_pretrained)
            all_name = []
            for name, v in t_checkpoint["state_dict"].items():
                if name.startswith("backbone."):
                    continue
                else:
                    all_name.append((name, v))

            state_dict = OrderedDict(all_name)
            load_state_dict(self.student, state_dict)

        if student_pretrained:
            self.init_weights_student(student_pretrained)

        self.distill_losses = nn.ModuleDict()
        self.distill_cfg = distill_cfg

        student_modules = dict(self.student.named_modules())
        teacher_modules = dict(self.teacher.named_modules())
        def regitster_hooks(student_module,teacher_module):
            def hook_teacher_forward(module, input, output):

                    self.register_buffer(teacher_module,output)
                
            def hook_student_forward(module, input, output):

                    self.register_buffer( student_module,output )
            return hook_teacher_forward,hook_student_forward
        
        if distill_cfg != None:
            
            for item_loc in distill_cfg:
                
                student_module = 'student_' + item_loc.student_module.replace('.','_')
                teacher_module = 'teacher_' + item_loc.teacher_module.replace('.','_')

                self.register_buffer(student_module,None)
                self.register_buffer(teacher_module,None)

                hook_teacher_forward,hook_student_forward = regitster_hooks(student_module ,teacher_module )
                teacher_modules[item_loc.teacher_module].register_forward_hook(hook_teacher_forward)
                student_modules[item_loc.student_module].register_forward_hook(hook_student_forward)

                # for item_loss in item_loc.methods:
                #     loss_name = item_loss.name
                #     self.distill_losses[loss_name] = build_distill_loss(item_loss)
        
        if distill_loss_pretrained:
            print('载入distill loss 的权重')
            init_weights_distill_loss(distill_loss_pretrained)

    def base_parameters(self):
        return nn.ModuleList([self.student,self.distill_losses])


    @property
    def with_neck(self):
        """bool: whether the detector has a neck"""
        return hasattr(self.student, 'neck') and self.student.neck is not None

    @property
    def with_shared_head(self):
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self.student, 'roi_head') and self.student.roi_head.with_shared_head

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return ((hasattr(self.student, 'roi_head') and self.student.roi_head.with_bbox)
                or (hasattr(self.student, 'bbox_head') and self.student.bbox_head is not None))

    @property
    def with_mask(self):
        """bool: whether the detector has a mask head"""
        return ((hasattr(self.student, 'roi_head') and self.student.roi_head.with_mask)
                or (hasattr(self.student, 'mask_head') and self.student.mask_head is not None))

    def init_weights_distill_loss(self, path=None):
        i = path
        weights_path = i['weight_path']
        print('+---------------------------------------------+')
        print('+ Loading weights into ' + 'distill loss' + ' state dict.. +')
        print('+ ', weights_path, ' +')
        print('+---------------------------------------------+')
        model_dict = self.distill_losses.state_dict()
        
        num = 0
        for k in model_dict.keys():
            print('model =>', k)
            num += 1
        print('total num: ', num)
        num = 0

        pretrained_dict = torch.load(weights_path)

        for k in pretrained_dict['state_dict'].keys():
            print('checkpoint =>', k)
            num += 1
        print('total num: ', num)
        num = 0

        momo_dict = {}
        for k, v in pretrained_dict['state_dict'].items(): 
            # if k in model_dict.keys() and cfg.load_checkpoint['load_keys'] in k: # 注意要修改的地方
            if k in model_dict.keys():
                if i['load_keys']:
                    for key in i['load_keys']:
                        if key in k:
                            if pretrained_dict['state_dict'][k].size() == model_dict[k].size():
                                momo_dict.update({k: v})
                elif i['delete_keys']:
                    for key in i['delete_keys']:
                        if key in k:
                            break
                else:
                    if pretrained_dict['state_dict'][k].size() == model_dict[k].size():
                        momo_dict.update({k: v})


        # momo_dict = {k:v for 'module.'+k, v in pretrained_dict['state_dict'].items() if k in model_dict.keys() and pretrained_dict['state_dict'][k].shape() == model_dict[k].shape() }
        for k, v in momo_dict.items():
            print('model load => ', k)
            num += 1
        print('total num: ', num)

        model_dict.update(momo_dict)
        self.distill_losses.load_state_dict(model_dict)
        print('+---------------------------------------------+')
        print('+                 Finished！                  +')
        print('+---------------------------------------------+')
        



    def init_weights_teacher(self, path=None):
        """Load the pretrained model in teacher detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        # checkpoint = load_checkpoint(self.teacher, path, map_location='cpu')
        i = path
        weights_path = i['weight_path']
        print('+---------------------------------------------+')
        print('+ Loading weights into ' + 'teacher model' + ' state dict.. +')
        print('+ ', weights_path, ' +')
        print('+---------------------------------------------+')
        model_dict = self.teacher.state_dict()
        
        num = 0
        for k in model_dict.keys():
            print('model =>', k)
            num += 1
        print('total num: ', num)
        num = 0

        # pretrained_dict = torch.load(weights_path)

        if hasattr(i, 'type'):
            from collections import OrderedDict
            fgd_model = torch.load(weights_path)
            all_name = []
            for name, v in fgd_model["state_dict"].items():
                if name.startswith("student."):
                    all_name.append((name[8:], v))
                else:
                    continue
            state_dict = OrderedDict(all_name)
            fgd_model['state_dict'] = state_dict

            pretrained_dict = fgd_model
        else :
            pretrained_dict = torch.load(weights_path)


        for k in pretrained_dict['state_dict'].keys():
            print('checkpoint =>', k)
            num += 1
        print('total num: ', num)
        num = 0

        momo_dict = {}
        for k, v in pretrained_dict['state_dict'].items(): 
            # if k in model_dict.keys() and cfg.load_checkpoint['load_keys'] in k: # 注意要修改的地方
            if k in model_dict.keys():
                if i['load_keys']:
                    for key in i['load_keys']:
                        if key in k:
                            if pretrained_dict['state_dict'][k].size() == model_dict[k].size():
                                momo_dict.update({k: v})
                elif i['delete_keys']:
                    for key in i['delete_keys']:
                        if key in k:
                            break
                else:
                    if pretrained_dict['state_dict'][k].size() == model_dict[k].size():
                        momo_dict.update({k: v})


        # momo_dict = {k:v for 'module.'+k, v in pretrained_dict['state_dict'].items() if k in model_dict.keys() and pretrained_dict['state_dict'][k].shape() == model_dict[k].shape() }
        for k, v in momo_dict.items():
            print('model load => ', k)
            num += 1
        print('total num: ', num)

        model_dict.update(momo_dict)
        self.teacher.load_state_dict(model_dict)
        print('+---------------------------------------------+')
        print('+                 Finished！                  +')
        print('+---------------------------------------------+')

    def init_weights_student(self, path=None):
        """Load the pretrained model in teacher detector.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        # checkpoint = load_checkpoint(self.teacher, path, map_location='cpu')
        i = path
        weights_path = i['weight_path']
        print('+---------------------------------------------+')
        print('+ Loading weights into ' + 'student model' + ' state dict.. +')
        print('+ ', weights_path, ' +')
        print('+---------------------------------------------+')
        model_dict = self.student.state_dict()
        
        num = 0
        for k in model_dict.keys():
            print('model =>', k)
            num += 1
        print('total num: ', num)
        num = 0

        # fgd_model = torch.load(weights_path)
        
        # all_name = []
        # for name, v in fgd_model["state_dict"].items():
        #     if name.startswith("student."):
        #         all_name.append((name[8:], v))
        #     else:
        #         continue
        # state_dict = OrderedDict(all_name)
        # fgd_model['state_dict'] = state_dict

        # pretrained_dict = fgd_model

        if hasattr(i, 'type'):
            from collections import OrderedDict
            fgd_model = torch.load(weights_path)
            all_name = []
            for name, v in fgd_model["state_dict"].items():
                if name.startswith("student."):
                    all_name.append((name[8:], v))
                else:
                    continue
            state_dict = OrderedDict(all_name)
            fgd_model['state_dict'] = state_dict

            pretrained_dict = fgd_model
        else :
            pretrained_dict = torch.load(weights_path)

        for k in pretrained_dict['state_dict'].keys():
            print('checkpoint =>', k)
            num += 1
        print('total num: ', num)
        num = 0

        momo_dict = {}
        for k, v in pretrained_dict['state_dict'].items(): 
            # if k in model_dict.keys() and cfg.load_checkpoint['load_keys'] in k: # 注意要修改的地方
            k = k.replace('student.', '')
            if k in model_dict.keys():
                if i['load_keys']:
                    for key in i['load_keys']:
                        if key in k:
                            if pretrained_dict['state_dict'][k].size() == model_dict[k].size():
                                momo_dict.update({k: v})
                elif i['delete_keys']:
                    for key in i['delete_keys']:
                        if key in k:
                            break
                else:
                    if pretrained_dict['state_dict'][k].size() == model_dict[k].size():
                        momo_dict.update({k: v})


        # momo_dict = {k:v for 'module.'+k, v in pretrained_dict['state_dict'].items() if k in model_dict.keys() and pretrained_dict['state_dict'][k].shape() == model_dict[k].shape() }
        for k, v in momo_dict.items():
            print('model load => ', k)
            num += 1
        print('total num: ', num)

        model_dict.update(momo_dict)
        self.student.load_state_dict(model_dict)
        print('+---------------------------------------------+')
        print('+                 Finished！                  +')
        print('+---------------------------------------------+')


    def forward_train(self, img, img_metas, **kwargs):
        

        # ----------------------------------------------
        # 计算前景和背景
        N, C, H, W = img.shape
        # Mask_fg = torch.zeros_like(S_attention_t)
        # Mask_bg = torch.ones_like(S_attention_t)
        Mask_fg = torch.zeros([N, H, W]).cuda()
        Mask_bg = torch.ones([N, H, W]).cuda()
        wmin,wmax,hmin,hmax = [],[],[],[]
        gt_bboxes = kwargs['gt_bboxes']
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
        
        
        Mask_fg = Mask_fg.unsqueeze(1)
        # ----------------------------------------------------------------

        # print(kwargs)
        # print('fuhao', torch.cat((kwargs['gt_labels'][0], kwargs['gt_labels'][1]), 0))
        # print(kwargs['gt_labels'])

        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components(student's losses and distiller's losses).
        """
       # teacher 输入前景mask * img
        # input_teacher = img
        with torch.no_grad():
            self.teacher.eval()
            input_teacher = img * Mask_fg
            feat = self.teacher.extract_feat(input_teacher)

        # alpah = random.random()
        # mix_img = alpah * img + (1 - alpah) * mix_img
        # mix = self.student.extract_feat(img)
        # student_loss = {}
        
        # mix_img = torch.pow(mix_img, 0.5)
                
        # mix_img = 0.5 * img[0, :, :, :] + 0.5 * img[1, :, :, :]
        # mix_img = mix_img.unsqueeze(0)
        # kwargs['gt_labels'] = [torch.cat((kwargs['gt_labels'][0], kwargs['gt_labels'][1]), 0)]
        # kwargs['gt_bboxes'] = [torch.cat((kwargs['gt_bboxes'][0], kwargs['gt_bboxes'][1]), 0)]
        # print('img_metas: ', img_metas)
        # print('img_metas: ', img_metas[0])
        # print(kwargs)
        # print('fuhao', torch.cat((kwargs['gt_labels'][0], kwargs['gt_labels'][1]), 0))
        # print(kwargs['gt_labels'])
        # print(kwargs['gt_bboxes'][0].append(kwargs['gt_bboxes'][1]), kwargs['gt_labels'][0].append(kwargs['gt_labels'][1]))

        # student_loss = self.student.forward_train(mix_img, [img_metas[0]], **kwargs)
        student_loss = self.student.forward_train(img, img_metas, **kwargs)

        # logger.info(
        #             f'kwargs: {kwargs}'
        #         )
        
        custom_loss = 0.0
        
        # save_image(img, './@fuhao/transfer/input.jpg', normalize=True)
        # save_image(mix_img, './@fuhao/transfer/mix_img.jpg', normalize=True)        

        buffer_dict = dict(self.named_buffers())
        # """
        if self.distill_cfg != None:
            for item_loc in self.distill_cfg:
                
                student_module = 'student_' + item_loc.student_module.replace('.','_')
                teacher_module = 'teacher_' + item_loc.teacher_module.replace('.','_')
                
                student_feat = buffer_dict[student_module]
                teacher_feat = buffer_dict[teacher_module]
                # print(student_feat.shape)
                # print(Mask_fg.shape)
                # mask = nn.functional.interpolate(Mask_fg, size=[student_feat.shape[2], student_feat.shape[3]], mode='bilinear', align_corners=False)
                # custom_loss += self.mse_loss(mask * student_feat, mask * teacher_feat)
                # custom_loss += self.mse_loss(mask[0] * student_feat, (mask[0] * teacher_feat[0]).unsqueeze(0))
                # custom_loss += self.mse_loss(mask[1] * student_feat, (mask[1] * teacher_feat[1]).unsqueeze(0))

                custom_loss += self.mse_loss(student_feat, teacher_feat)
                
                # student_feat = student_feat.pow(2).mean(1)
                # teacher_feat = teacher_feat.pow(2).mean(1)  
                # t = student_feat.data.cpu().numpy()
                # fig = sns.heatmap(data=t[0], cmap='viridis')  
                # heatmap = fig.get_figure()
                # heatmap.savefig('@fuhao/transfer/student_%s.jpg'%str(item_loc), dpi=400)
                # heatmap.clear()   

                # t = teacher_feat.data.cpu().numpy()
                # fig = sns.heatmap(data=t[0], cmap='viridis')  
                # heatmap = fig.get_figure()
                # heatmap.savefig('@fuhao/transfer/teacher_%s.jpg'%str(item_loc), dpi=400)
                # heatmap.clear()               

                # for item_loss in item_loc.methods:
                #     loss_name = item_loss.name
                    
                #     student_loss[loss_name] = self.distill_losses[loss_name](student_feat,teacher_feat,kwargs['gt_bboxes'], img_metas, input= input)
        
        student_loss['custom_loss'] = custom_loss
        # """
        return student_loss

    
    def simple_test(self, img, img_metas, **kwargs):
        return self.student.simple_test(img, img_metas, **kwargs)
    def aug_test(self, imgs, img_metas, **kwargs):
        return self.student.aug_test(imgs, img_metas, **kwargs)
    def extract_feat(self, imgs):
        """Extract features from images."""
        return self.student.extract_feat(imgs)


