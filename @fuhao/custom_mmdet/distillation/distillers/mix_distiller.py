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

import torch.nn.functional as F
from torchvision.utils import save_image
from torchvision.transforms import ToTensor, Normalize


@DISTILLER.register_module()
class MixDistiller(BaseDetector):
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

        super(MixDistiller, self).__init__()
        
        self.teacher = build_detector(teacher_cfg.model,
                                        train_cfg=teacher_cfg.get('train_cfg'),
                                        test_cfg=teacher_cfg.get('test_cfg'))
        self.init_weights_teacher(teacher_pretrained)
        self.teacher.eval()

        self.student= build_detector(student_cfg.model,
                                        train_cfg=student_cfg.get('train_cfg'),
                                        test_cfg=student_cfg.get('test_cfg'))

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

                for item_loss in item_loc.methods:
                    loss_name = item_loss.name
                    self.distill_losses[loss_name] = build_distill_loss(item_loss)
        
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

       
        input = img
        # with torch.no_grad():
        #     self.teacher.eval()
        #     feat = self.teacher.extract_feat(img)

        with torch.no_grad():
            self.student.eval()
            feat = self.student.extract_feat(img)
        self.student.train()
       

        # """ 光流估计 
        x = img 
        feat = feat[0]
        # feat = feat.pow(2).mean(1)

        """ fuhao7i
        S_t = feat
        # N_T,C_T,H_T,W_T = S_t.shape
        # N_T,H_T,W_T = S_t.shape
        # mean = S_t.detach().view(N_T, -1).mean(1).view(N_T,  1, 1)
        # std = S_t.detach().view(N_T,  -1).std(1).view(N_T,  1, 1)
        # S_t_min = S_t.detach().view(N_T, -1).min(1).values.unsqueeze(1).unsqueeze(1)
        # S_t_max = S_t.detach().view(N_T,  -1).max(1).values.unsqueeze(1).unsqueeze(1)

        mean = S_t.mean(axis=2,keepdim=False).mean(axis=2,keepdim=False).unsqueeze(2),unsqueeze(2)
        std = S_t.std(axis=2,keepdim=False).std(axis=2,keepdim=False).unsqueeze(2).unsqueeze(2)
        # print(mean.shape, std.shape)
        normalize = Normalize(
            mean=mean,  # 各 channel 的均值
            std=std,  # 各 channel 的标准差
            inplace=False
        )

        S_t = normalize(S_t)
        # print(S_t.shape, S_t_min.shape, S_t_max.shape)
        # S_t = ( S_t - S_t_min ) / ( S_t_max - S_t_min )

        feat = S_t 
        # """
        # feat = feat.unsqueeze(1)

        # feat = F.interpolate(feat, scale_factor=4, mode='bilinear', align_corners=False)
        # x = x - feat * (torch.pow(x, 2) - x.abs())
        # x = torch.clamp(x, 0., 1.)
        # x = x / feat
        # x = torch.sigmoid(x)
        # img = x
        # save_image(x, "@fuhao/transfer/input-x.jpg" , normalize=True)
        # over """

        """ fuhao7i
        feat_s = self.student.extract_feat(img)
        # """
        data = {
            'img':img,
            'feat':feat
        }
        print(data.keys(), end='')

        student_loss = self.student.forward_train(data, img_metas, **kwargs)
        # logger.info(
        #     f'student_loss: {student_loss}'
        # )
        # raise
        
        
        # buffer_dict = dict(self.named_buffers())
        # if self.distill_cfg != None:
        #     for item_loc in self.distill_cfg:
                
        #         student_module = 'student_' + item_loc.student_module.replace('.','_')
        #         teacher_module = 'teacher_' + item_loc.teacher_module.replace('.','_')
                
        #         student_feat = buffer_dict[student_module]
        #         teacher_feat = buffer_dict[teacher_module]

        #         for item_loss in item_loc.methods:
        #             loss_name = item_loss.name
                    
        #             student_loss[loss_name] = self.distill_losses[loss_name](student_feat,teacher_feat,kwargs['gt_bboxes'], img_metas, input= input)
        
        
        return student_loss
    
    def simple_test(self, img, img_metas, **kwargs):
        # """ fuhao7i
        # with torch.no_grad():
        #     self.teacher.eval()
        #     feat = self.teacher.extract_feat(img)

        with torch.no_grad():
            self.student.eval()
            feat = self.student.extract_feat(img)
            # self.student.train()


        feat = feat[0]
        data = {
            'img':img,
            'feat':feat
        }
        # """
        

        return self.student.simple_test(data, img_metas, **kwargs)
    def aug_test(self, imgs, img_metas, **kwargs):
        return self.student.aug_test(imgs, img_metas, **kwargs)
    def extract_feat(self, imgs):
        """Extract features from images."""
        return self.student.extract_feat(imgs)


