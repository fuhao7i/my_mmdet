import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.builder import BACKBONES
from mmdet.models.utils import ResLayer
from mmdet.models.backbones.resnet import ResNet

import sys
sys.path.append('@fuhao/')
from custom_mmdet.models.backbones.stem.scene_mining import *

import torch.nn.functional as F

import seaborn as sns
sns.set(font_scale=1.5)

from torchvision.utils import save_image

@BACKBONES.register_module()
class ResNet_wostem(ResNet):
    
    def __init__(self,
                 depth,
                 conv_cfg=None,
                 **kwargs
                ):
        super(ResNet_wostem, self).__init__(depth, **kwargs)
        self.conv_cfg = conv_cfg
                

    def forward(self, x):
        """Forward function."""
        # if self.deep_stem:
        #     x = self.stem(x)
        # else:
        #     x = self.conv1(x)
        #     x = self.norm1(x)
        #     x = self.relu(x)
        x = self.maxpool(x)
        # print(x.shape)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

@BACKBONES.register_module()
class ResNet_v1(ResNet):
    
    def __init__(self,
                 depth,
                 **kwargs
                ):
        super(ResNet_v1, self).__init__(depth, **kwargs)
        
        self.c1 = nn.Sequential(
            nn.Conv2d(12, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            # nn.ReLU(inplace=True),
        )

        self.ae = AE(conv=True)

        # self.custom_init_weights(self.c1)
        # self.custom_init_weights(self.ae)



    def partition(self, x):
        B, H, W, C = x.shape

        # padding
        # pad_input = (H % 2 == 1) or (W % 2 == 1)
        # if pad_input:
        #     x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, :, 0::2, 0::2]  # B C H/2 W/2 
        x1 = x[:, :, 1::2, 0::2]  # B C H/2 W/2 
        x2 = x[:, :, 0::2, 1::2]  # B C H/2 W/2 
        x3 = x[:, :, 1::2, 1::2]  # B C H/2 W/2 

        x = torch.cat([x0, x1, x2, x3], 1)  # B 4*C H/2 W/2

        return x

    def adaptive_ae(self, x):
        input = x
        
        # 下采样
        # print('ori_shape', input.shape)
        x = self.partition(x)
        # print(x.shape)
        x = self.c1(x)
        # print(x.shape)
        x, mix2 = self.ae(x)
        # print(x.shape)

        # ----------------------------------------------------------------
        # x[x < x.mean()] = 0
        # print(x.shape)
        # from torchvision.utils import save_image
        # save_image(x, "./x.jpg" , normalize=True)
        # 上采样 + 通道数降为3
        # print(input.shape)
        # print(x.shape)
        # x = torch.sigmoid(x)
        # x = input * x + input

        # x = input * x
        
        # from torchvision.utils import save_image
        # save_image(input, "./input.jpg" , normalize=True)
        # save_image(x, "./input-x.jpg" , normalize=True)

        # import seaborn as sns
        # sns.set(font_scale=1.5)
        # t = input.detach().pow(2).mean(1)
        # t = t.data.cpu().numpy()
        # # fig = sns.heatmap(data=t[0], vmax=9., vmin=6.5, cmap='viridis') 
        # # fig = sns.heatmap(data=t[0], vmax=4., vmin=3.4, cmap='viridis')
        # fig = sns.heatmap(data=t[0], cmap='viridis')  
        # heatmap = fig.get_figure()
        # # heatmap.savefig(output_images_path + name[0], dpi=400)
        # heatmap.savefig('./1.png', dpi=400)
        # heatmap.clear()

        # t = x.detach().pow(2).mean(1)
        # t = t.data.cpu().numpy()
        # # fig = sns.heatmap(data=t[0], vmax=9., vmin=6.5, cmap='viridis') 
        # # fig = sns.heatmap(data=t[0], vmax=4., vmin=3.4, cmap='viridis')
        # fig = sns.heatmap(data=t[0], cmap='viridis')  
        # heatmap = fig.get_figure()
        # # heatmap.savefig(output_images_path + name[0], dpi=400)
        # heatmap.savefig('./2.png', dpi=400)
        # heatmap.clear()
        # raise
        # ----------------------------------------------------------------

        # 上采样 + 通道数降为3
        # print(input.shape)
        # print(x.shape)
        x = input * x * 0.7 + input * 0.3

        return x, mix2

    def forward(self, x):

        x, mix2 = self.adaptive_ae(x)

        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)

        # x = self.partition(x)
        # x = self.c1(x)

        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                if i == 0:
                    x = x + mix2
                outs.append(x)
        return tuple(outs)


@BACKBONES.register_module()
class ResNet_v2(ResNet):
    
    def __init__(self,
                 depth,
                 **kwargs
                ):
        super(ResNet_v2, self).__init__(depth, **kwargs)
        
        self.c1 = nn.Sequential(
            nn.Conv2d(12, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            # nn.ReLU(inplace=True),
        )

        # self.c2 = nn.Sequential(
        #     nn.Conv2d(32, 64, 3, 2, 1),
        #     nn.BatchNorm2d(64),
        #     nn.GELU(),
        #     # nn.ReLU(inplace=True),
        # )

        self.ae = AE(conv=True)


        # self.custom_init_weights(self.c1)
        # self.custom_init_weights(self.ae)



    def partition(self, x):
        B, H, W, C = x.shape

        # padding
        # pad_input = (H % 2 == 1) or (W % 2 == 1)
        # if pad_input:
        #     x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, :, 0::2, 0::2]  # B C H/2 W/2 
        x1 = x[:, :, 1::2, 0::2]  # B C H/2 W/2 
        x2 = x[:, :, 0::2, 1::2]  # B C H/2 W/2 
        x3 = x[:, :, 1::2, 1::2]  # B C H/2 W/2 

        x = torch.cat([x0, x1, x2, x3], 1)  # B 4*C H/2 W/2

        return x

    def adaptive_ae(self, x):
        input = x
        
        # 下采样
        # print('ori_shape', input.shape)
        x = self.partition(x)
        # print(x.shape)
        x = self.c1(x)
        # x = self.c2(x)
        # print(x.shape)
        x, mix2 = self.ae(x)
        # print(x.shape)

        # ----------------------------------------------------------------
        # x[x < x.mean()] = 0
        # print(x.shape)
        # from torchvision.utils import save_image
        # save_image(x, "./x.jpg" , normalize=True)
        # 上采样 + 通道数降为3
        # print(input.shape)
        # print(x.shape)
        # x = torch.sigmoid(x)
        # x = input * x + input

        # x = input * x
        
        # from torchvision.utils import save_image
        # save_image(input, "./input.jpg" , normalize=True)
        # save_image(x, "./input-x.jpg" , normalize=True)

        # import seaborn as sns
        # sns.set(font_scale=1.5)
        # t = input.detach().pow(2).mean(1)
        # t = t.data.cpu().numpy()
        # # fig = sns.heatmap(data=t[0], vmax=9., vmin=6.5, cmap='viridis') 
        # # fig = sns.heatmap(data=t[0], vmax=4., vmin=3.4, cmap='viridis')
        # fig = sns.heatmap(data=t[0], cmap='viridis')  
        # heatmap = fig.get_figure()
        # # heatmap.savefig(output_images_path + name[0], dpi=400)
        # heatmap.savefig('./1.png', dpi=400)
        # heatmap.clear()

        # t = x.detach().pow(2).mean(1)
        # t = t.data.cpu().numpy()
        # # fig = sns.heatmap(data=t[0], vmax=9., vmin=6.5, cmap='viridis') 
        # # fig = sns.heatmap(data=t[0], vmax=4., vmin=3.4, cmap='viridis')
        # fig = sns.heatmap(data=t[0], cmap='viridis')  
        # heatmap = fig.get_figure()
        # # heatmap.savefig(output_images_path + name[0], dpi=400)
        # heatmap.savefig('./2.png', dpi=400)
        # heatmap.clear()
        # raise
        # ----------------------------------------------------------------

        # 上采样 + 通道数降为3
        # print(input.shape)
        # print(x.shape)
        # x = input * x * 0.7 + input * 0.3
        # print(input.shape, x.shape, x[:, 0, :, :].shape, x[:, 0, :, :].unsqueeze(1).shape)

        # x = input * x[:, 0, :, :].unsqueeze(1) - input * x[:, 1, :, :].unsqueeze(1)

        # temp= 0.5
        # N, C, H, W= x.shape
        # x = torch.abs(x) 
        # # Bs*W*H
        # x = x.mean(axis=1, keepdim=True)

        # temp= 0.5
        # N, C, H, W= mix2.shape
        # x = torch.abs(mix2) 
        # # Bs*W*H
        # x = x.mean(axis=1, keepdim=True)

        # x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        # x = (H * W * F.softmax((x/temp).view(N,-1), dim=1)).view(N, H, W)
        # print(x)
        # raise
        # x = x.unsqueeze(1)
        # print(x.shape)

        # import seaborn as sns
        # sns.set(font_scale=1.5)
        # t = x.detach().cpu()
        # t = t.data.cpu().numpy()
        # # fig = sns.heatmap(data=t[0], vmax=9., vmin=6.5, cmap='viridis') 
        # # fig = sns.heatmap(data=t[0], vmax=4., vmin=3.4, cmap='viridis')
        # fig = sns.heatmap(data=t[0].squeeze(), cmap='viridis')  
        # heatmap = fig.get_figure()
        # # heatmap.savefig(output_images_path + name[0], dpi=400)
        # heatmap.savefig('./1.png', dpi=400)
        # heatmap.clear()



        # x = torch.sigmoid(x)
        # x = (x - x.min()) / (x.max() - x.min())
        # x[x<0.5] = 0
        # x[x>0] = 1
        # print(x)
        # x = input * x
        x = x
        # x = mix2
        # from torchvision.utils import save_image
        # save_image(x, "@fuhao/transfer/input-x.jpg" , normalize=True)
        # save_image(input, "./input.jpg" , normalize=True)

        return x, mix2

    def forward(self, x):
        # self.ae.eval()
        # self.c1.eval()
        input = x
        x, mix2 = self.adaptive_ae(x)

        feat = x
        """Forward function."""
        if self.deep_stem:
            x = self.stem(input)
        else:
            x = self.conv1(input)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)

        x = x + mix2

        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                # if i == 0:
                #     x = x + mix2
                outs.append(x)
        return tuple(outs)


@BACKBONES.register_module()
class ResNet_v3(ResNet):
    
    def __init__(self,
                 depth,
                 **kwargs
                ):
        super(ResNet_v3, self).__init__(depth, **kwargs)
        
        self.c1 = nn.Sequential(
            nn.Conv2d(12, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            # nn.ReLU(inplace=True),
        )

        # self.c2 = nn.Sequential(
        #     nn.Conv2d(32, 64, 3, 2, 1),
        #     nn.BatchNorm2d(64),
        #     nn.GELU(),
        #     # nn.ReLU(inplace=True),
        # )

        self.ae = AE(conv=True)


        # self.custom_init_weights(self.c1)
        # self.custom_init_weights(self.ae)



    def partition(self, x):
        B, H, W, C = x.shape

        # padding
        # pad_input = (H % 2 == 1) or (W % 2 == 1)
        # if pad_input:
        #     x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, :, 0::2, 0::2]  # B C H/2 W/2 
        x1 = x[:, :, 1::2, 0::2]  # B C H/2 W/2 
        x2 = x[:, :, 0::2, 1::2]  # B C H/2 W/2 
        x3 = x[:, :, 1::2, 1::2]  # B C H/2 W/2 

        x = torch.cat([x0, x1, x2, x3], 1)  # B 4*C H/2 W/2

        return x

    def adaptive_ae(self, x):
        input = x
        
        # 下采样
        # print('ori_shape', input.shape)
        x = self.partition(x)
        # print(x.shape)
        x = self.c1(x)
        # x = self.c2(x)
        # print(x.shape)
        x, mix2 = self.ae(x)
        # print(x.shape)

        # """ mix3 归一化

        # S_t = mix2.pow(2).mean(1)
        # N_T,H_T,W_T = S_t.shape
        # S_t_min = S_t.detach().view(N_T, -1).min(1).values.unsqueeze(1).unsqueeze(1)
        # S_t_max = S_t.detach().view(N_T,  -1).max(1).values.unsqueeze(1).unsqueeze(1)

        # # print(S_t.shape, S_t_min.shape)

        # S_t = ( S_t - S_t_min ) / ( S_t_max - S_t_min + 1e-4) 
        mix2 = S_t
        # """


        # ----------------------------------------------------------------
        # x[x < x.mean()] = 0
        # print(x.shape)
        # from torchvision.utils import save_image
        # save_image(x, "./x.jpg" , normalize=True)
        # 上采样 + 通道数降为3
        # print(input.shape)
        # print(x.shape)
        # x = torch.sigmoid(x)
        # x = input * x + input

        # x = input * x
        
        # from torchvision.utils import save_image
        # save_image(input, "./input.jpg" , normalize=True)
        # save_image(x, "./input-x.jpg" , normalize=True)

        # import seaborn as sns
        # sns.set(font_scale=1.5)
        # t = input.detach().pow(2).mean(1)
        # t = t.data.cpu().numpy()
        # # fig = sns.heatmap(data=t[0], vmax=9., vmin=6.5, cmap='viridis') 
        # # fig = sns.heatmap(data=t[0], vmax=4., vmin=3.4, cmap='viridis')
        # fig = sns.heatmap(data=t[0], cmap='viridis')  
        # heatmap = fig.get_figure()
        # # heatmap.savefig(output_images_path + name[0], dpi=400)
        # heatmap.savefig('./1.png', dpi=400)
        # heatmap.clear()

        # t = x.detach().pow(2).mean(1)
        # t = t.data.cpu().numpy()
        # # fig = sns.heatmap(data=t[0], vmax=9., vmin=6.5, cmap='viridis') 
        # # fig = sns.heatmap(data=t[0], vmax=4., vmin=3.4, cmap='viridis')
        # fig = sns.heatmap(data=t[0], cmap='viridis')  
        # heatmap = fig.get_figure()
        # # heatmap.savefig(output_images_path + name[0], dpi=400)
        # heatmap.savefig('./2.png', dpi=400)
        # heatmap.clear()
        # raise
        # ----------------------------------------------------------------

        # 上采样 + 通道数降为3
        # print(input.shape)
        # print(x.shape)
        # x = input * x * 0.7 + input * 0.3
        # print(input.shape, x.shape, x[:, 0, :, :].shape, x[:, 0, :, :].unsqueeze(1).shape)

        # x = input * x[:, 0, :, :].unsqueeze(1) - input * x[:, 1, :, :].unsqueeze(1)

        # temp= 0.5
        # N, C, H, W= x.shape
        # x = torch.abs(x) 
        # # Bs*W*H
        # x = x.mean(axis=1, keepdim=True)

        # temp= 0.5
        # N, C, H, W= mix2.shape
        # x = torch.abs(mix2) 
        # # Bs*W*H
        # x = x.mean(axis=1, keepdim=True)
        # mix2 = mix2.unsqueeze(1)
        # print(mix2.shape)
        # mix2 = F.interpolate(mix2, scale_factor=4, mode='bilinear', align_corners=False)

        # mix2[mix2<0.2] = 0.
        # mix2[mix2>0.6] = 1.
        # x = (H * W * F.softmax((x/temp).view(N,-1), dim=1)).view(N, H, W)
        # print(x)
        # raise
        # x = x.unsqueeze(1)
        # print(x.shape)

        # import seaborn as sns
        # sns.set(font_scale=1.5)
        # t = x.detach().cpu()
        # t = t.data.cpu().numpy()
        # # fig = sns.heatmap(data=t[0], vmax=9., vmin=6.5, cmap='viridis') 
        # # fig = sns.heatmap(data=t[0], vmax=4., vmin=3.4, cmap='viridis')
        # fig = sns.heatmap(data=t[0].squeeze(), cmap='viridis')  
        # heatmap = fig.get_figure()
        # # heatmap.savefig(output_images_path + name[0], dpi=400)
        # heatmap.savefig('./1.png', dpi=400)
        # heatmap.clear()



        # x = torch.sigmoid(x)
        # x = (x - x.min()) / (x.max() - x.min())
        # x[x<0.5] = 0
        # x[x>0] = 1
        # print(x)
        # x = input * mix2 
        # x = x
        # x = mix2
        # from torchvision.utils import save_image
        # save_image(x, "@fuhao/transfer/input-x.jpg" , normalize=True)
        # save_image(input, "./input.jpg" , normalize=True)

        return x, mix2

    def forward(self, x):
        self.ae.eval()
        self.c1.eval()
        input = x
        x, mix2 = self.adaptive_ae(x)

        feat = x
        feat = feat.unsqueeze(1)
        """Forward function."""
        if self.deep_stem:
            x = self.stem(input)
        else:
            x = self.conv1(input)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)

        # x = x + mix2
        # x = x * mix2[mix2<0.3]
        x = x * feat

        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                # if i == 0:
                #     x = x + mix2
                outs.append(x)
        return tuple(outs)



@BACKBONES.register_module()
class ResNet_v4(ResNet):
    
    def __init__(self,
                 depth,
                 **kwargs
                ):
        super(ResNet_v4, self).__init__(depth, **kwargs)
        
        self.c1 = nn.Sequential(
            nn.Conv2d(12, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            # nn.ReLU(inplace=True),
        )

        # self.c2 = nn.Sequential(
        #     nn.Conv2d(32, 64, 3, 2, 1),
        #     nn.BatchNorm2d(64),
        #     nn.GELU(),
        #     # nn.ReLU(inplace=True),
        # )

        self.ae = AE(conv=True)


        # self.custom_init_weights(self.c1)
        # self.custom_init_weights(self.ae)



    def partition(self, x):
        B, H, W, C = x.shape

        # padding
        # pad_input = (H % 2 == 1) or (W % 2 == 1)
        # if pad_input:
        #     x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, :, 0::2, 0::2]  # B C H/2 W/2 
        x1 = x[:, :, 1::2, 0::2]  # B C H/2 W/2 
        x2 = x[:, :, 0::2, 1::2]  # B C H/2 W/2 
        x3 = x[:, :, 1::2, 1::2]  # B C H/2 W/2 

        x = torch.cat([x0, x1, x2, x3], 1)  # B 4*C H/2 W/2

        return x

    def adaptive_ae(self, x):
        input = x
        
        # 下采样
        # print('ori_shape', input.shape)
        x = self.partition(x)
        # print(x.shape)
        x = self.c1(x)
        # x = self.c2(x)
        # print(x.shape)
        mix2, mix3 = self.ae(x)
        # print(x.shape)

        # """ mix3 归一化

        # S_t = mix2.pow(2).mean(1)
        # N_T,H_T,W_T = S_t.shape
        # S_t_min = S_t.detach().view(N_T, -1).min(1).values.unsqueeze(1).unsqueeze(1)
        # S_t_max = S_t.detach().view(N_T,  -1).max(1).values.unsqueeze(1).unsqueeze(1)

        # # print(S_t.shape, S_t_min.shape)

        # S_t = ( S_t - S_t_min ) / ( S_t_max - S_t_min + 1e-4) 
        # mix2 = S_t
        # """


        # ----------------------------------------------------------------
        # x[x < x.mean()] = 0
        # print(x.shape)
        # from torchvision.utils import save_image
        # save_image(x, "./x.jpg" , normalize=True)
        # 上采样 + 通道数降为3
        # print(input.shape)
        # print(x.shape)
        # x = torch.sigmoid(x)
        # x = input * x + input

        # x = input * x
        
        # from torchvision.utils import save_image
        # save_image(input, "./input.jpg" , normalize=True)
        # save_image(x, "./input-x.jpg" , normalize=True)

        # import seaborn as sns
        # sns.set(font_scale=1.5)
        # t = input.detach().pow(2).mean(1)
        # t = t.data.cpu().numpy()
        # # fig = sns.heatmap(data=t[0], vmax=9., vmin=6.5, cmap='viridis') 
        # # fig = sns.heatmap(data=t[0], vmax=4., vmin=3.4, cmap='viridis')
        # fig = sns.heatmap(data=t[0], cmap='viridis')  
        # heatmap = fig.get_figure()
        # # heatmap.savefig(output_images_path + name[0], dpi=400)
        # heatmap.savefig('./1.png', dpi=400)
        # heatmap.clear()

        # t = x.detach().pow(2).mean(1)
        # t = t.data.cpu().numpy()
        # # fig = sns.heatmap(data=t[0], vmax=9., vmin=6.5, cmap='viridis') 
        # # fig = sns.heatmap(data=t[0], vmax=4., vmin=3.4, cmap='viridis')
        # fig = sns.heatmap(data=t[0], cmap='viridis')  
        # heatmap = fig.get_figure()
        # # heatmap.savefig(output_images_path + name[0], dpi=400)
        # heatmap.savefig('./2.png', dpi=400)
        # heatmap.clear()
        # raise
        # ----------------------------------------------------------------

        # 上采样 + 通道数降为3
        # print(input.shape)
        # print(x.shape)
        # x = input * x * 0.7 + input * 0.3
        # print(input.shape, x.shape, x[:, 0, :, :].shape, x[:, 0, :, :].unsqueeze(1).shape)

        # x = input * x[:, 0, :, :].unsqueeze(1) - input * x[:, 1, :, :].unsqueeze(1)

        # temp= 0.5
        # N, C, H, W= x.shape
        # x = torch.abs(x) 
        # # Bs*W*H
        # x = x.mean(axis=1, keepdim=True)

        # temp= 0.5
        # N, C, H, W= mix2.shape
        # x = torch.abs(mix2) 
        # # Bs*W*H
        # x = x.mean(axis=1, keepdim=True)
        # mix2 = mix2.unsqueeze(1)
        # print(mix2.shape)
        # mix2 = F.interpolate(mix2, scale_factor=4, mode='bilinear', align_corners=False)

        # mix2[mix2<0.2] = 0.
        # mix2[mix2>0.6] = 1.
        # x = (H * W * F.softmax((x/temp).view(N,-1), dim=1)).view(N, H, W)
        # print(x)
        # raise
        # x = x.unsqueeze(1)
        # print(x.shape)

        # import seaborn as sns
        # sns.set(font_scale=1.5)
        # t = x.detach().cpu()
        # t = t.data.cpu().numpy()
        # # fig = sns.heatmap(data=t[0], vmax=9., vmin=6.5, cmap='viridis') 
        # # fig = sns.heatmap(data=t[0], vmax=4., vmin=3.4, cmap='viridis')
        # fig = sns.heatmap(data=t[0].squeeze(), cmap='viridis')  
        # heatmap = fig.get_figure()
        # # heatmap.savefig(output_images_path + name[0], dpi=400)
        # heatmap.savefig('./1.png', dpi=400)
        # heatmap.clear()



        # x = torch.sigmoid(x)
        # x = (x - x.min()) / (x.max() - x.min())
        # x[x<0.5] = 0
        # x[x>0] = 1
        # print(x)
        # x = input * mix2 
        # x = x
        # x = mix2
        # from torchvision.utils import save_image
        # save_image(x, "@fuhao/transfer/input-x.jpg" , normalize=True)
        # save_image(input, "./input.jpg" , normalize=True)

        return x, mix2

    def forward(self, x):
        # self.ae.eval()
        # self.c1.eval()
        input = x
        mix2, mix3 = self.adaptive_ae(x)

        # feat = x
        # feat = feat.unsqueeze(1)
        # """Forward function."""
        # if self.deep_stem:
        #     x = self.stem(input)
        # else:
        #     x = self.conv1(input)
        #     x = self.norm1(x)
        #     x = self.relu(x)
        # x = self.maxpool(x)

        # x = x + mix2
        # x = x * mix2[mix2<0.3]
        x = mix2

        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)




@BACKBONES.register_module()
class ResNet_norm(ResNet):
    
    def __init__(self,
                 depth,
                 **kwargs
                ):
        super(ResNet_norm, self).__init__(depth, **kwargs)
        self.mean= torch.tensor([123.675, 116.28, 103.53]).cuda()
        self.std= torch.tensor([58.395, 57.12, 57.375]).cuda()

    def forward(self, x):
        # print(x.max(), x.min())
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

        # x = self.partition(x)
        # x = self.c1(x)

        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)