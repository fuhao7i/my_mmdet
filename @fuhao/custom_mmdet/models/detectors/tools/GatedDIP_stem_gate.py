import math 
import torch
import torchvision 
import torch.nn as nn
# from model.vision_encoder import VisionEncoder

class VisionEncoder(torch.nn.Module):
    def __init__(self,encoder_output_dim=256):
        super(VisionEncoder,self).__init__()
        # conv_1
        
        self.conv_1 = torch.nn.Sequential(torch.nn.Conv2d(3,64,kernel_size = 3 , stride = 2, padding=1),
                                        nn.BatchNorm2d(64),
                                        torch.nn.ReLU(True))
        
        # conv_2
        self.conv_2 = torch.nn.Sequential(torch.nn.Conv2d(64,128,kernel_size = 3 , stride = 2, padding=1),
                                          nn.BatchNorm2d(128),
                                        torch.nn.ReLU(True))

        # conv_3
        self.conv_3 = torch.nn.Sequential(torch.nn.Conv2d(128,256,kernel_size = 3 , stride = 2, padding=1),
                                          nn.BatchNorm2d(256),
                                        torch.nn.ReLU(True))
        
        # conv_4
        self.conv_4 = torch.nn.Sequential(torch.nn.Conv2d(256,512,kernel_size = 3 , stride = 2, padding=1),
                                          nn.BatchNorm2d(512),
                                        torch.nn.ReLU(True))

        
        # conv_5
        self.conv_5 = torch.nn.Sequential(torch.nn.Conv2d(512,1024,kernel_size = 3 , stride = 1),
                                        torch.nn.ReLU(True))
        self.adp_pool_5 = torch.nn.AdaptiveAvgPool2d((1,1))
        self.linear_proj_5 = torch.nn.Sequential(torch.nn.Linear(1024,encoder_output_dim),
                                                torch.nn.ReLU(True))
        

    def forward(self,x):
        out_x = self.conv_1(x)
        
        out_x = self.conv_2(out_x)
        
        out_x = self.conv_3(out_x)
        
        out_x = self.conv_4(out_x)
        
        out_x = self.conv_5(out_x)
        adp_pool_5 = self.adp_pool_5(out_x)
        linear_proj_5 = self.linear_proj_5(adp_pool_5.view(adp_pool_5.shape[0],-1))


        return linear_proj_5

# class VisionEncoder(torch.nn.Module):
#     def __init__(self,encoder_output_dim=256):
#         super(VisionEncoder,self).__init__()
#         # conv_1
#         self.conv_1 = torch.nn.Sequential(torch.nn.Conv2d(3,64,kernel_size = 3 , stride = 1),
#                                         torch.nn.ReLU(True))
#         self.max_pool_1 = torch.nn.AvgPool2d((3,3),(2,2))
        
#         # conv_2
#         self.conv_2 = torch.nn.Sequential(torch.nn.Conv2d(64,128,kernel_size = 3 , stride = 1),
#                                         torch.nn.ReLU(True))
#         self.max_pool_2 = torch.nn.AvgPool2d((3,3),(2,2))
#         # conv_3
#         self.conv_3 = torch.nn.Sequential(torch.nn.Conv2d(128,256,kernel_size = 3 , stride = 1),
#                                         torch.nn.ReLU(True))
#         self.max_pool_3 = torch.nn.AvgPool2d((3,3),(2,2))
        
#         # conv_4
#         self.conv_4 = torch.nn.Sequential(torch.nn.Conv2d(256,512,kernel_size = 3 , stride = 1),
#                                         torch.nn.ReLU(True))
#         self.max_pool_4 = torch.nn.AvgPool2d((3,3),(2,2))
        
#         # conv_5
#         self.conv_5 = torch.nn.Sequential(torch.nn.Conv2d(512,1024,kernel_size = 3 , stride = 1),
#                                         torch.nn.ReLU(True))
#         self.adp_pool_5 = torch.nn.AdaptiveAvgPool2d((1,1))
#         self.linear_proj_5 = torch.nn.Sequential(torch.nn.Linear(1024,encoder_output_dim),
#                                                 torch.nn.ReLU(True))
        

#     def forward(self,x):
#         out_x = self.conv_1(x)
#         max_pool_1 = self.max_pool_1(out_x)
        
#         out_x = self.conv_2(max_pool_1)
#         max_pool_2 = self.max_pool_2(out_x)
        
#         out_x = self.conv_3(max_pool_2)
#         max_pool_3 = self.max_pool_3(out_x)
        
#         out_x = self.conv_4(max_pool_3)
#         max_pool_4 = self.max_pool_4(out_x)
        
#         out_x = self.conv_5(max_pool_4)
#         adp_pool_5 = self.adp_pool_5(out_x)
#         linear_proj_5 = self.linear_proj_5(adp_pool_5.view(adp_pool_5.shape[0],-1))


#         return linear_proj_5


class GatedDIP_stem_gate(torch.nn.Module):
    """_summary_

    Args:
        torch (_type_): _description_
    """
    def __init__(self,
                encoder_output_dim : int = 256,
                num_of_gates : int = 7):
        """_summary_

        Args:
            encoder_output_dim (int, optional): _description_. Defaults to 256.
            num_of_gates (int, optional): _description_. Defaults to 7.
        """
        super(GatedDIP_stem_gate,self).__init__()
        print("GatedDIP with custom Encoder!!")

        # Encoder Model
        # self.encoder = torchvision.models.vgg16(pretrained=False)
        self.encoder = VisionEncoder(encoder_output_dim=encoder_output_dim)

        self.stem0 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(True),
            # nn.Conv2d(64, 64, 3, 2, 1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(True),
            # nn.Conv2d(64, 65, 3, 1, 1)
        )

        self.stem1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(True),
            # nn.Conv2d(64, 64, 3, 2, 1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(True),
            # nn.Conv2d(64, 65, 3, 1, 1)
        )

        self.stem2 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(True),
            # nn.Conv2d(64, 64, 3, 2, 1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(True),
            # nn.Conv2d(64, 65, 3, 1, 1)
        )

        self.stem3 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(True),
            # nn.Conv2d(64, 64, 3, 2, 1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(True),
            # nn.Conv2d(64, 65, 3, 1, 1)
        )

        self.stem4 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(True),
            # nn.Conv2d(64, 64, 3, 2, 1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(True),
            # nn.Conv2d(64, 65, 3, 1, 1)
        )

        self.stem5 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(True),
            # nn.Conv2d(64, 64, 3, 2, 1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(True),
            # nn.Conv2d(64, 65, 3, 1, 1)
        )
        
        self.stem6 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(True),
            # nn.Conv2d(64, 64, 3, 2, 1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(True),
            # nn.Conv2d(64, 65, 3, 1, 1)
        )

        self.stem7 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2, 1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(True),
            # nn.Conv2d(64, 64, 3, 2, 1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(True),
            # nn.Conv2d(64, 65, 3, 1, 1)
        )

        self.mid_layer = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
        )
        # Gating Module
        self.gate_module = torch.nn.Sequential(torch.nn.Linear(encoder_output_dim,num_of_gates,bias=True))
        
        # White-Balance Module
        self.wb_module = torch.nn.Sequential(torch.nn.Linear(encoder_output_dim,3,bias=True))
        
        # Gamma Module
        self.gamma_module = torch.nn.Sequential(torch.nn.Linear(encoder_output_dim,1,bias=True))
        
        # Sharpning Module
        self.gaussian_blur = torchvision.transforms.GaussianBlur(13, sigma=(0.1, 5.0))
        self.sharpning_module = torch.nn.Sequential(torch.nn.Linear(encoder_output_dim,1,bias=True))

        # De-Fogging Module
        self.defogging_module = torch.nn.Sequential(torch.nn.Linear(encoder_output_dim,1,bias=True))

        # Contrast Module
        self.contrast_module = torch.nn.Sequential(torch.nn.Linear(encoder_output_dim,1,bias=True))

        # Contrast Module
        self.tone_module = torch.nn.Sequential(torch.nn.Linear(encoder_output_dim,8,bias=True))
        
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()
    
    
    def init_weights(self):
        from mmcv.cnn import ConvModule, xavier_init
        from timm.models.layers import DropPath, to_2tuple, trunc_normal_
        from mmcv.cnn import (ConvModule, caffe2_xavier_init, constant_init, is_norm,
                      normal_init)
        from torch.nn.modules.batchnorm import _BatchNorm
        from mmcv.cnn import constant_init, kaiming_init
        """Initialize the weights of FPN module."""
        for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         xavier_init(m, distribution='uniform')
            
        # for m in self.fm5.modules():
            if isinstance(m, nn.Conv2d):
                # kaiming_init(m)
                xavier_init(m, distribution='uniform')
                print('init c_ ', end='')
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    
    def rgb2lum(self,img: torch.tensor):
        """_summary_

        Args:
            img (torch.tensor): _description_

        Returns:
            _type_: _description_
        """
        img = 0.27 * img[:, 0, :, :] + 0.67 * img[:, 1, :,:] + 0.06 * img[:, 2, :, :]
        return img 
    
    def lerp(self ,a : int , b : int , l : torch.tensor):
        return (1 - l.unsqueeze(2).unsqueeze(3)) * a + l.unsqueeze(2).unsqueeze(3) * b


    def dark_channel(self,x : torch.tensor):
        """_summary_

        Args:
            x (torch.tensor): _description_

        Returns:
            _type_: _description_
        """
        z = x.min(dim=1)[0].unsqueeze(1)
        return z 
    
    def atmospheric_light(self,x : torch.tensor,dark : torch.tensor ,top_k : int=1000):
        """_summary_

        Args:
            x (torch.tensor): _description_
            top_k (int, optional): _description_. Defaults to 1000.

        Returns:
            _type_: _description_
        """
        h,w = x.shape[2],x.shape[3]
        imsz = h * w 
        numpx = int(max(math.floor(imsz/top_k),1))
        darkvec = dark.reshape(x.shape[0],imsz,1)
        imvec = x.reshape(x.shape[0],3,imsz).transpose(1,2)
        indices = darkvec.argsort(1)
        indices = indices[:,imsz-numpx:imsz]
        atmsum = torch.zeros([x.shape[0],1,3]).cuda()
        # print(imvec[:,indices[0,0]].shape)
        for b in range(x.shape[0]):
            for ind in range(1,numpx):
                atmsum[b,:,:] = atmsum[b,:,:] + imvec[b,indices[b,ind],:]
        a = atmsum/numpx
        a = a.squeeze(1).unsqueeze(2).unsqueeze(3)
        return a
    
    def blur(self,x : torch.tensor):
        """_summary_

        Args:
            x (torch.tensor): _description_

        Returns:
            _type_: _description_
        """
        return self.gaussian_blur(x)


    def defog(self,x:torch.tensor ,latent_out : torch.tensor ,fog_gate : torch.tensor):
        """Defogging module is used for removing the fog from the image using ASM 
        (Atmospheric Scattering Model).
        I(X) = (1-T(X)) * J(X) + T(X) * A(X)
        I(X) => image containing the fog.
        T(X) => Transmission map of the image.
        J(X) => True image Radiance.
        A(X) => Atmospheric scattering factor.

        Args:
            x (torch.tensor): Input image I(X)
            latent_out (torch.tensor): Feature representation from DIP Module.
            fog_gate (torch.tensor): Gate value raning from (0. - 1.) which enables defog module.

        Returns:
            torch.tensor : Returns defogged image with true image radiance.
        """
        omega = self.defogging_module(latent_out).unsqueeze(2).unsqueeze(3)
        omega = self.tanh_range(omega,torch.tensor(0.1).cuda(),torch.tensor(1.).cuda())
        dark_i = self.dark_channel(x) 
        a = self.atmospheric_light(x,dark_i)
        i = x/a 
        i = self.dark_channel(i)
        t = 1. - (omega*i)
        j = ((x-a)/(torch.maximum(t,torch.tensor(0.01).cuda())))+a
        j = (j - j.min())/(j.max()-j.min())
        # j = j* fog_gate.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return j
        
    def white_balance(self,x : torch.tensor,latent_out : torch.tensor ,wb_gate: torch.tensor):
        """ White balance of the image is predicted using latent output of an encoder.

        Args:
            x (torch.tensor): Input RGB image.
            latent_out (torch.tensor): Output from the last layer of an encoder.
            wb_gate (torch.tensor): White-balance gate used to change the influence of color scaled image.

        Returns:
            torch.tensor: returns White-Balanced image. 
        """
        log_wb_range = 0.5
        wb = self.wb_module(latent_out)
        wb = torch.exp(self.tanh_range(wb,-log_wb_range,log_wb_range))
        
        color_scaling = 1./(1e-5 + 0.27 * wb[:, 0] + 0.67 * wb[:, 1] +
        0.06 * wb[:, 2])
        wb = color_scaling.unsqueeze(1)*wb
        wb_out = wb.unsqueeze(2).unsqueeze(3)*x
        wb_out = (wb_out-wb_out.min())/(wb_out.max()-wb_out.min())
        # wb_out = wb_gate.unsqueeze(1).unsqueeze(2).unsqueeze(3)*wb_out
        return wb_out

    def tanh01(self,x : torch.tensor):
        """_summary_

        Args:
            x (torch.tensor): _description_

        Returns:
            _type_: _description_
        """
        return torch.tanh(x)*0.5+0.5

    def tanh_range(self,x : torch.tensor,left : float,right : float):
        """_summary_

        Args:
            x (torch.tensor): _description_
            left (float): _description_
            right (float): _description_

        Returns:
            _type_: _description_
        """
        return self.tanh01(x)*(right-left)+ left

    def gamma_balance(self,x : torch.tensor,latent_out : torch.tensor,gamma_gate : torch.tensor):
        """_summary_

        Args:
            x (torch.tensor): _description_
            latent_out (torch.tensor): _description_
            gamma_gate (torch.tensor): _description_

        Returns:
            _type_: _description_
        """
        log_gamma = torch.log(torch.tensor(2.5).cuda())
        gamma = self.gamma_module(latent_out).unsqueeze(2).unsqueeze(3)
        gamma = torch.exp(self.tanh_range(gamma,-log_gamma,log_gamma))
        g = torch.pow(torch.maximum(x,torch.tensor(1e-4).cuda()),gamma)
        g = (g-g.min())/(g.max()-g.min())
        # g = g*gamma_gate.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return g
    
    def sharpning(self,x : torch.tensor,latent_out: torch.tensor,sharpning_gate : torch.tensor):
        """_summary_

        Args:
            x (torch.tensor): _description_
            latent_out (torch.tensor): _description_
            sharpning_gate (torch.tensor): _description_

        Returns:
            _type_: _description_
        """
        out_x = self.blur(x)
        y = self.sharpning_module(latent_out).unsqueeze(2).unsqueeze(3)
        y = self.tanh_range(y,torch.tensor(0.1).cuda(),torch.tensor(1.).cuda())
        s = x + (y*(x-out_x))
        s = (s-s.min())/(s.max()-s.min())
        # s = s * (sharpning_gate.unsqueeze(1).unsqueeze(2).unsqueeze(3))
        return s
    
    def identity(self,x : torch.tensor,identity_gate : torch.tensor):
        """_summary_

        Args:
            x (torch.tensor): _description_
            identity_gate (torch.tensor): _description_

        Returns:
            _type_: _description_
        """
        # x = x*identity_gate.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return x
    
    def contrast(self,x : torch.tensor,latent_out : torch.tensor,contrast_gate : torch.tensor):
        """_summary_

        Args:
            x (torch.tensor): _description_
            latent_out (torch.tensor): _description_
            contrast_gate (torch.tensor): _description_

        Returns:
            _type_: _description_
        """
        alpha = torch.tanh(self.contrast_module(latent_out))
        luminance = torch.minimum(torch.maximum(self.rgb2lum(x), torch.tensor(0.0).cuda()), torch.tensor(1.0).cuda()).unsqueeze(1)
        contrast_lum = -torch.cos(math.pi * luminance) * 0.5 + 0.5
        contrast_image = x / (luminance + 1e-6) * contrast_lum
        contrast_image = self.lerp(x, contrast_image, alpha) 
        contrast_image = (contrast_image-contrast_image.min())/(contrast_image.max()-contrast_image.min())
        # contrast_image = contrast_image * contrast_gate.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return contrast_image
    
    def tone(self,x : torch.tensor,latent_out : torch.tensor,tone_gate : torch.tensor):
        """_summary_

        Args:
            x (torch.tensor): _description_
            latent_out (torch.tensor): _description_
            tone_gate (torch.tensor): _description_

        Returns:
            _type_: _description_
        """
        curve_steps = 8
        tone_curve = self.tone_module(latent_out).reshape(-1,1,curve_steps)
        tone_curve = self.tanh_range(tone_curve,0.5, 2)
        tone_curve_sum = torch.sum(tone_curve, dim=2) + 1e-30
        total_image = x * 0
        for i in range(curve_steps):
            total_image += torch.clamp(x - 1.0 * i /curve_steps, 0, 1.0 /curve_steps) \
                            * tone_curve[:,:,i].unsqueeze(2).unsqueeze(3)
        total_image *= curve_steps / tone_curve_sum.unsqueeze(2).unsqueeze(3)
        total_image = (total_image-total_image.min())/(total_image.max()-total_image.min())
        # total_image = total_image * tone_gate.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return total_image

    def normalize01(self, x):
        x = (x-x.min())/(x.max()-x.min())
        return x
    
    def forward(self, x : torch.Tensor):
        """_summary_

        Args:
            x (torch.Tensor): _description_

        Returns:
            _type_: _description_
        """
        # print(x.max(), x.min())
        x = self.normalize01(x)
        # latent_out = torch.nn.functional.relu_(self.encoder(x))
        latent_out = self.encoder(x)
        gate = self.tanh_range(self.gate_module(latent_out),0.01,1.0)
        # gate = self.gate_module(latent_out)
        # gate = self.softmax(gate)
        wb_out = self.white_balance(x,latent_out,gate[:,0])
        gamma_out = self.gamma_balance(x,latent_out,gate[:,1])
        identity_out = self.identity(x,gate[:,2])
        sharpning_out = self.sharpning(x,latent_out,gate[:,3])
        fog_out = self.defog(x,latent_out,gate[:,4])
        contrast_out = self.contrast(x,latent_out,gate[:,5])
        tone_out = self.tone(x,latent_out,gate[:,6])
        
        # x = wb_out + gamma_out   + fog_out + sharpning_out + contrast_out + tone_out + identity_out
        
        # wb_out
        wb_out = self.stem0(wb_out)
        # wb_out = self.normalize01(wb_out)
        wb_out = wb_out * gate[:, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        # gamma_out
        gamma_out = self.stem1(gamma_out)
        # gamma_out = self.normalize01(gamma_out)
        gamma_out = gamma_out * gate[:, 1].unsqueeze(1).unsqueeze(2).unsqueeze(3)  
        # identity_out
        a = self.stem2(identity_out)
        # a = self.normalize01(a)
        identity_out = a * gate[:, 2].unsqueeze(1).unsqueeze(2).unsqueeze(3)  
        # sharpning_out
        a = self.stem3(sharpning_out)
        # a = self.normalize01(a)
        sharpning_out = a * gate[:, 3].unsqueeze(1).unsqueeze(2).unsqueeze(3)       
        # fog_out
        a = self.stem4(fog_out)
        # a = self.normalize01(a)
        fog_out = a * gate[:, 4].unsqueeze(1).unsqueeze(2).unsqueeze(3)  
        # contrast_out
        a = self.stem5(contrast_out)
        # a = self.normalize01(a)
        contrast_out = a * gate[:, 5].unsqueeze(1).unsqueeze(2).unsqueeze(3)  
        # tone_out
        a = self.stem6(tone_out)
        # a = self.normalize01(a)
        tone_out = a * gate[:, 6].unsqueeze(1).unsqueeze(2).unsqueeze(3)  
        
        # x = torch.cat([wb_out, gamma_out, fog_out, sharpning_out, contrast_out, tone_out, identity_out], 1)
        x = wb_out + gamma_out + fog_out + sharpning_out + contrast_out + tone_out + identity_out 
        
        # x = self.normalize01(x)
        # mean = torch.mean(torch.mean(x, dim=3, keepdim=True), dim=2, keepdim=True)
        # std = torch.std(torch.std(x, dim=3, keepdim=True), dim=2, keepdim=True)
        # x = (x - mean) / std
        # print(x.max(), x.min())
        # x = (x-x.min())/(x.max()-x.min())
        x = self.mid_layer(x)
        
        return x,gate



class Yolov3GatedDIP(torch.nn.Module):
    def __init__(self):
        super(Yolov3GatedDIP,self).__init__()
        print("Yolov3GatedDIP with custom Encoder for 10 classes!!")
        self.gated_dip = GatedDIP(256)
        self.yolov3 = Yolov3()
    
    def forward(self,x):
        out_x,gates = self.gated_dip(x)
        p,p_d = self.yolov3(out_x)
        return out_x,gates,p,p_d

if __name__ == '__main__':
    batch_size = 2
    encoder_out_dim = 256
    x = torch.randn(batch_size,3,448,448).cuda()
    x = (x-x.min())/(x.max()-x.min())
    model = GatedDIP_stem(encoder_output_dim = encoder_out_dim).cuda()
    print(model)
    out,gate= model(x)
    print('out shape:',out.shape)
    print('gate shape:',gate.shape)