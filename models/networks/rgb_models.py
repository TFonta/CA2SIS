import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from torch import Tensor
from torch.autograd import Variable

from inspect import isfunction
import math
from einops import rearrange, repeat

from models.networks.modules import res,att


class SingleLinearMaskEncoder(nn.Module):
    def __init__(self, latent_size, size = 128):
        super(SingleLinearMaskEncoder, self).__init__()  
        self.size = size 
        self.latent_size = latent_size

        self.encs = nn.Sequential(
            nn.Linear(self.size**2, self.latent_size)
        )
    
    def forward(self, x):
        x = nn.functional.interpolate(x, size = (self.size, self.size))
        x = x.view(x.size(0), x.size(1), -1)
        return self.encs(x)

class RGB_model(nn.Module):
    def __init__(self, opt, nc, ngf, ndf, latent_variable_size):
        super(RGB_model, self).__init__()
        #self.cuda = True
        #self.input_styles = opt.input_styles
        #self.linear_enc = opt.linear_enc
        #self.use_sean = opt.sean_style_encoder
        self.opt = opt
        
        self.parts_for_dec = nc

        if self.opt.contain_dontcare_label:
            self.nc = nc+1
        else:
            self.nc = nc

        self.ngf = ngf
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size
        self.norm = nn.InstanceNorm2d

        if self.opt.ds_mode == "cityscapes":
            self.linear_enc_size = 512
        else:
            self.linear_enc_size = 256
        
        self.encs = SingleLinearMaskEncoder(self.linear_enc_size)
        
        # style encoder        
        self.style_encoder = MultiScaleEffStyleEncoder(num_downsample=6, num_upsample=5, 
                                                    num_feat = 4, num_mask_channels = self.nc, 
                                                    output_dim = self.latent_variable_size)
        
        self.noise_encoder = MappingNetwork(latent_dim=self.latent_variable_size, style_dim=self.latent_variable_size*5, num_domains=self.nc)

        # decoder
        self.n_heads = 8
        self.d_head = 64
        
        if self.opt.exclude_bg:
            self.reshape_conv = nn.Conv2d(self.nc - 1, ngf*16, 1, 1)
        else:
            self.reshape_conv = nn.Conv2d(self.nc, ngf*16, 1, 1)
                
        self.latent_variable_size = self.latent_variable_size*5

        self.res2 = res.ResBlock(ngf*16, dropout=0, out_channels=ngf*8, dims=2, up=False)
        self.cross2 = att.SpatialTransformer(in_channels=ngf*8, n_heads=self.n_heads, d_head=self.d_head, depth=1,
            context_dim=self.latent_variable_size,feat_height=16, no_self_att=False)
        
        self.res3 = res.ResBlock(ngf*8, dropout=0, out_channels=ngf*4, dims=2, up=True)
        self.cross3 = att.SpatialTransformer(in_channels=ngf*4, n_heads=self.n_heads, d_head=self.d_head,depth=1,
            context_dim=self.latent_variable_size,feat_height=32, no_self_att=False)
        
        self.res4 = res.ResBlock(ngf*4, dropout=0, out_channels=ngf*2, dims=2, up=True)
        self.cross4 = att.SpatialTransformer(in_channels=ngf*2, n_heads=self.n_heads, d_head=self.d_head,depth=1,
            context_dim=self.latent_variable_size,feat_height=64, no_self_att=False)
        
        # from here no self attention
        self.res5 = res.ResBlock(ngf*2, dropout=0, out_channels=ngf, dims=2, up=True)       
        self.cross5 = att.SpatialTransformer(in_channels=ngf, n_heads=self.n_heads, d_head=self.d_head,depth=1,
            context_dim=self.latent_variable_size,feat_height=128,no_self_att = True)        
        self.res6 = res.ResBlock(ngf, dropout=0, out_channels=ngf, dims=2, up=True)
        self.cross6 = att.SpatialTransformer(in_channels=ngf, n_heads=4, d_head=32,depth=1,
            context_dim=self.latent_variable_size,feat_height=256,no_self_att = True)    
        self.out_conv = nn.Conv2d(ngf,3,1,1)

    def encode_mask_parts(self, x): #B,18,256,256 
        mu = self.encs(x)
        return mu
    def transformer_pass(self, mu):
        mu = mu + self.pos_enc
        mu = self.transformer_encoder(mu)
        return mu

    def encode(self, x):
        
        mu = self.encode_mask_parts(x)
        return mu
    
    def cross_att(self, out, s, c_layer, m = None):
        return c_layer(out, s, m)

    def decode(self, z, s, m = None):

        if self.opt.ds_mode == "cityscapes":
            out = z.view(z.size(0), z.size(1), 16, 32)
        else:
            out = z.view(z.size(0), z.size(1), 16, 16)

        out = self.reshape_conv(out)
        out = self.res2(out)
        out, loss2 = self.cross_att(out = out, s = s, c_layer = self.cross2, m = m)
        out = self.res3(out)
        out, loss3 = self.cross_att(out = out, s = s, c_layer = self.cross3, m = m)
        out = self.res4(out)
        out, loss4 = self.cross_att(out = out, s = s, c_layer = self.cross4, m = m)
        out = self.res5(out)
        out, loss5 = self.cross_att(out = out, s = s, c_layer = self.cross5, m = m)
        out = self.res6(out)
        out, loss6 = self.cross_att(out = out, s = s, c_layer = self.cross6, m = m)
        out = self.out_conv(out)
        att_loss = (loss2 + loss3 + loss4 + loss5 + loss6)/5
        return out, att_loss 

    def get_latent_var(self, x):
        mu = self.encode(x)
        return mu

    def forward_noise(self, z, m):
        m_enc = m.clone()        
        m_enc = m_enc[:,1:]
        e = self.encode(m_enc)
        s = self.noise_encoder(z)
        return self.decode(e,s,m), s

    def forward(self, rgb, m, m_sw, mode = None):
        #remove bg
        if self.opt.exclude_bg:
            m_sw = m_sw[:,1:]
        z = self.encode(m_sw)
        s = self.style_encoder(rgb, m)
        
        res = self.decode(z,s,m)

        return res # returns rgb, att_loss

class MultiScaleEffStyleEncoder(nn.Module):
    def __init__(self, input_channels = 3, num_mask_channels = 19, num_downsample = 4, num_upsample = 3, 
        num_feat = 4, output_dim = 256, kernel_dim = 3):
        super(MultiScaleEffStyleEncoder, self).__init__()

        self.nmc = num_mask_channels
        self.num_downsample = num_downsample
        self.num_upsample = num_upsample

        self.kernels = []
        for i in range(0,num_downsample):
            self.kernels += [(num_mask_channels*num_feat*(2**(i)), (num_mask_channels*num_feat*(2**(i+1))))]
        
        self.Encoder = nn.ModuleDict()
        self.Decoder = nn.ModuleDict()
        self.out = nn.ModuleDict()

        # input layer
        self.Encoder['first_layer'] = nn.Sequential(nn.Conv2d(input_channels, self.kernels[0][0], kernel_dim, padding=1),
                    nn.GroupNorm(self.nmc, self.kernels[0][0]),
                    nn.ReLU())
        
        # Encoding
        for i, (in_kernel, out_kernel) in enumerate(self.kernels):
            
            self.Encoder[f'enc_layer_{i}'] = nn.Sequential(nn.Conv2d(in_kernel,out_kernel, 3, 
                        stride = 2, padding=1, groups=self.nmc), 
                        nn.GroupNorm(self.nmc, out_kernel),
                        nn.ReLU())
        
        # Upsampling
        for i, (in_kernel, out_kernel) in reversed(list(enumerate(self.kernels))):
            prev_kernel = out_kernel
            if i == num_downsample - 1:
                prev_kernel = 0

            if i == (num_downsample - 1 - num_upsample):
                break
            self.Decoder[f'dec_layer_{num_downsample-1-i}'] = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear'),
                        nn.Conv2d(out_kernel + prev_kernel, in_kernel, 3, 
                        stride = 1, padding=1, groups=self.nmc),
                        nn.GroupNorm(self.nmc, in_kernel),
                        nn.ReLU())

        for i, (in_kernel, out_kernel) in reversed(list(enumerate(self.kernels))):
            if i == (num_downsample - 1 - num_upsample):
                break
            self.out[f'out_{num_downsample-1-i}'] = nn.Sequential(nn.Conv1d(in_kernel, output_dim*self.nmc, 1, 
                                groups=self.nmc), nn.Tanh())
                
        self.eps = 1e-5
    
    def forward(self, x, mask):    
        
        x = self.Encoder['first_layer'](x)

        enc_feat = []
        for i in range(self.num_downsample):
            x = self.Encoder[f'enc_layer_{i}'](x)
            enc_feat.append(x)

        dec_style_feat = []
        for i in range(self.num_upsample):
            x = self.Decoder[f'dec_layer_{i}'](x)

            _,_,side_h,side_w = x.shape
            mask_int = nn.functional.interpolate(mask, size=(side_h, side_w), mode='nearest')   
            repetitions = torch.tensor([self.kernels[self.num_downsample-1-i][0]//self.nmc]*self.nmc).to(mask.device)  
            mask_int = torch.repeat_interleave(mask_int, repeats=repetitions, dim=1)

            h = x * mask_int # B, G*19, H, W

            # pooling
            h = torch.sum(h, dim=(2,3))  # B, G*19     
            div = torch.sum(mask_int, dim=(2,3)) # B, G*19    
            h = h / (div + self.eps) 

            h = self.out[f'out_{i}'](h.unsqueeze(-1)) # B, 256*19, 1

            h = h.reshape((h.shape[0], self.nmc, h.shape[1]//self.nmc))
            dec_style_feat.append(h)

            # prepare skip connection
            x = torch.cat((x, enc_feat[self.num_upsample-1-i]), dim = 1)
        
        #[s1,s2,s3,s4,s5]
        return torch.cat(dec_style_feat, dim = 2)

class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=512, style_dim=512, num_domains=19):
        super().__init__()
        layers = []
        layers += [nn.Linear(latent_dim, 512)]
        layers += [nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(512, 512)]
            layers += [nn.ReLU()]
        self.shared = nn.Sequential(*layers)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Sequential(nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, style_dim))]

    def forward(self, z):
        h = self.shared(z)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        return torch.stack(out, dim=1) 


