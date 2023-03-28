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

class LinearMaskEncoder(nn.Module):
    def __init__(self, latent_size, size = 64):
        super(LinearMaskEncoder, self).__init__()  
        self.size = size 
        self.latent_size = latent_size

        self.encs = nn.Sequential(
            nn.Linear(self.size**2, self.latent_size),
            nn.ReLU(),
            nn.Linear(self.latent_size, self.latent_size),
            nn.ReLU(),
            nn.Linear(self.latent_size, self.latent_size)
        )
    
    def forward(self, x):
        x = nn.functional.interpolate(x, size = (self.size, self.size))
        x = x.view(x.size(0), x.size(1), -1)
        return self.encs(x)

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

class MaskEncoder(nn.Module):
    def __init__(self, ndf):
        super(MaskEncoder, self).__init__()    
        
        self.encs = nn.Sequential(
            nn.Conv2d(1, ndf, 4, 2, 1), 
            self.norm(ndf),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf, ndf*2, 4, 2, 1),
            self.norm(ndf*2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1),
            self.norm(ndf*4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1),
            self.norm(ndf*8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf*8, ndf*16, 4, 2, 1),
            self.norm(ndf*16),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf*16, ndf*32, 4, 2, 1),
            self.norm(ndf*32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(ndf*32, ndf*64, 4, 2, 1),
            self.norm(ndf*64),
            nn.LeakyReLU(0.2)
        )
    def forward(self, x):
        return self.encs(x)

class RGB_model(nn.Module):
    def __init__(self, opt, nc, ngf, ndf, latent_variable_size, nc_input = 1, nc_output = 18):
        super(RGB_model, self).__init__()
        #self.cuda = True
        self.input_styles = opt.input_styles
        self.linear_enc = opt.linear_enc
        self.use_sean = opt.sean_style_encoder
        self.use_T = opt.use_T
        self.chann_out_21 = opt.chann_out_21
        self.cross_att_all_layers = opt.cross_att_all_layers
        self.sek = opt.style_enc_kernel
        self.sefd = opt.style_enc_feat_dim
        self.multi_style = opt.multi_scale_style_enc
        self.single_layer_mask_enc = opt.single_layer_mask_enc
        self.no_self_last_layers = opt.no_self_last_layers
        self.use_noise = opt.use_noise
        self.exclude_bg = opt.exclude_bg
        self.generate_masks = opt.generate_masks
        self.elegant_solution = opt.elegant_solution
        self.ds_mode = opt.dataset_mode
        self.no_embedding = opt.no_embedding

        if self.input_styles == False:            
            self.parts_for_dec = nc
        else:
            self.parts_for_dec = nc + 1
        
        if opt.contain_dontcare_label:
            self.nc = nc+1
        else:
            self.nc = nc

        self.ngf = ngf
        self.ndf = ndf
        self.latent_variable_size = latent_variable_size
        self.norm = nn.InstanceNorm2d

        if self.ds_mode == "cityscapes":
            self.linear_enc_size = 512
        else:
            self.linear_enc_size = 256

        # encoder
        if self.linear_enc == False and self.single_layer_mask_enc == False:
            self.encs = MaskEncoder(self.ndf)
            self.vaes = nn.Linear(ndf*64*2*2, self.linear_enc_size)
        elif self.single_layer_mask_enc == True:
            self.encs = SingleLinearMaskEncoder(self.linear_enc_size)
        else:
            self.encs = LinearMaskEncoder(self.linear_enc_size)

        # style encoder
        if self.use_sean:
            self.style_encoder = Zencoder(input_nc = 3, output_nc = self.latent_variable_size)
        elif self.multi_style:
            if self.cross_att_all_layers:
                self.style_encoder = MultiScaleEffStyleEncoder(num_downsample=6, num_upsample=5, num_feat = self.sefd, num_mask_channels = self.nc, 
                output_dim = self.latent_variable_size, elegant_solution = self.elegant_solution)
            else:
                self.style_encoder = MultiScaleEffStyleEncoder(num_downsample=4, num_upsample=3, num_feat = self.sefd, num_mask_channels = self.nc, 
                output_dim = self.latent_variable_size, elegant_solution = self.elegant_solution)
        else:
            self.style_encoder = EffStyleEncoder(output_dim = self.latent_variable_size, num_feat = self.sefd, kernel_dim = self.sek, num_mask_channels = self.nc)                

        if self.use_noise:
            self.noise_encoder = MappingNetwork(style_dim = self.latent_variable_size, num_domains=self.nc)

        if self.generate_masks:
            if self.exclude_bg:
                self.mask_encoder = MappingNetwork(style_dim = 256, num_domains=self.nc-1)
            else:
                self.mask_encoder = MappingNetwork(style_dim = 256, num_domains=self.nc)

        # decoder
        self.n_heads = 8
        self.d_head = 64
        
        if self.exclude_bg:
            self.reshape_conv = nn.Conv2d(self.nc - 1, ngf*16, 1, 1)
        else:
            self.reshape_conv = nn.Conv2d(self.nc, ngf*16, 1, 1)

        if self.multi_style and not self.elegant_solution:
            self.latent_variable_size = self.latent_variable_size*5

        self.res2 = res.ResBlock(ngf*16, dropout=0, out_channels=ngf*8, dims=2, up=False)
        self.cross2 = att.SpatialTransformer(opt = opt, in_channels=ngf*8, n_heads=self.n_heads, d_head=self.d_head, depth=1,
            context_dim=self.latent_variable_size,feat_height=16)
        
        self.res3 = res.ResBlock(ngf*8, dropout=0, out_channels=ngf*4, dims=2, up=True)
        self.cross3 = att.SpatialTransformer(opt = opt, in_channels=ngf*4, n_heads=self.n_heads, d_head=self.d_head,depth=1,
            context_dim=self.latent_variable_size,feat_height=32)
        
        self.res4 = res.ResBlock(ngf*4, dropout=0, out_channels=ngf*2, dims=2, up=True)
        self.cross4 = att.SpatialTransformer(opt = opt, in_channels=ngf*2, n_heads=self.n_heads, d_head=self.d_head,depth=1,
            context_dim=self.latent_variable_size,feat_height=64)
        
        if self.no_self_last_layers:
            opt.no_self_att = True

        self.res5 = res.ResBlock(ngf*2, dropout=0, out_channels=ngf, dims=2, up=True)       
        if self.cross_att_all_layers:
            self.cross5 = att.SpatialTransformer(opt = opt, in_channels=ngf, n_heads=self.n_heads, d_head=self.d_head,depth=1,
                context_dim=self.latent_variable_size,feat_height=128)
        
        self.res6 = res.ResBlock(ngf, dropout=0, out_channels=ngf, dims=2, up=True)
        if self.cross_att_all_layers:
            self.cross6 = att.SpatialTransformer(opt = opt, in_channels=ngf, n_heads=4, d_head=32,depth=1,
                context_dim=self.latent_variable_size,feat_height=256)
        
        if self.chann_out_21:
            self.out_conv = nn.Conv2d(ngf, 3+18,1,1)
        else:
            self.out_conv = nn.Conv2d(ngf,3,1,1)


        if self.use_T:
            self.pos_enc = nn.Parameter(torch.zeros(1, self.nc, self.latent_variable_size))
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.latent_variable_size, nhead=8, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)

    def encode_mask_parts(self, x): #B,18,256,256 
        mu = self.encs(x)
        return mu
    def transformer_pass(self, mu):
        mu = mu + self.pos_enc
        mu = self.transformer_encoder(mu)
        return mu

    def encode(self, x):
        
        mu = self.encode_mask_parts(x)

        if self.use_T:
            mu = self.transformer_pass(mu)
        
        return mu
    
    def cross_att(self, out, s, c_layer):
        return c_layer(out,s)

    def decode(self, z, s):
        if not self.no_embedding:
            if self.ds_mode == "cityscapes":
                out = z.view(z.size(0), z.size(1), 16, 32)
            else:
                out = z.view(z.size(0), z.size(1), 16, 16)
        else:
            out = z
        out = self.reshape_conv(out)
        out = self.res2(out)
        out = self.cross_att(out = out, s = s, c_layer = self.cross2)
        out = self.res3(out)
        out = self.cross_att(out = out, s = s, c_layer = self.cross3)
        out = self.res4(out)
        out = self.cross_att(out = out, s = s, c_layer = self.cross4)
        out = self.res5(out)
        if self.cross_att_all_layers:
            out = self.cross_att(out = out, s = s, c_layer = self.cross5)
        out = self.res6(out)
        if self.cross_att_all_layers:
            out = self.cross_att(out = out, s = s, c_layer = self.cross6)
        out = self.out_conv(out)
        return out 

    def get_latent_var(self, x):
        mu = self.encode(x)
        return mu

    def forward_noise(self, z, m):
        if self.exclude_bg:
            m = m[:,1:]
        e = self.encode(m)
        s = self.noise_encoder(z)
        return self.decode(e,s)

    def forward_mask(self, z, rgb, m):
        m_gen = self.mask_encoder(z)
        s = self.style_encoder(rgb, m)
        res = self.decode(m_gen,s)
        return res

    def forward(self, rgb, m, m_sw, mode = None):
        #remove bg
        if self.exclude_bg:
            m_sw = m_sw[:,1:]
        if not self.no_embedding:
            z = self.encode(m_sw)
        else:
            z = nn.functional.interpolate(m_sw, size=(16, 16), mode='nearest') 
        s = self.style_encoder(rgb, m)

        if self.input_styles == False:
            res = self.decode(z,s)
        else:
            res = self.decode(s,z)
        return res 




class Zencoder(torch.nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=2, norm_layer=nn.InstanceNorm2d):
        super(Zencoder, self).__init__()
        self.output_nc = output_nc

        model = [nn.ReflectionPad2d(1), nn.Conv2d(input_nc, ngf, kernel_size=3, padding=0),
                 norm_layer(ngf), nn.LeakyReLU(0.2, False)]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), nn.LeakyReLU(0.2, False)]

        ### upsample
        for i in range(1):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), nn.LeakyReLU(0.2, False)]

        model += [nn.ReflectionPad2d(1), nn.Conv2d(256, output_nc, kernel_size=3, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)


    def forward(self, input, segmap):

        codes = self.model(input)

        segmap = F.interpolate(segmap, size=codes.size()[2:], mode='nearest')

        # print(segmap.shape)
        # print(codes.shape)


        b_size = codes.shape[0]
        # h_size = codes.shape[2]
        # w_size = codes.shape[3]
        f_size = codes.shape[1]

        s_size = segmap.shape[1]

        codes_vector = torch.zeros((b_size, s_size, f_size), dtype=codes.dtype, device=codes.device)

        # codes = codes.unsqueeze(1)
        # segmap = segmap.unsqueeze(2)
        
        # codes_vector = torch.sum(codes*segmap, dim = (3,4))/torch.sum(segmap, dim = (3,4))

        for i in range(b_size):
            for j in range(s_size):
                component_mask_area = torch.sum(segmap.bool()[i, j])

                if component_mask_area > 0:
                    codes_component_feature = codes[i].masked_select(segmap.bool()[i, j]).reshape(f_size,  component_mask_area).mean(1)
                    codes_vector[i][j] = codes_component_feature

                    #codes_avg[i].masked_scatter_(segmap.bool()[i, j], codes_component_mu)

        return codes_vector


class EffStyleEncoder(nn.Module):
    def __init__(self, input_channels = 3, num_mask_channels = 19, num_downsample = 2, num_upsample = 1, 
        num_feat = 16, output_dim = 256, kernel_dim = 3):
        super(EffStyleEncoder, self).__init__()

        self.nmc = num_mask_channels

        self.kernels = []
        for i in range(1,num_downsample+1):
            self.kernels += [(num_mask_channels*num_feat*(2**(i)), (num_mask_channels*num_feat*(2**(i+1))))]
        
        # input layer
        model = [nn.Conv2d(input_channels, self.kernels[0][0], kernel_dim),
                    nn.GroupNorm(self.nmc, self.kernels[0][0]),
                    nn.ReLU()] 
        
        # Encoding
        for in_kernel, out_kernel in self.kernels:
            model += [nn.Conv2d(in_kernel,out_kernel, 3, 
                        stride = 2, padding=1, groups=num_mask_channels), 
                        nn.GroupNorm(self.nmc, out_kernel),
                        nn.ReLU()]
        
        # Upsampling
        for i in range(num_upsample):
            model += [nn.Upsample(scale_factor=2, mode='bilinear'),
                        nn.Conv2d(self.kernels[-1][-1],self.kernels[-1][-1], 3, 
                        stride = 1, padding=1, groups=num_mask_channels),
                        nn.GroupNorm(self.nmc, self.kernels[-1][-1]),
                        nn.ReLU()]

        self.model = nn.Sequential(*model)
        
        self.out = nn.Conv1d(self.kernels[-1][-1], output_dim*num_mask_channels, 1, 
                             groups=num_mask_channels)
                
        self.eps = 1e-5
    
    def forward(self, x, mask):    

        h = self.model(x)

        side = h.shape[-1]
        mask = nn.functional.interpolate(mask, size=(side, side), mode='nearest')   
        repetitions = torch.tensor([self.kernels[-1][-1]//self.nmc]*self.nmc).to(mask.device)  

        mask = torch.repeat_interleave(mask, repeats=repetitions, dim=1)
        

        h = h * mask # B, G*19, H, W
        # pooling
        h = torch.sum(h, dim=(2,3))  # B, G*19     
        div = torch.sum(mask, dim=(2,3)) # B, G*19    
        h = h / (div + self.eps) 

        h = self.out(h.unsqueeze(-1)) # B, 256*19, 1
        h = nn.functional.tanh(h)

        h = h.reshape((h.shape[0], self.nmc, h.shape[1]//self.nmc)) # B, 19, 256
    
        return h

class MultiScaleEffStyleEncoder(nn.Module):
    def __init__(self, input_channels = 3, num_mask_channels = 19, num_downsample = 4, num_upsample = 3, 
        num_feat = 4, output_dim = 256, kernel_dim = 3, elegant_solution = True):
        super(MultiScaleEffStyleEncoder, self).__init__()

        self.nmc = num_mask_channels
        self.num_downsample = num_downsample
        self.num_upsample = num_upsample
        self.elegant_solution = elegant_solution

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
        if self.elegant_solution:
            return torch.cat(dec_style_feat, dim = 1)
        else:
            return torch.cat(dec_style_feat, dim = 2)

        #return torch.stack(dec_style_feat) #5xbsx19x256



class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=16, style_dim=256, num_domains=19):
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
        return torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
