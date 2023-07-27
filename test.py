"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
from options.test_options import TestOptions
import data
from models.CA2SIS_model import CA2SISModel
from util import util
from torchvision import transforms
import os
import torch
import random

def denorm(x):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def one_hot(targets, nclasses):
    targets_extend = targets.clone()        
    targets_extend.unsqueeze_(1)  # convert to Nx1xHxW
    one_hot = torch.FloatTensor(targets_extend.size(0), nclasses, targets_extend.size(2), targets_extend.size(3)).zero_()
    one_hot = one_hot.cuda()
    one_hot.scatter_(1, targets_extend, 1)
    return one_hot

# Fix the seed so to generate always the same swaps
# -------------------------------------------------
# -------------------------------------------------
#random.seed(1)
# -------------------------------------------------
# -------------------------------------------------

print("Start executing...", flush = True)

label_list = ['bkgrnd', 'skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth',
        'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
gen_parts = [1, 2, 4, 5, 6, 7, 10, 11, 12, 13]
part2swap = 1

# parse options
opt = TestOptions().parse() # --> test options? base options?
opt.status = 'test'

# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataloader = data.create_dataloader(opt)
print("Data Loaded!!!", flush = True)

# create model
model = CA2SISModel(opt)

model.eval()


# Creates three folders, one for reconstructions, one for swapped and one for style swaps
rec_dir = os.path.join(opt.sample_dir, 'rec/')
swap_dir = os.path.join(opt.sample_dir, 'swap/')
style_dir = os.path.join(opt.sample_dir, 'style/')

os.makedirs(rec_dir, exist_ok=True)
os.makedirs(swap_dir, exist_ok=True)
os.makedirs(style_dir, exist_ok=True)

num_samples = 0

with torch.no_grad():
    for i, data_i in enumerate(dataloader):

        num_s = len(data_i['image'])
        
        if num_s != opt.batchSize:
            continue

        num_samples += num_s
        img_name = [data_i['path'][j].split('/')[-1] for j in range(num_s)]

        if i*opt.batchSize % 64  == 0:
            print(i, flush = True)

        # Run generator
        p = random.choices(gen_parts, k=part2swap)
        generated = model(data_i, 'swap_part', p)

        style_swap = model(data_i, 'swap_styles', p)['fake_sw']
        
        for idx in range(num_s):
            ext = img_name[idx].split(".")[-1]
            
            transforms.ToPILImage()(util.tensor2im(generated['fake'][idx].squeeze().detach().cpu())).save(rec_dir + img_name[idx])
            src = img_name[idx].split('.')[0]
            if idx < opt.batchSize//2:
                dst = img_name[idx + opt.batchSize//2].split('.')[0]
            else:
                dst = img_name[idx - opt.batchSize//2].split('.')[0]

            transforms.ToPILImage()(util.tensor2im(generated['fake_sw'][idx].squeeze().detach().cpu())).save(swap_dir + '|' + src + '|' + dst + '|' + label_list[p[0]] + '.' + ext) 
            transforms.ToPILImage()(util.tensor2im(style_swap[idx].squeeze().detach().cpu())).save(style_dir + '|' + src + '|' + dst + '|' + label_list[p[0]] + '.' + ext)

print('Testing was successfully finished.', flush = True)