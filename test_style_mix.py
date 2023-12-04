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
import numpy as np
from torchvision.utils import save_image

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

print("Start executing...", flush = True)

label_list = ['bkgrnd', 'skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth',
        'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']

# parse options
opt = TestOptions().parse() # --> test options? base options?
opt.status = 'test'

# Flag to choose swap of style or shape
swap_type = opt.swap_type
# Choose part to swap for style
if opt.part_to_swap == 'eyes':
    p = [4,5]
elif opt.part_to_swap == 'mouth':
    p = [10,11,12]
elif opt.part_to_swap == 'hair':
    p = [13]
elif opt.part_to_swap == 'eyebrows':
    p = [6,7]
elif opt.part_to_swap == 'eyes-brows':
    p = [4,5,6,7]
elif opt.part_to_swap == 'skin':
    p = [1,2]
elif opt.part_to_swap == 'makeup':
    p =[5,12]
elif opt.part_to_swap == 'background':
    p = [0]
elif opt.part_to_swap == 'nose':
    p = [2]
elif opt.part_to_swap == 'expression':
    p = [10,11,12]
    
# TODO: 
# impostare anche lo swap di shapes

# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataloader = data.create_dataloader(opt)
print("Data Loaded!!!", flush = True)

# create model
model = CA2SISModel(opt)
model.eval()
print("Model Loaded!!!", flush = True)


# Creates two folders, one for reconstructions, one for swapped, inside the base one with the part name
base_dir = os.path.join(opt.sample_dir, swap_type, opt.part_to_swap)
os.makedirs(base_dir, exist_ok=True)
mask_dir = os.path.join(opt.sample_dir, swap_type, opt.part_to_swap, 'mask')
os.makedirs(mask_dir, exist_ok=True)
img_dir = os.path.join(opt.sample_dir, swap_type, opt.part_to_swap, 'img')
os.makedirs(img_dir, exist_ok=True)

# Select proper samples with correct semantic classes
labels = []
images = []
paths = []
instances = []
valid_imgs = 0
for i, data_i in enumerate(dataloader):
    num_s = len(data_i['image'])
    
    # Check if all masks have the semantic classes of interest
    for m, mask in enumerate(data_i['label']):
        if len(np.nonzero([torch.sum(mask == part) for part in p])[0]) == len(p):
            labels.append(data_i['label'][m])
            instances.append(data_i['instance'][m])
            images.append(data_i['image'][m])
            paths.append(data_i['path'][m])
            valid_imgs+=1

num_samples = 0  
max_samples = 1000;  
with torch.no_grad():
    for i in range(0,valid_imgs):
        
        if i+opt.batchSize > valid_imgs:
            break
        
        if num_samples >= max_samples:
            break
        
        index = [j for j in range(i,i+opt.batchSize)] + [j for j in range(i+opt.batchSize-1,i-1,-1)]
        data_i = {"label":[],"instance":[],"image":[],"path":[]}
        data_i['label'] = torch.stack([labels[j] for j in index])
        data_i['instance'] = torch.stack([instances[j] for j in index])
        data_i['image'] = torch.stack([images[j] for j in index])
        
        num_s = len(data_i['image'])
        num_samples += num_s
        img_name =[paths[j].split('/')[-1] for j in index]

        if i*opt.batchSize % 16  == 0:
            print(i, flush = True)
            
        # Run generator
        if swap_type == 'style':
            style_swap = model(data_i, 'swap_styles', p)
        elif swap_type == 'shape':
            style_swap = model(data_i, 'swap_part', p)
        else:
            style_swap = model(data_i, 'swap_all', p)
            
        for idx in range(num_s):
            ext = img_name[idx].split(".")[-1]
            
            src = img_name[idx].split('.')[0]
            if idx < num_s//2:
                dst = img_name[idx + num_s//2].split('.')[0]
            else:
                dst = img_name[idx - num_s//2].split('.')[0]

            transforms.ToPILImage()(util.tensor2im(style_swap['fake'][idx].squeeze().detach().cpu())).save(img_dir + '/|' + src + '|' + dst + '|' + label_list[p[0]] + '.' + ext) 
            save_image(torch.sum(style_swap['mask'][idx][p],dim=0),mask_dir + '/|' + src + '|' + dst + '|' + label_list[p[0]] + '.png')
            
print('Testing was successfully finished.', flush = True)