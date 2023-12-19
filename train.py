"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
from options.train_options import TrainOptions
import data
from util.iter_counter import IterationCounter
from trainers.CA2SIS_trainer import CA2SISTrainer
from util import util
import torchvision
import numpy as np
from torchvision import transforms
import os
import torch
import random

torch.backends.cudnn.benchmark = True

def print_current_errors(epoch, i, errors, t):
    message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
    for k, v in errors.items():
        #print(v)
        #if v != 0:
        v = v.mean().float()
        message += '%s: %.3f ' % (k, v)

    print(message, flush = True)


print("Start executing...", flush = True)

label_list = ['bkgrnd', 'skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth',
        'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
gen_parts = [1, 2, 4, 5, 6, 7, 10, 11, 12, 13]
part2swap = 1

# parse options
opt = TrainOptions().parse()

# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataloader = data.create_dataloader(opt)
print("Data Loaded!!!", flush = True)

# create trainer for our model
trainer = CA2SISTrainer(opt)

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))


for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()

        # Training
        # train generator
        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data_i)

        # train discriminator
        trainer.run_discriminator_one_step(data_i)

        if opt.use_ema:
            trainer.update_ema()

        # Visualizations
        with torch.no_grad():
            if i % opt.print_freq == 0:

                # print losses 
                losses = trainer.get_latest_losses()
                print_current_errors(epoch, iter_counter.epoch_iter,
                                        losses, iter_counter.time_per_iter)

                # save images
                p = random.choices(gen_parts, k=part2swap)
                if opt.use_noise:
                    rec_img = trainer.generate_with_noise(data_i, p)
                    if opt.att_loss:
                        rec_img = rec_img[0]
                else:
                    rec_img = trainer.run_generator_swapped(data_i, p)[0]
            
                grid_img_real = util.tensor2im(torchvision.utils.make_grid(data_i['image']).detach().cpu())
                grid_img_rec = util.tensor2im(torchvision.utils.make_grid(rec_img).detach().cpu())
                #grid_img_swap = util.tensor2im(torchvision.utils.make_grid(out_sw['fake_sw']).detach().cpu())                
                grid_list = [grid_img_real, grid_img_rec]


                grid = np.concatenate(grid_list, axis=0)
                    
                os.makedirs(opt.sample_dir + str(epoch) + '/', exist_ok=True)

                transforms.ToPILImage()(grid).save(opt.sample_dir + str(epoch) + '/' + str(i) + '_' + label_list[p[0]] + '.png')
                #transforms.ToPILImage()(grid).save(opt.sample_dir + str(epoch) + '/' + str(i) + '_' + '.png')

        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            iter_counter.record_current_iter()

    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    if epoch % opt.save_epoch_freq == 0 or \
       epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, iter_counter.total_steps_so_far))
        trainer.save('latest')
        trainer.save(epoch)

print('Training was successfully finished.')