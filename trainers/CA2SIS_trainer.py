"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch
import os
from models.networks.sync_batchnorm import DataParallelWithCallback
from models.CA2SIS_model import CA2SISModel
from ema_pytorch import EMA
class CA2SISTrainer():
    """
    Trainer creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """


    def __init__(self, opt):
        self.opt = opt
        self.pix2pix_model = CA2SISModel(opt)
        print("Model initialized!!!", flush = True)
        if len(opt.gpu_ids) > 1:
            self.pix2pix_model = DataParallelWithCallback(self.pix2pix_model,
                                                          device_ids=opt.gpu_ids)

            self.pix2pix_model_on_one_gpu = self.pix2pix_model.module
        else:
            self.pix2pix_model_on_one_gpu = self.pix2pix_model

        print("Model on GPU!!!", flush = True)

        self.generated = None
        if opt.isTrain:
            self.optimizer_G, self.optimizer_D = \
                self.pix2pix_model_on_one_gpu.create_optimizers(opt)
            self.old_lr = opt.lr

        if self.opt.use_ema:
            self.load_ema()

    def load_ema(self):
        self.ema = EMA(self.pix2pix_model_on_one_gpu,
                  beta = 0.9999,              # exponential moving average factor
                  update_after_step = 100,    # only after this number of .update() calls will it start updating
                  update_every = 10,          # how often to actually update, to save on compute (updates every 10th .update() call)
                  power = 3 / 4              # decay factor to prevent the EMA from ever fully "catching up" with the raw model
                )
        print("EMA initialized!!!", flush = True)
        
        if self.opt.continue_train or self.opt.isTrain==False:
            self.ema.load_state_dict(torch.load(os.path.join(self.opt.checkpoints_dir, 'ema.pth')))
            print("EMA loaded!!!", flush = True)

    def update_ema(self):
        if self.opt.use_ema:
            self.ema.update()

    def run_generator_one_step(self, data):
        self.optimizer_G.zero_grad()
        g_losses, generated = self.pix2pix_model(data, mode='generator')
        g_loss = sum(g_losses.values()).mean()
        g_loss.backward()
        self.optimizer_G.step()
        self.g_losses = g_losses
        self.generated = generated

    def run_discriminator_one_step(self, data):
        self.optimizer_D.zero_grad()
        d_losses = self.pix2pix_model(data, mode='discriminator')
        d_loss = sum(d_losses.values()).mean()
        d_loss.backward()
        self.optimizer_D.step()
        self.d_losses = d_losses

    def run_generator_swapped(self, data, p):
        #generated = self.pix2pix_model(data, mode='swap_part')
        if self.opt.use_ema:
            print("Data generated with EMA!!!", flush = True)
            generated = self.ema(data, mode='swap_part', p=p)
        else:
            generated = self.pix2pix_model_on_one_gpu(data, mode='swap_part', p=p)
        return generated
    
    def generate_with_noise(self, data, p):
        #generated = self.pix2pix_model(data, mode='swap_part')
        if self.opt.use_ema:
            generated = self.ema(data, mode='generate_with_noise', p=p)
        else:
            generated = self.pix2pix_model_on_one_gpu(data, mode='generate_with_noise', p=p)
        return generated

    def generate_with_mask(self, data, p):
        #generated = self.pix2pix_model(data, mode='swap_part')
        if self.opt_use_ema:
            generated = self.ema(data, mode='generate_with_mask', p=p)
        else:
            generated = self.pix2pix_model_on_one_gpu(data, mode='generate_with_mask', p=p)
        return generated

    def get_latest_losses(self):
        return {**self.g_losses, **self.d_losses}

    def get_latest_generated(self):
        return self.generated
    

    def update_learning_rate(self, epoch):
        self.update_learning_rate(epoch)

    def save(self, epoch):
        if self.opt.use_ema:
            print("Saving EMA!!!", flush = True)
            torch.save(self.ema.state_dict(), os.path.join(self.opt.checkpoints_dir, 'ema.pth'))
        self.pix2pix_model_on_one_gpu.save(epoch)

    ##################################################################
    # Helper functions
    ##################################################################

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            if self.opt.no_TTUR:
                new_lr_G = new_lr
                new_lr_D = new_lr
            else:
                new_lr_G = new_lr / 2
                new_lr_D = new_lr * 2

            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = new_lr_D
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = new_lr_G
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr