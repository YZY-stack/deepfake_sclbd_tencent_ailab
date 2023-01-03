import torch
import torch.nn as nn
from torch.nn import parameter
from model.disfin import disfin
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, MultiStepLR, StepLR
from torchtoolbox.nn.init import KaimingInitializer
from torchtoolbox.optimizer import CosineWarmupLr
from timm.loss import LabelSmoothingCrossEntropy
from model.disc import Disc


def initModel(mod, gpu_ids):
    mod = mod.to(f'cuda:{gpu_ids[0]}')
    mod = nn.DataParallel(mod, gpu_ids)
    return mod

class Trainer(): 
    def __init__(self, gpu_ids):
        # device
        self.device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu')

        # model
        self.model = disfin(num_classes=1, encoder_feat_dim=512)
        self.model = initModel(self.model, gpu_ids)

        # discriminator
        self.discriminator = Disc()
        initializer = KaimingInitializer()
        self.discriminator.apply(initializer)
        self.discriminator.to(self.device)

        # loss
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.loss_fn_ce = nn.CrossEntropyLoss()
        # self.adv_loss = nn.CrossEntropyLoss()
        self.adv_loss = nn.MSELoss()  # lsgan
        # self.loss_fn_ce = LabelSmoothingCrossEntropy(0.1)
        self.loss_fn_mse = torch.nn.HuberLoss()

        # opt
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=0.0002, weight_decay=0, betas=(0.9, 0.999)
        )
        self.optimizer_d = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.discriminator.parameters()),
            lr=0.0002, weight_decay=0, betas=(0.9, 0.999)
        )
        # self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
        #                                         lr=0.002, momentum=0.9, weight_decay=0)
        # self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=3, T_mult=2, eta_min=1e-6, last_epoch=-1)
        
        self.scheduler = CosineWarmupLr(
                            self.optimizer, 32, 15,
                            base_lr=0.0002, warmup_epochs=3
                        )

    def set_input(self, input, label):
        self.input = input.to(self.device)
        self.label = label.to(self.device)

    def forward(self, x, train=True):
        if train:
            fea, out = self.model(x, train=train)
            del fea
            return out
        else:
            fea, out, feature = self.model(x, train=train)
            del fea
            return out, feature
    
    def optimize_weight(self, stu_cla):
        weight = torch.zeros_like(self.label).float().to(self.device)
        weight = torch.fill_(weight, 0.4)
        weight[self.label>0]=0.6
        loss_func = nn.BCEWithLogitsLoss(weight=weight)
        self.loss_cla = loss_func(stu_cla.squeeze(1), self.label.float()) # classify loss
        # self.loss_cla = self.loss_fn(stu_cla.squeeze(1), self.label.long()) # classify loss
        # self.loss = self.loss_cla

        # self.optimizer.zero_grad()
        # self.loss.backward()
        # self.optimizer.step()
        return self.loss_cla

    def optimize_weight_ce(self, stu_cla):
        self.loss_cla_spe = self.loss_fn_ce(stu_cla, self.label) # classify loss
        # self.loss = self.loss_cla

        # self.optimizer.zero_grad()
        # self.loss.backward()
        # self.optimizer.step()
        return self.loss_cla_spe

    def optimize_weight_mse(
        self, 
        real, 
        fake_instance,
        self_reconstruction_image_1,
        self_reconstruction_image_2,
        reconstruction_image_1,
        reconstruction_image_2,
        # reconstruction_image_11,
        # reconstruction_image_22,
        # forgery_image_12,
        # forgery_image_21,
        # f1, f2, c1, c2,
        # f12, c12, f21, c21,
        # f1_recon, c1_recon, f2_recon, c2_recon
    ):
        # image reconstruction loss
        self_loss_reconstruction_1 = self.loss_fn_mse(fake_instance, self_reconstruction_image_1)
        self_loss_reconstruction_2 = self.loss_fn_mse(real, self_reconstruction_image_2)

        loss_reconstruction_1 = self.loss_fn_mse(fake_instance, reconstruction_image_1)
        loss_reconstruction_2 = self.loss_fn_mse(real, reconstruction_image_2)

        # loss_reconstruction_3 = self.loss_fn_mse(fake_instance, reconstruction_image_22)
        # loss_reconstruction_4 = self.loss_fn_mse(real, reconstruction_image_11)

        # loss_reconstruction_5 = self.loss_fn_mse(real, forgery_image_12)
        # loss_reconstruction_6 = self.loss_fn_mse(fake_instance, forgery_image_21)

        # # feature reconstruction loss
        # # cross
        # loss_f1_f12 = self.loss_fn_mse(f1, f12)
        # loss_f2_f21 = self.loss_fn_mse(f2, f21)
        # loss_c2_c12 = self.loss_fn_mse(c2, c12)
        # loss_c1_c21 = self.loss_fn_mse(c1, c21)
        # # within
        # loss_f1_f1prime = self.loss_fn_mse(f1, f1_recon)
        # loss_c1_c1prime = self.loss_fn_mse(c1, c1_recon)
        # loss_f2_f2prime = self.loss_fn_mse(f2, f2_recon)
        # loss_c2_c2prime = self.loss_fn_mse(c2, c2_recon)

        # feature_loss = \
        #     loss_f1_f1prime + loss_c1_c1prime + \
        #     loss_f2_f2prime + loss_c2_c2prime + \
        #     loss_f1_f12 + loss_f2_f21 + loss_c2_c12 + loss_c1_c21

        self.loss_reconstruction =  \
            self_loss_reconstruction_1 + \
            self_loss_reconstruction_2 + \
            loss_reconstruction_1 + \
            loss_reconstruction_2
        return self_loss_reconstruction_1, self_loss_reconstruction_2, \
               loss_reconstruction_1, loss_reconstruction_2

    def optimize_weight_discriminator(
        self,
        adv_label,
        reconstruction_image_1,
        reconstruction_image_2,
        # reconstruction_image_11,
        # reconstruction_image_22,
        # forgery_image_12,
        # forgery_image_21,
    ):
        data = torch.cat((
            self.input, 
            reconstruction_image_2, 
            reconstruction_image_1,
            # reconstruction_image_11,
            # reconstruction_image_22,
            # forgery_image_12,
            # forgery_image_21
        ), dim=0)
        adv = self.discriminator(data)
        self.loss_discriminator = self.adv_loss(adv, adv_label.float())  # real,fake,real,fake
        # self.loss_discriminator = self.adv_loss(adv, adv_label.long())  # real,fake,real,fake

        self.optimizer_d.zero_grad()
        self.loss_discriminator.backward()
        self.optimizer_d.step()

        return self.loss_discriminator
    
    @staticmethod
    def vgg_preprocess(batch):
        tensortype = type(batch.data)
        (r, g, b) = torch.chunk(batch, 3, dim = 1)
        batch = torch.cat((b, g, r), dim = 1) # convert RGB to BGR
        batch = (batch + 1) * 255 * 0.5 # [-1, 1] -> [0, 255]
        mean = tensortype(batch.data.size()).cuda()
        mean[:, 0, :, :] = 103.939
        mean[:, 1, :, :] = 116.779
        mean[:, 2, :, :] = 123.680
        batch = batch.sub(Variable(mean)) # subtract mean
        return batch
    
    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = self.vgg_preprocess(img)
        target_vgg = self.vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def get_final_loss(
        self,
        loss_sha, 
        loss_spe, 
        loss_reconstruction,
        # feature_loss,
        loss_discriminator,
        lambda_sha=1,
        lambda_spe=1,
        lambda_mse=0.3,
        # lambda_feat=0.5,
        lambda_adv=0.01,
    ):
        self.loss = \
            lambda_sha  * loss_sha + \
            lambda_spe  * loss_spe + \
            lambda_mse  * loss_reconstruction + \
            lambda_adv  * loss_discriminator
            # lambda_feat * feature_loss 
        
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return self.loss

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)
