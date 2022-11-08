import torch
import torch.nn as nn
from torch.nn import parameter
from model.disfin import disfin
import torch.nn.functional as F
import numpy as np
import os


def initModel(mod, gpu_ids):
    mod = mod.to(f'cuda:{gpu_ids[0]}')
    mod = nn.DataParallel(mod, gpu_ids)
    return mod

class Trainer(): 
    def __init__(self, gpu_ids, mode, pretrained_path):
        self.device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu')
        # self.model = F3Net(mode=mode, device=self.device)
        self.model = disfin(num_classes=1, img_size=299, encoder_feat_dim=512)
        self.model = initModel(self.model, gpu_ids)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.loss_fn_ce = nn.CrossEntropyLoss()
        self.loss_fn_mse = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                              lr=0.0002, betas=(0.9, 0.999))
        # self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
        #                                         lr=0.002, momentum=0.9, weight_decay=0)

    def set_input(self, input, label):
        self.input = input.to(self.device)
        self.label = label.to(self.device)

    def forward(self, x, train=True):
        fea, out = self.model(x, train)
        del fea
        return out
    
    def optimize_weight(self, stu_cla):
        self.loss_cla = self.loss_fn(stu_cla.squeeze(1), self.label.float()) # classify loss
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

    def optimize_weight_mse(self, real, fake_instance, reconstruction_image_1, reconstruction_image_2, forgery_image_12):
        loss_reconstruction_1 = self.loss_fn_mse(fake_instance, reconstruction_image_1)
        loss_reconstruction_2 = self.loss_fn_mse(real, reconstruction_image_2)
        loss_reconstruction_3 = self.loss_fn_mse(real, forgery_image_12)
        self.loss_reconstruction = loss_reconstruction_1 + loss_reconstruction_2 + loss_reconstruction_3
        return self.loss_reconstruction
    
    def get_final_loss(self, loss_sha, loss_spe, loss_reconstruction):
        self.loss = loss_sha + loss_spe + loss_reconstruction
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        return self.loss

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)
