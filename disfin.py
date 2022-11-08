#-*- coding: utf-8 -*-
from model.xception import Xception
from cunet import Conditional_UNet as CUNet
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

class MLP(nn.Module):
    def __init__(self, in_f, hidden_dim, out_f):
        super(MLP, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(nn.Linear(in_f, hidden_dim),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(hidden_dim, hidden_dim),
                                nn.LeakyReLU(inplace=True),
                                nn.Linear(hidden_dim, out_f),)

    def forward(self, x):
        x = self.pool(x)
        x = self.mlp(x)
        return x

class Head(torch.nn.Module):
  def __init__(self, in_f, hidden_dim, out_f):
    super(Head, self).__init__()
    self.do = nn.Dropout(0.2)
    self.pool = nn.AdaptiveAvgPool2d(1)
    self.mlp = nn.Sequential(nn.Linear(in_f, hidden_dim),
                            nn.LeakyReLU(inplace=True),
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.LeakyReLU(inplace=True),
                            nn.Linear(hidden_dim, in_f//2),
                            nn.LeakyReLU(inplace=True),
                            nn.Linear(in_f//2, out_f),)

  def forward(self, x):
    bs = x.size()[0]
    x = self.pool(x).view(bs, -1)
    x = self.mlp(x)
    x = self.do(x)
    return x

class disfin(nn.Module):
    def __init__(self, num_classes, img_size, encoder_feat_dim) -> None:
        super(disfin, self).__init__()
        # init variable
        self.num_classes = num_classes
        self.img_size = img_size
        self.encoder_feat_dim = encoder_feat_dim
        self.half_fingerprint_dim = encoder_feat_dim//2
        
        # basic function
        self.lr = nn.LeakyReLU(inplace=True)
        self.do = nn.Dropout(0.2)

        # models
        self.encoder = self.init_xcep()
        self.con_gan = CUNet(2)
        # self.block_fake = MLP(in_f=1024, hidden_dim=512, out_f=512)
        # self.block_real = MLP(in_f=1024, hidden_dim=512, out_f=512)

        self.head_spe = Head(in_f=256, hidden_dim=512, out_f=5)
        self.head_sha = Head(in_f=256, hidden_dim=512, out_f=1)

        # self.block_fake_fin = Head(in_f=1024, hidden_dim=512, out_f=512)
        # self.block_fake_con = Head(in_f=1024, hidden_dim=512, out_f=512)
        
        # self.block_real_fin = Head(in_f=1024, hidden_dim=512, out_f=512)
        # self.block_real_con = Head(in_f=1024, hidden_dim=512, out_f=512)

        # self.block_fake_spe = Head(in_f=512, hidden_dim=256, out_f=256)
        # self.block_fake_sha = Head(in_f=512, hidden_dim=256, out_f=256)
        # self.block_real_spe = Head(in_f=512, hidden_dim=256, out_f=256)
        # self.block_real_sha = Head(in_f=512, hidden_dim=256, out_f=256)

    def init_xcep(self, pretrained_path='pretrained/xception-b5690688.pth'):
        xcep = Xception(self.num_classes)
        # load pre-trained Xception
        state_dict = torch.load(pretrained_path)
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        state_dict = {k:v for k, v in state_dict.items() if 'fc' not in k}
        xcep.load_state_dict(state_dict, False)
        return xcep
    
    def forward(self, cat_data, train=True):
        bs = cat_data.shape[0]
        hidden = self.encoder.features(cat_data)  # -> bs,1024
        # out = self.head(hidden)
        hidden_real = hidden[:bs//2, :, :, :]
        hidden_fake = hidden[bs//2:, :, :, :]
        # out_spe = self.head_spe(hidden)
        # out_sha = self.head_sha(hidden)
        # out = (out_spe, out_sha)

        c1, f1 = hidden_fake[:, :self.encoder_feat_dim, :, :], hidden_fake[:, self.encoder_feat_dim:, :, :]
        f1_spe, f1_share = f1[:, :self.half_fingerprint_dim, :, :], f1[:, self.half_fingerprint_dim:, :, :]

        c2, f2 = hidden_real[:, :self.encoder_feat_dim, :, :], hidden_real[:, self.encoder_feat_dim:, :, :]
        f2_spe, f2_share = f2[:, :self.half_fingerprint_dim, :, :], f2[:, self.half_fingerprint_dim:, :, :]

        # f1 = self.block_fake_fin(hidden_fake).view(bs//2, -1, 1, 1)
        # c1 = self.block_fake_con(hidden_fake).view(bs//2, -1, 1, 1)
        # f1_spe = self.block_fake_spe(f1).view(bs//2, -1, 1, 1)
        # f1_share = self.block_fake_sha(f1).view(bs//2, -1, 1, 1)

        # f2 = self.block_real_fin(hidden_real).view(bs//2, -1, 1, 1)
        # c2 = self.block_real_con(hidden_real).view(bs//2, -1, 1, 1)
        # f2_spe = self.block_real_spe(f2).view(bs//2, -1, 1, 1)
        # f2_share = self.block_real_sha(f2).view(bs//2, -1, 1, 1)

        # head for spe and sha
        out_spe = self.head_spe(torch.cat((f2_spe, f1_spe), dim=0))
        out_sha = self.head_sha(torch.cat((f2_share, f1_share), dim=0))
        out = (out_spe, out_sha)

        if train:
            # pair combination
            pair_index = random.randint(0, 1)
            # f1 + c2 -> f12, f3 + c1 -> near~I1, c3 + f2 -> near~I2
            if pair_index == 0:
                # reconstruction mse loss
                forgery_image_12 = self.con_gan(f1, c2)
                hidden_fake_plus = self.encoder.features(forgery_image_12)
                c3, f3 = hidden_fake_plus[:, :self.encoder_feat_dim, :, :], hidden_fake[:, self.encoder_feat_dim:, :, :]
                # f3_spe, f3_share = f3[:, :self.half_fingerprint_dim, :, :], f2[:, self.half_fingerprint_dim:, :, :]

                reconstruction_image_1 = self.con_gan(f3, c1)
                reconstruction_image_2 = self.con_gan(f2, c3)
            
            # f2 + c1 -> f21, f3 + c2 -> near~I2, c3 + f1 -> near~I1
            else:
                # reconstruction mse loss
                forgery_image_12 = self.con_gan(f2, c1)
                hidden_fake_plus = self.encoder.features(forgery_image_12)
                c3, f3 = hidden_fake_plus[:, :self.encoder_feat_dim, :, :], hidden_fake[:, self.encoder_feat_dim:, :, :]
                # f3_spe, f3_share = f3[:, :self.half_fingerprint_dim, :, :], f2[:, self.half_fingerprint_dim:, :, :]

                reconstruction_image_2 = self.con_gan(f3, c2)
                reconstruction_image_1 = self.con_gan(f1, c3)

            return None, (out, reconstruction_image_1, reconstruction_image_2, forgery_image_12)
        
        # inference only consider share loss
        else:
            return None, out

        # return f1_spe, f1_share, f2_spe, f2_share, forgery_image_12, reconstruction_image_1, reconstruction_image_2