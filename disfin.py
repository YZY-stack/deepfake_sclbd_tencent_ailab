#-*- coding: utf-8 -*-
from model.xception import Xception
from model.resnest import Resnest
# from model.efficient import Efficient
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

class Conv2d1x1(nn.Module):
    def __init__(self, in_f, hidden_dim, out_f):
        super(Conv2d1x1, self).__init__()
        self.conv2d = nn.Sequential(nn.Conv2d(in_f, hidden_dim, 1, 1),
                                nn.LeakyReLU(inplace=True),
                                nn.Conv2d(hidden_dim, hidden_dim, 1, 1),
                                nn.LeakyReLU(inplace=True),
                                nn.Conv2d(hidden_dim, out_f, 1, 1),)

    def forward(self, x):
        x = self.conv2d(x)
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
    def __init__(self, num_classes, encoder_feat_dim) -> None:
        super(disfin, self).__init__()
        # init variable
        self.num_classes = num_classes
        self.encoder_feat_dim = encoder_feat_dim
        self.half_fingerprint_dim = encoder_feat_dim//2
        
        # basic function
        self.lr = nn.LeakyReLU(inplace=True)
        self.do = nn.Dropout(0.2)
        self.pool = nn.AdaptiveAvgPool2d(1)

        # models
        self.encoder = self.init_xcep()  # xception
        # self.encoder = self.init_resnest()  # resnest
        # self.encoder = self.init_efficient()  # efficient
        # self.encoder = self.init_convnext()  # convnext


        self.con_gan = CUNet(2)
        # self.block_fake = MLP(in_f=1024, hidden_dim=512, out_f=512)
        # self.block_real = MLP(in_f=1024, hidden_dim=512, out_f=512)

        self.head_spe = Head(in_f=256, hidden_dim=512, out_f=5)
        self.head_sha = Head(in_f=256, hidden_dim=512, out_f=1)

        self.block_fin = Conv2d1x1(in_f=1024, hidden_dim=512, out_f=512)
        self.block_con = Conv2d1x1(in_f=1024, hidden_dim=512, out_f=512)
        
        self.block_spe = Conv2d1x1(in_f=512, hidden_dim=256, out_f=256)
        self.block_sha = Conv2d1x1(in_f=512, hidden_dim=256, out_f=256)

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

    def init_resnest(self):
        from resnest.torch import resnest50
        resnest_layer = resnest50(pretrained=True)
        net = Resnest(resnest_layer, num_classes=self.num_classes)
        return net

    def init_efficient(self):
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_pretrained('efficientnet-b7')
        net = Efficient(model, num_classes=self.num_classes)
        return net

    def init_convnext(self):
        from timm.models import create_model
        net = create_model(
        'convnext_base', 
        pretrained=True, 
        num_classes=2, 
        drop_path_rate=0.0,
        # layer_scale_init_value=1e-6,
        head_init_scale=1.0,
        )
        return net

    def forward(self, cat_data, train=True, pair_index=0):
        bs = cat_data.shape[0]
        hidden = self.encoder.features(cat_data)  # -> bs,1024
        # hidden = self.encoder.forward_features(cat_data)  # -> bs,1024
        # out = self.head(hidden)
        hidden_real = hidden[:bs//2, :, :, :]
        hidden_fake = hidden[bs//2:, :, :, :]
        # out_spe = self.head_spe(hidden)
        # out_sha = self.head_sha(hidden)
        # out = (out_spe, out_sha)

        # c1, f1 = hidden_fake[:, :self.encoder_feat_dim, :, :], hidden_fake[:, self.encoder_feat_dim:, :, :]
        # f1_spe, f1_share = f1[:, :self.half_fingerprint_dim, :, :], f1[:, self.half_fingerprint_dim:, :, :]

        # c2, f2 = hidden_real[:, :self.encoder_feat_dim, :, :], hidden_real[:, self.encoder_feat_dim:, :, :]
        # f2_spe, f2_share = f2[:, :self.half_fingerprint_dim, :, :], f2[:, self.half_fingerprint_dim:, :, :]

        f1 = self.block_fin(hidden_fake)
        c1 = self.block_con(hidden_fake)
        f1_spe = self.block_spe(f1)
        f1_share = self.block_sha(f1)

        f2 = self.block_fin(hidden_real)
        c2 = self.block_con(hidden_real)
        f2_spe = self.block_spe(f2)
        f2_share = self.block_sha(f2)

        if train:
            # pair combination
            # f1 + c2 -> f12, f3 + c1 -> near~I1, c3 + f2 -> near~I2
            if pair_index == 0:
                # reconstruction mse loss
                forgery_image_12 = self.con_gan(f1, c2)
                hidden_fake_plus = self.encoder.features(forgery_image_12)
                # hidden_fake_plus = self.encoder.forward_features(forgery_image_12)
                c3, f3 = hidden_fake_plus[:, :self.encoder_feat_dim, :, :], hidden_fake_plus[:, self.encoder_feat_dim:, :, :]
                f3_spe, f3_share = f3[:, :self.half_fingerprint_dim, :, :], f3[:, self.half_fingerprint_dim:, :, :]

                reconstruction_image_1 = self.con_gan(f3, c1)
                reconstruction_image_2 = self.con_gan(f2, c3)
            
            # f2 + c1 -> f21, f3 + c2 -> near~I2, c3 + f1 -> near~I1
            else:
                # reconstruction mse loss
                forgery_image_12 = self.con_gan(f2, c1)
                hidden_fake_plus = self.encoder.features(forgery_image_12)
                # hidden_fake_plus = self.encoder.forward_features(forgery_image_12)
                c3, f3 = hidden_fake_plus[:, :self.encoder_feat_dim, :, :], hidden_fake[:, self.encoder_feat_dim:, :, :]
                f3_spe, f3_share = f3[:, :self.half_fingerprint_dim, :, :], f2[:, self.half_fingerprint_dim:, :, :]

                reconstruction_image_2 = self.con_gan(f3, c2)
                reconstruction_image_1 = self.con_gan(f1, c3)

            # head for spe and sha
            # out_spe = self.head_spe(torch.cat((f2_spe, f1_spe, f3_spe), dim=0))
            # out_sha = self.head_sha(torch.cat((f2_share, f1_share, f3_share), dim=0))
            out_spe = self.head_spe(torch.cat((f2_spe, f1_spe), dim=0))
            out_sha = self.head_sha(torch.cat((f2_share, f1_share), dim=0))
            # out_sha = self.head_sha(torch.cat((f2_share, f1_share, f3_share), dim=0))
            out = (out_spe, out_sha)

            f3_spe = self.pool(f3_spe)
            f1_spe = self.pool(f1_spe)

            return None, (out, reconstruction_image_1, reconstruction_image_2, forgery_image_12, f1_spe, f3_spe)
        
        # inference only consider share loss
        else:
            # head for spe and sha
            out_spe = self.head_spe(torch.cat((f2_spe, f1_spe), dim=0))
            out_sha = self.head_sha(torch.cat((f2_share, f1_share), dim=0))
            out = (out_spe, out_sha)
            return None, out

        # return f1_spe, f1_share, f2_spe, f2_share, forgery_image_12, reconstruction_image_1, reconstruction_image_2
