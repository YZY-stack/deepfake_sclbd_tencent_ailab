#-*- coding: utf-8 -*-
from model.xception import Xception
# from model.resnest import Resnest
# from model.efficient import Efficient
from cunet import Conditional_UNet as CUNet
from model.disc import Disc
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
        x_feat = self.pool(x).view(bs, -1)
        x = self.mlp(x_feat)
        x = self.do(x)
        return x, x_feat

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
        # self.encoder = self.init_xcep()  # xception
        self.encoder_f = self.init_xcep()  # xception
        self.encoder_c = self.init_xcep()  # xception
        self.discriminator = Disc()        # multiscale-discriminator
        # self.encoder = self.init_resnest()  # resnest
        # self.encoder = self.init_efficient()  # efficient
        # self.encoder = self.init_convnext()  # convnext

        # conditional gan
        self.con_gan = CUNet(2)
        # self.block_fake = MLP(in_f=1024, hidden_dim=512, out_f=512)
        # self.block_real = MLP(in_f=1024, hidden_dim=512, out_f=512)
        self.adjust_conv = Conv2d1x1(in_f=1024, hidden_dim=512, out_f=512)

        # head
        self.head_spe = Head(in_f=256, hidden_dim=512, out_f=5)
        self.head_sha = Head(in_f=256, hidden_dim=512, out_f=num_classes)

        # block
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

    def forward(self, cat_data, train=True):
        real, fake = cat_data.chunk(2, dim=0)
        bs = cat_data.shape[0]

        # encoder
        f_all = self.adjust_conv(self.encoder_f.features(cat_data))  # -> bs,1024
        c_all = self.adjust_conv(self.encoder_c.features(cat_data))  # -> bs,1024

        # classification, multi-task
        f_spe = self.block_spe(f_all)
        f_share = self.block_sha(f_all)
        
        # reconstruction loss
        f2, f1 = f_all.chunk(2, dim=0)
        c2, c1 = c_all.chunk(2, dim=0)
        if train:
            # ==== self reconstruction ==== #
            # f1 + c1 -> f11, f11 + c1 -> near~I1
            self_reconstruction_image_1 = self.con_gan(f1, c1)

            # f2 + c2 -> f2, f2 + c2 -> near~I2
            self_reconstruction_image_2 = self.con_gan(f2, c2)

            # ==== cross combine ==== #
            # pair combination 1
            # f1 + c2 -> f12, f12 + c1 -> near~I1, c3 + f2 -> near~I2
            # reconstruction mse loss
            # forgery_image_12 = self.con_gan(f1, c2)
            # f12 = self.adjust_conv(self.encoder_f.features(forgery_image_12))
            # c12 = self.adjust_conv(self.encoder_c.features(forgery_image_12))
            # f12_spe = self.block_spe(f12)
            # f12_share = self.block_sha(f12)
            # reconstruction_image_1 = self.con_gan(f12, c1)
            # reconstruction_image_2 = self.con_gan(f2, c12)

            reconstruction_image_1 = self.con_gan(f1, c2)
            reconstruction_image_2 = self.con_gan(f2, c1)

            # pair combination 2
            # f2 + c1 -> f21, f21 + c2 -> near~I2, c21 + f1 -> near~I1
            # reconstruction mse loss
            # forgery_image_21 = self.con_gan(f2, c1)
            # f21 = self.adjust_conv(self.encoder_f.features(forgery_image_21))
            # c21 = self.adjust_conv(self.encoder_c.features(forgery_image_21))
            # f21_spe = self.block_spe(f21)
            # f21_share = self.block_sha(f21)
            # reconstruction_image_11 = self.con_gan(f21, c2)
            # reconstruction_image_22 = self.con_gan(f1, c21)

            # feature loss
            # f1_recon = self.adjust_conv(self.encoder_f.features(reconstruction_image_1))
            # c1_recon = self.adjust_conv(self.encoder_c.features(reconstruction_image_1))
            # f2_recon = self.adjust_conv(self.encoder_f.features(reconstruction_image_2))
            # c2_recon = self.adjust_conv(self.encoder_c.features(reconstruction_image_2))

            # *** not calculate this gan loss here, but in trainer instead *** #
            # # gan loss
            # adv = self.discriminator(
            #     torch.cat((
            #         real, reconstruction_image_2, 
            #         fake, reconstruction_image_1,
            #         reconstruction_image_11,
            #         reconstruction_image_22,
            #         forgery_image_12, forgery_image_21
            #         ), dim=0)
            # )

            # head for spe and sha
            # out_spe, _ = self.head_spe(torch.cat((f_spe, f21_spe, f12_spe), dim=0))
            # out_sha, _ = self.head_sha(torch.cat((f_share, f21_share, f12_share), dim=0))
            out_spe, _ = self.head_spe(f_spe)
            out_sha, _ = self.head_sha(f_share)
            out = (out_spe, out_sha)

            # f3_spe = self.pool(f3_spe)
            # f1_spe = self.pool(f1_spe)
            f1 = self.pool(f1)
            c1 = self.pool(c1)
            # f3 = self.pool(f3)
            c2 = self.pool(c2)
            f2 = self.pool(f2)
            # c3 = self.pool(c3)

            return None, (out, self_reconstruction_image_1, self_reconstruction_image_2, \
                            reconstruction_image_1, reconstruction_image_2, \
                            # reconstruction_image_11, reconstruction_image_22, \
                            # forgery_image_12, forgery_image_21, \
                            # f1, f2, c1, c2, \
                            # f1_recon, c1_recon, f2_recon, c2_recon, \
                            None)
                            # adv_2, adv_1)
        
        # inference only consider share loss
        else:
            # head for spe and sha
            out_spe, spe_feat = self.head_spe(f_spe)
            out_sha, sha_feat = self.head_sha(f_share)
            out = (out_spe, out_sha)
            content = self.pool(torch.cat((c2, c1), dim=0)).view(bs, -1)
            feat = (spe_feat, sha_feat, content)
            return None, out, feat
