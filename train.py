import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
osenvs = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
import sys
import time
import torch
import torch.nn
import torch.backends.cudnn as cudnn
from torchvision import utils as vutils

from utils import evaluate, get_dataset, FFDataset, setup_logger
from trainer import Trainer
# from model.vgg import load_vgg16
import numpy as np
from copy import deepcopy
import random

# config
dataset_path = '/mntnfs/sec_data2/yanzhiyuan/HQ/'
celeb_path = '/mntnfs/sec_data2/yanzhiyuan/deepfakes_dataset/CelebDF1_crop'
pretrained_path = 'pretrained/xception-b5690688.pth'
batch_size = 32
gpu_ids = [*range(osenvs)]
max_epoch = 15
loss_freq = 40
visualization = True
ckpt_dir = './weights'
ckpt_name = 'disfin_paircombination_midloss_warmup_selfcon_direct_adv_nomoreadv_nodoublecycle_lsgan_1'
img_save_path = 'img_save'
os.makedirs(img_save_path, exist_ok=True)

# fix the seed for reproducibility
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
cudnn.benchmark = True


if __name__ == '__main__':
    dataset = FFDataset(
        dataset_root=os.path.join(dataset_path, 'train', 'real'), 
        size=256, 
        frame_num=100, 
        augment=True
    )

    dataloader_real = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size // 2,
        shuffle=True,
        num_workers=8,
        collate_fn=FFDataset.collate,
    )
    
    len_dataloader = dataloader_real.__len__()

    # init checkpoint and logger
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    logger = setup_logger(ckpt_path, 'result.log', 'logger')
    best_val = 0.
    ckpt_model_name = 'best.pkl'
    
    # train
    model = Trainer(gpu_ids)
    model.total_steps = 0
    epoch = 0
    best_auc = 0.
    
    while epoch < max_epoch:

        dataset_img, total_len = get_dataset(
        name='train', 
        size=256, 
        root=dataset_path, 
        frame_num=25, 
        augment=True
        )

        dataloader_fake = torch.utils.data.DataLoader(
            dataset=dataset_img,
            batch_size=batch_size // 2,
            shuffle=True,
            num_workers=8,
            collate_fn=FFDataset.collate,
        )

        fake_iter = iter(dataloader_fake)
        real_iter = iter(dataloader_real)
        
        logger.debug(f'No {epoch}')
        i = 0

        while i < len_dataloader:
            
            i += 1
            model.total_steps += 1

            try:
                data_real, label_real = real_iter.next()
                data_fake, label_fake_instance = fake_iter.next()
                label_fake = torch.ones_like(label_real)
            except StopIteration:
                break
            # -------------------------------------------------
            
            if data_real.shape[0] != data_fake.shape[0]:
                continue

            # *** share label task *** #
            data = torch.cat([data_real,data_fake],dim=0)
            label = torch.cat([label_real,label_fake],dim=0)
            # label = torch.cat([label_real,label_fake,label_real,label_fake],dim=0)
            # adv_label = deepcopy(label)
            # adv_label = torch.cat(
            #     [label_real,label_real,label_fake,label_fake],
            #     dim=0)
            adv_label = torch.cat(
                [label_real,label_real,label_fake,label_fake,
                #  label_fake,label_fake,label_fake,label_fake,
                ],
                dim=0)

            # # manually shuffle
            # idx = list(range(data.shape[0]))
            # random.shuffle(idx)
            # data = data[idx]
            # label = label[idx]

            data = data.detach()
            label = label.detach()

            model.set_input(data,label)

            stu_fea, stu_cla = model.model(model.input)
            (spe_out, sha_out), \
            self_reconstruction_image_1, \
            self_reconstruction_image_2, \
            reconstruction_image_1, \
            reconstruction_image_2, \
            f1, f2, c1, c2, \
            _ = stu_cla

            # reconstruction_image_11, \
            # reconstruction_image_22, \
            # forgery_image_12, \
            # forgery_image_21, \

            # f1_recon, c1_recon, f2_recon, c2_recon, \

            # *** share label task *** #
            loss_share = model.optimize_weight(sha_out)

            # *** instance label task *** #
            spe_label = torch.cat([label_real,label_fake_instance],dim=0)
            # spe_label = torch.cat([label_real,label_fake_instance,label_real,label_fake_instance],dim=0)
            spe_label = spe_label.detach()
            model.label = spe_label.to(model.device)
            loss_specific = model.optimize_weight_ce(spe_out)

            # *** mse construction task *** #
            self_loss_reconstruction_1, \
            self_loss_reconstruction_2, \
            loss_reconstruction_1, \
            loss_reconstruction_2 \
                = model.optimize_weight_mse(
                data_real.to(model.device),
                data_fake.to(model.device),
                self_reconstruction_image_1,
                self_reconstruction_image_2,
                reconstruction_image_1,
                reconstruction_image_2,
                # reconstruction_image_11,
                # reconstruction_image_22,
                # forgery_image_12,
                # forgery_image_21,
                # f1, f2, c1, c2,
                # f1_recon, c1_recon, f2_recon, c2_recon
            )

            loss_reconstruction = self_loss_reconstruction_1 + \
                                  self_loss_reconstruction_2 + \
                                  loss_reconstruction_1 + \
                                  loss_reconstruction_2

            # *** discriminator loss *** #
            loss_discriminator \
                = model.optimize_weight_discriminator(
                adv_label.to(model.device),
                self_reconstruction_image_1.detach(),
                self_reconstruction_image_2.detach(),
                # reconstruction_image_11.detach(),
                # reconstruction_image_22.detach(),
                # forgery_image_12.detach(),
                # forgery_image_21.detach(),
            )

            # # *** vgg loss *** #
            # load_vgg16('./vgg16')

            # total loss
            loss = model.get_final_loss(
                loss_share, 
                loss_specific,
                loss_reconstruction,
                # feature_loss,
                -loss_discriminator.detach().to(model.device),  # for the view of generator
            )

            if model.total_steps % loss_freq == 0:
                logger.debug(
                    f'loss: {loss:.5f}' 
                    f' spe_loss: {loss_specific:.5f}' 
                    f' sha_loss: {loss_share:.5f}' 
                    f' mse_loss: {loss_reconstruction:.5f}' 
                    f' adv_loss: {loss_discriminator:.5f}' 
                    f' loss_fake_reconstruction1: {self_loss_reconstruction_1:.5f}' 
                    f' loss_real_reconstruction2: {self_loss_reconstruction_2:.5f}' 
                    # f' loss_forgery_real: {loss_reconstruction_3:.5f}' 
                    # f' loss_feature: {feature_loss:.5f}' 
                    f' at step: {model.total_steps}'
                )

                # save img
                if visualization:
                    vutils.save_image(
                        self_reconstruction_image_1.detach().cpu(), 
                        f'{img_save_path}/epoch{epoch}_step{model.total_steps}_self_reconstruction_image_1.png')

                    vutils.save_image(
                        self_reconstruction_image_2.detach().cpu(), 
                        f'{img_save_path}/epoch{epoch}_step{model.total_steps}_self_reconstruction_image_2.png')
                    
                    vutils.save_image(
                        reconstruction_image_1.detach().cpu(), 
                        f'{img_save_path}/epoch{epoch}_step{model.total_steps}_reconstruction_image_11.png')

                    vutils.save_image(
                        reconstruction_image_2.detach().cpu(), 
                        f'{img_save_path}/epoch{epoch}_step{model.total_steps}_reconstruction_image_22.png')

                    # vutils.save_image(
                    #     forgery_image_12.detach().cpu(), 
                    #     f'{img_save_path}/epoch{epoch}_step{model.total_steps}_forgery_image_12.png')
                    
                    # vutils.save_image(
                    #     forgery_image_21.detach().cpu(), 
                    #     f'{img_save_path}/epoch{epoch}_step{model.total_steps}_forgery_image_21.png')

                    vutils.save_image(
                        data_real.detach().cpu(), 
                        f'{img_save_path}/epoch{epoch}_step{model.total_steps}_data_real.png')

                    vutils.save_image(
                        data_fake.detach().cpu(), 
                        f'{img_save_path}/epoch{epoch}_step{model.total_steps}_data_fake.png')

            if i % int(len_dataloader / 10) == 0:
                model.model.eval()
                # auc, r_acc, f_acc = evaluate(model, dataset_path, mode='val', test_data_name='FF++')
                # logger.debug(f'(Val @ epoch {epoch}) auc: {auc:.5f}, r_acc: {r_acc:.5f}, f_acc: {f_acc:.5f}')
                auc, r_acc, f_acc, spe_acc = evaluate(
                    model, 
                    dataset_path, 
                    mode='test', 
                    test_data_name='FF',
                    epoch=epoch,
                )

                logger.debug(
                    f'(Test @ epoch {epoch})' 
                    f' auc: {auc:.5f}'
                    f' r_acc: {r_acc:.5f}'
                    f' f_acc: {f_acc:.5f}'
                    f' spe_acc: {spe_acc:.5f}' 
                    f' data_name: FF++'
                )

                auc, r_acc, f_acc, spe_acc = evaluate(
                    model, 
                    celeb_path, 
                    mode='test', 
                    test_data_name='celeb',
                    epoch=epoch,
                )

                logger.debug(
                    f'(Test @ epoch {epoch})' 
                    f' auc: {auc:.5f}'
                    f' r_acc: {r_acc:.5f}'
                    f' f_acc: {f_acc:.5f}'
                    f' spe_acc: {spe_acc:.5f}' 
                    f' data_name: celeb'
                )

                if auc > best_auc and r_acc > 0.5 and f_acc > 0.5:
                    best_auc = auc
                    logger.debug(f'Current Best AUC: {best_auc}')
                    model.save(path=f'{ckpt_name}_best.pth')
                model.model.train()
        epoch = epoch + 1

    model.save(path=f'{ckpt_name}_last.pth')
    # model.model.eval()
    # auc, r_acc, f_acc = evaluate(model, dataset_path, mode='test')
    # logger.debug(f'(Test @ epoch {epoch}) auc: {auc}, r_acc: {r_acc}, f_acc:{f_acc}')
