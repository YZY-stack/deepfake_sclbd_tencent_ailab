import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
osenvs = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
import sys
import time
import torch
import torch.nn
from torchvision import utils as vutils

from utils import evaluate, get_dataset, FFDataset, setup_logger
from trainer import Trainer
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
visualization = False
mode = 'FAD' # ['Original', 'FAD', 'LFS', 'Both', 'Mix']
ckpt_dir = './weights'
ckpt_name = 'paper_disfin_notta_bs32_lr00002_mlp_addlambda_imgshow_new'
img_save_path = 'img_save'
os.makedirs(img_save_path, exist_ok=True)


if __name__ == '__main__':
    dataset = FFDataset(dataset_root=os.path.join(dataset_path, 'train', 'real'), size=256, frame_num=100, augment=True)
    dataloader_real = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size // 2,
        shuffle=True,
        num_workers=8,
        collate_fn=FFDataset.collate,
    )
    
    len_dataloader = dataloader_real.__len__()

    dataset_img, total_len =  get_dataset(name='train', size=256, root=dataset_path, frame_num=100, augment=True)
    dataloader_fake = torch.utils.data.DataLoader(
        dataset=dataset_img,
        batch_size=batch_size // 2,
        shuffle=True,
        num_workers=8,
        collate_fn=FFDataset.collate,
    )

    # init checkpoint and logger
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    logger = setup_logger(ckpt_path, 'result.log', 'logger')
    best_val = 0.
    ckpt_model_name = 'best.pkl'
    
    # train
    model = Trainer(gpu_ids, mode, pretrained_path)
    model.total_steps = 0
    epoch = 0
    best_auc = 0.
    
    while epoch < max_epoch:

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

            # pair combination, label augmentation
            pair_index = random.randint(0, 1)
            if pair_index == 0:  # fake
                aug_label = torch.ones_like(label_real)
                aug_label_instance = deepcopy(label_fake_instance)
            elif pair_index == 1:  # real
                aug_label = torch.zeros_like(label_real)
                aug_label_instance = deepcopy(aug_label)
            else:
                raise ValueError("pair index should be 0 or 1")

            # *** share label task *** #
            data = torch.cat([data_real,data_fake],dim=0)
            label = torch.cat([label_real,label_fake],dim=0)

            # # manually shuffle
            # idx = list(range(data.shape[0]))
            # random.shuffle(idx)
            # data = data[idx]
            # label = label[idx]

            data = data.detach()
            label = label.detach()

            model.set_input(data,label)

            stu_fea, stu_cla = model.model(model.input, pair_index=pair_index)
            (spe_out, sha_out), \
            reconstruction_image_1, \
            reconstruction_image_2, \
            forgery_image_12, \
            f1_spe, f3_spe \
            = stu_cla

            # *** share label task *** #
            # model.label = label.to(model.device)
            loss_share = model.optimize_weight(sha_out)

            # *** instance label task *** #
            spe_label = torch.cat([label_real,label_fake_instance],dim=0)

            # spe_label = spe_label[idx]
            spe_label = spe_label.detach()

            model.label = spe_label.to(model.device)
            loss_specific = model.optimize_weight_ce(spe_out)

            # *** mse construction task *** # 
            loss_reconstruction_1, \
            loss_reconstruction_2, \
            loss_reconstruction_3, \
            loss_f1_f3 \
                = model.optimize_weight_mse(
                data_real.to(model.device),
                data_fake.to(model.device),
                reconstruction_image_1,
                reconstruction_image_2,
                forgery_image_12,
                f1_spe, 
                f3_spe,
            )

            loss_reconstruction = loss_reconstruction_1 + \
                                  loss_reconstruction_2 + \
                                  loss_reconstruction_3 + \
                                  loss_f1_f3

            # total loss
            loss = model.get_final_loss(
                loss_share, 
                loss_specific,
                loss_reconstruction
            )

            if model.total_steps % loss_freq == 0:
                logger.debug(
                    f'loss: {loss:.5f}' 
                    f'spe_loss: {loss_specific:.5f}' 
                    f'sha_loss: {loss_share:.5f}' 
                    f'mse_loss: {loss_reconstruction:.5f}' 
                    f'loss_fake_reconstruction1: {loss_reconstruction_1:.5f}' 
                    f'loss_real_reconstruction2: {loss_reconstruction_2:.5f}' 
                    f'loss_forgery_real: {loss_reconstruction_3:.5f}' 
                    f'loss_f1_f3_spe: {loss_f1_f3:.5f}' 
                    f'at step: {model.total_steps}'
                )
                # save img
                if visualization:
                    vutils.save_image(
                        reconstruction_image_1.detach().cpu(), 
                        f'{img_save_path}/epoch{epoch}_step{model.total_steps}_reconstruction_image_1.png')

                    vutils.save_image(
                        reconstruction_image_2.detach().cpu(), 
                        f'{img_save_path}/epoch{epoch}_step{model.total_steps}_reconstruction_image_2.png')

                    vutils.save_image(
                        forgery_image_12.detach().cpu(), 
                        f'{img_save_path}/epoch{epoch}_step{model.total_steps}_forgery_image_12.png')

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
                auc, r_acc, f_acc, spe_acc = evaluate(model, dataset_path, mode='test', test_data_name='FF')
                logger.debug(f'(Test @ epoch {epoch}) auc: {auc:.5f}, r_acc: {r_acc:.5f}, f_acc: {f_acc:.5f}, spe_acc: {spe_acc:.5f}, data_name: FF++')
                auc, r_acc, f_acc, spe_acc = evaluate(model, celeb_path, mode='test', test_data_name='celeb')
                logger.debug(f'(Test @ epoch {epoch}) auc: {auc:.5f}, r_acc: {r_acc:.5f}, f_acc: {f_acc:.5f}, spe_acc: {spe_acc:.5f}, data_name: celeb')
                if auc > best_auc:
                    best_auc = auc
                    logger.debug(f'Current Best AUC: {best_auc}')
                    model.save(path=f'{ckpt_name}.pth')
                model.model.train()
        epoch = epoch + 1

    # model.model.eval()
    # auc, r_acc, f_acc = evaluate(model, dataset_path, mode='test')
    # logger.debug(f'(Test @ epoch {epoch}) auc: {auc}, r_acc: {r_acc}, f_acc:{f_acc}')
