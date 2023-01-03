import os
import cv2
import numpy as np
import random
import math

import torch
from torch.utils import data
from torchvision import transforms as trans

from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc as cal_auc

from more_itertools import chunked
from matplotlib import pyplot as plt
from PIL import Image
from copy import deepcopy
import sys
import logging

import albumentations as A
from albumentations import DualTransform
import ttach as tta
from torchtoolbox.transform import Cutout

from tsne import *
from grad_cam_utils import GradCAM, show_cam_on_image, center_crop_img


fake_dict = {
    'Deepfakes': 1, 
    'Face2Face': 2,
    'FaceSwap': 3, 
    'NeuralTextures': 4, 
    # 'Deepfakes_Face2Face': 5, 
    # 'Deepfakes_FaceSwap': 6, 
    # 'Deepfakes_NeuralTextures': 7, 
    # 'Deepfakes_real': 8, 
    # 'Face2Face_FaceSwap': 9, 
    # 'Face2Face_NeuralTextures': 10, 
    # 'Face2Face_real': 11, 
    # 'FaceSwap_NeuralTextures': 12, 
    # 'FaceSwap_real': 13, 
    # 'NeuralTextures_real': 14,
}


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


inverser = UnNormalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )


def isotropically_resize_image(img, size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC):
    h, w = img.shape[:2]
    if max(w, h) == size:
        return img
    if w > h:
        scale = size / w
        h = h * scale
        w = size
    else:
        scale = size / h
        w = w * scale
        h = size
    interpolation = interpolation_up if scale > 1 else interpolation_down
    resized = cv2.resize(img, (int(w), int(h)), interpolation=interpolation)
    return resized


class IsotropicResize(DualTransform):
    def __init__(self, max_side, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC,
                 always_apply=False, p=1):
        super(IsotropicResize, self).__init__(always_apply, p)
        self.max_side = max_side
        self.interpolation_down = interpolation_down
        self.interpolation_up = interpolation_up

    def apply(self, img, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC, **params):
        return isotropically_resize_image(img, size=self.max_side, interpolation_down=interpolation_down,
                                          interpolation_up=interpolation_up)

    def apply_to_mask(self, img, **params):
        return self.apply(img, interpolation_down=cv2.INTER_NEAREST, interpolation_up=cv2.INTER_NEAREST, **params)

    def get_transform_init_args_names(self):
        return ("max_side", "interpolation_down", "interpolation_up")


def get_aug(img_arr):
    size = img_arr.shape[0]
    trans = A.Compose([
        A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
        A.Downscale(scale_min=0.7, scale_max=0.9,
                    interpolation=cv2.INTER_LINEAR, p=0.3),
        A.GaussNoise(p=0.1),
        A.GaussianBlur(blur_limit=3, p=0.05),
        A.HorizontalFlip(),
        A.OneOf([
                IsotropicResize(
                    max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                IsotropicResize(
                    max_side=size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
                IsotropicResize(
                    max_side=size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
                ], p=1),
        A.PadIfNeeded(min_height=size, min_width=size,
                      border_mode=cv2.BORDER_CONSTANT),
        A.OneOf([A.RandomBrightnessContrast(), A.FancyPCA(),
                A.HueSaturationValue()], p=0.3),
        # # add cut out method
        # A.Cutout(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2,
                           rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),

        # A.OneOf([
        #         A.CoarseDropout(),
        #         A.GridDistortion(),
        #         A.GridDropout(),
        #         A.OpticalDistortion()
        #         ]),
    ]
    )
    trans_img = trans(image=img_arr)['image']
    return trans_img


class FFDataset(data.Dataset):

    def __init__(self, dataset_root, frame_num=300, size=299, augment=True):
        self.data_root = dataset_root
        mode = 'real' if 'real' in dataset_root else 'fake'
        self.frame_num = frame_num
        self.train_list = self.collect_image(self.data_root, mode)
        self.augment = augment
        self.transform = trans.ToTensor()
        self.max_val = 1.
        self.min_val = -1.
        self.size = size

    def collect_image(self, root, mode):
        image_path_list = []
        for split in os.listdir(root):
            split_root = os.path.join(root, split)
            img_list = os.listdir(split_root)
            random.shuffle(img_list)
            img_list = img_list if len(img_list) < self.frame_num else img_list[:self.frame_num]
            for img in img_list:
                img_path = os.path.join(split_root, img)
                image_path_list.append(img_path)
        return image_path_list
        # return image_path_list * len(fake_dict) if mode == 'real' else image_path_list

    def read_image(self, path):
        img = Image.open(path)
        return img

    def resize_image(self, image, size):
        img = image.resize((size, size))
        return img

    def get_label(self, data_root: str):
        label = fake_dict[data_root.split('/')[-1]]
        return label

    def __getitem__(self, index):
        try:
            label = self.get_label(self.data_root)
        except:
            label = 0  # real
        image_path = self.train_list[index]
        img = self.read_image(image_path)
        img = self.resize_image(img, size=self.size)
        img = np.array(img)  # from pil to numpy format
        # do data aug
        if self.augment:
            img = get_aug(img)
        img = trans.functional.to_tensor(img)
        img = trans.functional.normalize(
            img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
        return img, label

    def collate(batch):
        data, label = list(zip(*batch))
        data = torch.stack(data, 0)
        label = torch.tensor(label)
        sample = (data, label)
        return sample

    def __len__(self):
        return len(self.train_list)


class CelebDataset(data.Dataset):

    def __init__(self, dataset_root, size=299, frame_num=50, augment=True):
        self.data_root = dataset_root
        self.frame_num = frame_num
        self.train_list = self.collect_image(self.data_root)
        self.augment = augment
        if augment:
            self.transform = trans.Compose([
                trans.RandomHorizontalFlip(p=0.5),
                trans.ToTensor()
            ])
            print("Augment True!")
        else:
            self.transform = trans.ToTensor()
        self.max_val = 1.
        self.min_val = -1.
        self.size = size

    def collect_image(self, root):
        image_path_list = []
        for split in os.listdir(root):
            split_root = os.path.join(root, split)
            img_list = os.listdir(split_root)
            random.shuffle(img_list)
            img_list = img_list if len(
                img_list) < self.frame_num else img_list[:self.frame_num]
            for img in img_list:
                img_path = os.path.join(split_root, img)
                image_path_list.append(img_path)
        return image_path_list

    def read_image(self, path):
        img = Image.open(path)
        return img

    def resize_image(self, image, size):
        img = image.resize((size, size))
        return img

    def get_label(self, data_root: str):
        label = fake_dict[data_root.split('/')[-1]]
        return label

    def __getitem__(self, index):
        image_path = self.train_list[index]
        img = self.read_image(image_path)
        img = self.resize_image(img, size=self.size)
        img = trans.functional.to_tensor(img)
        img = trans.functional.normalize(
            img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
        label = image_path.split('/')[-3]
        if label == 'real':
            label = 0
        elif label == 'fake':
            label = 1
        else:
            raise ValueError("label is not fake or real")
        return img, label

    def collate(batch):
        data, label = list(zip(*batch))
        data = torch.stack(data, 0)
        label = torch.tensor(label)
        sample = (data, label)
        return sample

    def __len__(self):
        return len(self.train_list)


def get_dataset(
    name='train', 
    size=299, 
    root='/mntnfs/sec_data2/yanzhiyuan/FFc23/',
    frame_num=300, 
    augment=True, 
    fake_list=['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
    # fake_list=[
    #         'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures', 
    #         'Deepfakes_Face2Face', 'Deepfakes_FaceSwap', 'Deepfakes_NeuralTextures', 
    #         'Deepfakes_real', 'Face2Face_FaceSwap', 'Face2Face_NeuralTextures', 
    #         'Face2Face_real', 'FaceSwap_NeuralTextures', 'FaceSwap_real', 
    #         'NeuralTextures_real',
    #         ]
    ):

    root = os.path.join(root, name)
    fake_root = os.path.join(root, 'fake')

    total_len = len(fake_list)
    dset_lst = []
    for i in range(total_len):
        fake = os.path.join(fake_root, fake_list[i])
        dset = FFDataset(fake, frame_num, size, augment)
        dset.size = size
        dset_lst.append(dset)
    return torch.utils.data.ConcatDataset(dset_lst), total_len


def evaluate(model, data_path, mode='val', test_data_name='celeb', epoch=0):
    root = data_path
    origin_root = root
    assert test_data_name != 'celeb' \
        or test_data_name != 'FF', "only support celeb and FF++ two dataset name"
    
    if test_data_name == 'FF':
        root = os.path.join(data_path, mode)
    real_root = os.path.join(root, 'real')
    fake_root = os.path.join(root, 'fake')
    
    # *** FF++ *** #
    if test_data_name == 'FF':
        dataset_real = FFDataset(
            dataset_root=real_root, 
            size=256, 
            frame_num=50, 
            augment=False
        )
        dataset_fake, _ = get_dataset(
            name=mode, 
            root=origin_root, 
            size=256, 
            frame_num=50, 
            augment=False, 
            # fake_list=[
            # 'Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures', 
            # 'Deepfakes_Face2Face', 'Deepfakes_FaceSwap', 'Deepfakes_NeuralTextures', 
            # 'Deepfakes_real', 'Face2Face_FaceSwap', 'Face2Face_NeuralTextures', 
            # 'Face2Face_real', 'FaceSwap_NeuralTextures', 'FaceSwap_real', 
            # 'NeuralTextures_real',
            # ]
            fake_list=['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures'],
        )
        dataset_img = torch.utils.data.ConcatDataset(
            [dataset_real, dataset_fake]
        )
    # *** celeb *** #
    elif test_data_name == 'celeb':
        dataset_real = CelebDataset(
            dataset_root=real_root, 
            size=256, 
            frame_num=50,
            augment=False
        )
        dataset_fake = CelebDataset(
            dataset_root=fake_root, 
            size=256, 
            frame_num=50, 
            augment=False
        )
        dataset_img = torch.utils.data.ConcatDataset(
            [dataset_real, dataset_fake]
        )

    # TTA
    transforms = tta.Compose(
        [
            tta.HorizontalFlip(),
            # tta.Rotate90(angles=[0, 90]),
            # tta.Scale(scales=[1, 2]),
            # tta.Multiply(factors=[0.9, 1, 1.1]),
        ]
    )
    tta_model = tta.ClassificationTTAWrapper(model, transforms)
    # tta_model = tta.ClassificationTTAWrapper(model, transforms, merge_mode='mean')

    TSNE = False
    CAM = False
    TSNE_PATH = 'tsne'
    CAM_PATH = 'grad_cam_save'
    video_auc = False
    TTA = False

    if TSNE:
        os.makedirs(TSNE_PATH, exist_ok=True)
    if CAM:
        os.makedirs(CAM_PATH, exist_ok=True)

    with torch.no_grad():
        y_true, y_pred = [], []
        spe_true, spe_pred = [], []
        feature_vector_list = []
        content_list = []
        specific_list = []

        bz = 64
        for i, d in enumerate(dataset_img.datasets):
            dataloader = torch.utils.data.DataLoader(
                dataset=d,
                batch_size=bz,
                shuffle=True,
                num_workers=8
            )
            for index, (img, label) in enumerate(dataloader):
                # get specific label
                spe_label = deepcopy(label)
                # make instance label to share label
                # only use share label for evaluation
                if not label.any() == 0:
                    label = torch.ones_like(label)
                img = img.detach().cuda()
                if CAM:
                    cam_img = img[:10, :, :, :].detach()

                # *** normal test *** #
                if not TTA:
                    output, feat = model.forward(img, train=False)

                    feat_vec = (feat[1]
                                .detach()
                                .cpu()
                                .numpy()
                                )
                    spe_vec = (feat[0]
                                .detach()
                                .cpu()
                                .numpy()
                                )
                    con_vec = (feat[-1]
                                .detach()
                                .cpu()
                                .numpy()
                                )
                    pred_sp = (output[0]
                               .data.max(1, keepdim=True)[1]
                               .detach()
                               .cpu()
                               .numpy()
                               .flatten()
                               .tolist()
                               )
                    # pred_sh = (output[1]
                    #            .data.max(1, keepdim=True)[1]
                    #            .detach()
                    #            .cpu()
                    #            .numpy()
                    #            .flatten()
                    #            .tolist()
                    #            )
                    pred_sh = (output[1]
                               .sigmoid()
                               .flatten()
                               .tolist()
                               )
                    true_sh = (label
                               .flatten()
                               .tolist()
                               )
                    true_sp = (spe_label
                               .flatten()
                               .tolist()
                               )

                    if video_auc:
                        # video
                        # specific acc
                        spe_pred.append(np.mean(pred_sp))  # 0:spe, 1:sha
                        spe_true.append(np.argmax(np.bincount(true_sp)))
                        # share acc
                        y_pred.append(np.mean(pred_sh))  # 0:spe, 1:sha
                        y_true.append(np.argmax(np.bincount(true_sh)))
                    else:
                        # frame
                        # specific acc
                        spe_pred.extend(pred_sp)
                        spe_true.extend(true_sp)
                        # share acc
                        # print(f"y_true: {np.array(true_sh).shape}")
                        # print(f"y_pred: {np.array(pred_sh).shape}")
                        y_pred.extend(pred_sh)  # 0:spe, 1:sha
                        y_true.extend(true_sh)
                        # feature vector
                        feature_vector_list.extend(feat_vec)
                        # content vector
                        content_list.extend(con_vec)
                        # specific vector
                        specific_list.extend(spe_vec)
                else:
                    # *** TTA *** #
                    tmp_pred, tmp_true = [], []
                    for transformer in transforms:  # custom transforms or e.g. tta.aliases.d4_transform()
                        # augment image
                        augmented_image = transformer.augment_image(img)
                        # pass to model
                        output, feat = model.forward(augmented_image, train=False)
                        feat_vec = (feat[1]
                                    .detach()
                                    .cpu()
                                    .numpy()
                                    )
                        spe_vec = (feat[0]
                                    .detach()
                                    .cpu()
                                    .numpy()
                                    )
                        con_vec = (feat[-1]
                                    .detach()
                                    .cpu()
                                    .numpy()
                                    )
                        pred_sp = (output[0]
                                    .data.max(1, keepdim=True)[1]
                                    .detach()
                                    .cpu()
                                    .numpy()
                                    .flatten()
                                    .tolist()
                                    )
                        # pred_sh = (output[1]
                        #            .data.max(1, keepdim=True)[1]
                        #            .detach()
                        #            .cpu()
                        #            .numpy()
                        #            .flatten()
                        #            .tolist()
                        #            )
                        pred_sh = (output[1]
                                    .sigmoid()
                                    .flatten()
                                    .tolist()
                                    )
                        true_sh = (label
                                    .flatten()
                                    .tolist()
                                    )
                        true_sp = (spe_label
                                    .flatten()
                                    .tolist()
                                    )
                        # 0:spe, 1:sha
                        tmp_pred.extend(pred_sh)
                        tmp_true.extend(true_sh)
                    pred = [np.mean(x)
                            for x in chunked(tmp_pred, len(transforms))]
                    true = [np.mean(x)
                            for x in chunked(tmp_true, len(transforms))]
                    if video_auc:
                        # video
                        y_pred.append(np.mean(pred))
                        y_true.append(np.argmax(np.bincount(true)))
                    else:
                        # frame
                        # specific acc
                        spe_pred.extend(pred_sp)
                        spe_true.extend(true_sp)
                        # share acc
                        y_pred.extend(pred)  # 0:spe, 1:sha
                        y_true.extend(true)
                        # feature vector
                        feature_vector_list.extend(feat_vec)
                        # content vector
                        content_list.extend(con_vec)
                        # specific vector
                        specific_list.extend(spe_vec)
                    torch.cuda.empty_cache()

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    spe_true, spe_pred = np.array(spe_true), np.array(spe_pred)
    # print(f"y_true: {y_true.shape}")
    # print(f"y_pred: {y_pred.shape}")
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    AUC = cal_auc(fpr, tpr)

    idx_real = np.where(y_true == 0)[0]
    idx_fake = np.where(y_true == 1)[0]

    r_acc = accuracy_score(y_true[idx_real], y_pred[idx_real] > 0.5)
    f_acc = accuracy_score(y_true[idx_fake], y_pred[idx_fake] > 0.5)

    spe_acc = accuracy_score(spe_true, spe_pred)

    # tsne
    if TSNE:
        sample_num = 500
        assert len(feature_vector_list) == len(content_list), \
            "Length between content and share should be the same"
        # index_list = np.random.randint(0, len(feature_vector_list), sample_num)
        # random.shuffle(idx_real)
        # random.shuffle(idx_fake)
        idx_real = idx_real[:sample_num//2]
        idx_fake = idx_fake[:sample_num//2]
        index_list = np.vstack((idx_real, idx_fake)).flatten()
        l = y_true[index_list]
        l_sep = spe_true[index_list]

        # share visualization between real and fake
        X = np.array(feature_vector_list)[index_list, :]
        Y = tsne(X, 2, 50, 20.0)
        Y_1 = Y[l==0, :]
        Y_2 = Y[l==1, :]
        plt.scatter(Y_1[:, 0], Y_1[:, 1], 20, color='red', label='real')
        plt.scatter(Y_2[:, 0], Y_2[:, 1], 20, color='blue', label='fake')
        plt.legend()
        plt.savefig(os.path.join(TSNE_PATH, f'share_{test_data_name}_{epoch}.png'))
        np.save(os.path.join(TSNE_PATH, f'share_X_{test_data_name}_{epoch}.npy'), X)
        np.save(os.path.join(TSNE_PATH, f'share_Y_{test_data_name}_{epoch}.npy'), Y)
        np.save(os.path.join(TSNE_PATH, f'share_label_{test_data_name}_{epoch}.npy'), l)
        plt.clf()

        # specific visualization between real and fake
        X = np.array(specific_list)[index_list, :]
        Y = tsne(X, 2, 50, 20.0)
        Y_0 = Y[l_sep==0, :]
        Y_1 = Y[l_sep==1, :]
        Y_2 = Y[l_sep==2, :]
        Y_3 = Y[l_sep==3, :]
        Y_4 = Y[l_sep==4, :]
        plt.scatter(Y_0[:, 0], Y_0[:, 1], 20, color='red', label='real')
        plt.scatter(Y_1[:, 1], Y_1[:, 1], 20, color='orange', label='Deepfakes')
        plt.scatter(Y_2[:, 1], Y_2[:, 1], 20, color='blue', label='Face2Face')
        plt.scatter(Y_3[:, 1], Y_3[:, 1], 20, color='green', label='FaceSwap')
        plt.scatter(Y_4[:, 1], Y_4[:, 1], 20, color='slategrey', label='NeuralTextures')
        plt.legend()
        plt.savefig(os.path.join(TSNE_PATH, f'specific_{test_data_name}_{epoch}.png'))
        np.save(os.path.join(TSNE_PATH, f'specific_X_{test_data_name}_{epoch}.npy'), X)
        np.save(os.path.join(TSNE_PATH, f'specific_Y_{test_data_name}_{epoch}.npy'), Y)
        np.save(os.path.join(TSNE_PATH, f'specific_label_{test_data_name}_{epoch}.npy'), l_sep)
        plt.clf()

        # content visualization between real and fake
        X = np.array(content_list)[index_list, :]
        Y = tsne(X, 2, 50, 20.0)
        Y_1 = Y[l==0, :]
        Y_2 = Y[l==1, :]
        plt.scatter(Y_1[:, 0], Y_1[:, 1], 20, color='red', label='real')
        plt.scatter(Y_2[:, 0], Y_2[:, 1], 20, color='blue', label='fake')
        plt.legend()
        plt.savefig(os.path.join(TSNE_PATH, f'content_{test_data_name}_{epoch}.png'))
        np.save(os.path.join(TSNE_PATH, f'content_X_{test_data_name}_{epoch}.npy'), X)
        np.save(os.path.join(TSNE_PATH, f'content_Y_{test_data_name}_{epoch}.npy'), Y)
        np.save(os.path.join(TSNE_PATH, f'content_label_{test_data_name}_{epoch}.npy'), l)
        plt.clf()

    # grad cam visualization for feature map of both the fingerprint and content
    if CAM:
        _, c, h, w = cam_img.shape
        for i in range(10):
            cam_one_img = cam_img[i, :, :, :].unsqueeze(0)
            numpy_img = (
                inverser(cam_one_img.squeeze(0))  # inverse normalization
                .permute(1,2,0)                   # inverse to_tensor (adjust channel order)
                .detach()
                .cpu()
                .numpy()
                * 255.
            )
            # fingerprint
            target_layers = [model.model.module.encoder_f.block12]
            cam = GradCAM(
                model=model.model.module, 
                target_layers=target_layers, 
                use_cuda=True,
            )
            grayscale_cam = (
                cam(input_tensor=cam_one_img, target_category=1)[0, :]
            )
            visualization = show_cam_on_image(
                numpy_img.astype(dtype=np.float32) / 255.,
                grayscale_cam, 
                use_rgb=True,
            )
            img_name = f"{epoch}_{index}_{i}_{test_data_name}_encoder_f_block12.png"
            cv2.imwrite(
                os.path.join(CAM_PATH, img_name), 
                cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB)
            )

    return AUC, r_acc, f_acc, spe_acc


# python 3.7
"""Utility functions for logging."""

__all__ = ['setup_logger']

DEFAULT_WORK_DIR = 'results'

def setup_logger(work_dir=None, logfile_name='log.txt', logger_name='logger'):
    """Sets up logger from target work directory.

    The function will sets up a logger with `DEBUG` log level. Two handlers will
    be added to the logger automatically. One is the `sys.stdout` stream, with
    `INFO` log level, which will print improtant messages on the screen. The other
    is used to save all messages to file `$WORK_DIR/$LOGFILE_NAME`. Messages will
    be added time stamp and log level before logged.

    NOTE: If `logfile_name` is empty, the file stream will be skipped. Also,
    `DEFAULT_WORK_DIR` will be used as default work directory.

    Args:
    work_dir: The work directory. All intermediate files will be saved here.
        (default: None)
    logfile_name: Name of the file to save log message. (default: `log.txt`)
    logger_name: Unique name for the logger. (default: `logger`)

    Returns:
    A `logging.Logger` object.

    Raises:
    SystemExit: If the work directory has already existed, of the logger with
        specified name `logger_name` has already existed.
    """

    logger = logging.getLogger(logger_name)
    if logger.hasHandlers():  # Already existed
        raise SystemExit(f'Logger name `{logger_name}` has already been set up!\n'
                         f'Please use another name, or otherwise the messages '
                         f'may be mixed between these two loggers.')

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
    # Print log message with `INFO` level or above onto the screen.
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    if not logfile_name:
        return logger

    work_dir = work_dir or DEFAULT_WORK_DIR
    logfile_name = os.path.join(work_dir, logfile_name)
    # if os.path.isfile(logfile_name):
    #   print(f'Log file `{logfile_name}` has already existed!')
    #   while True:
    #     decision = input(f'Would you like to overwrite it (Y/N): ')
    #     decision = decision.strip().lower()
    #     if decision == 'n':
    #       raise SystemExit(f'Please specify another one.')
    #     if decision == 'y':
    #       logger.warning(f'Overwriting log file `{logfile_name}`!')
    #       break

    os.makedirs(work_dir, exist_ok=True)

    # Save log message with all levels in log file.
    fh = logging.FileHandler(logfile_name)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
