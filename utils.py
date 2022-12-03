import torch
import os
import cv2
import albumentations as A
from albumentations import DualTransform
import ttach as tta
import numpy as np
import random
from torch.utils import data
from torchvision import transforms as trans
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc as cal_auc
from PIL import Image
from copy import deepcopy
from more_itertools import chunked
import sys
import logging


fake_dict = {'Deepfakes': 1, 'Face2Face': 2,
             'FaceSwap': 3, 'NeuralTextures': 4}


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
    # trans = A.Compose([
    #         A.OneOf([
    #             A.RandomGamma(gamma_limit=(60, 120), p=0.9),
    #             A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9),
    #             A.CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=0.9),
    #             A.GaussianBlur(),
    #         ]),
    #         A.HorizontalFlip(p=0.5),
    #         A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20,
    #                             interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, p=1),
    #         A.ImageCompression(quality_lower=60, quality_upper=90, p=0.5),
    #         A.OneOf([
    #                 A.CoarseDropout(),
    #                 A.GridDistortion(),
    #                 A.GridDropout(),
    #                 A.OpticalDistortion()
    #                 ]),
    # ])
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
                A.HueSaturationValue()], p=0.7),
        A.ToGray(p=0.2),
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
        self.frame_num = frame_num
        self.train_list = self.collect_image(self.data_root)
        self.augment = augment
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
            img_list = (img_list 
            if len(img_list) < self.frame_num 
            else img_list[:self.frame_num]
            )
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
        # img = trans.functional.to_tensor(img)
        img = self.transform(img)
        img = img * (self.max_val - self.min_val) + self.min_val
        # img = trans.functional.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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
        img = self.transform(img)
        img = img * (self.max_val - self.min_val) + self.min_val
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


def evaluate(model, data_path, mode='val', test_data_name='celeb'):
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
            fake_list=['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']
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
            tta.Multiply(factors=[0.9, 1, 1.1]),
        ]
    )
    tta_model = tta.ClassificationTTAWrapper(model, transforms)

    TTA = False
    video_auc = False

    with torch.no_grad():
        y_true, y_pred = [], []
        spe_true, spe_pred = [], []

        bz = 64
        for i, d in enumerate(dataset_img.datasets):
            dataloader = torch.utils.data.DataLoader(
                dataset=d,
                batch_size=bz,
                shuffle=True,
                num_workers=8
            )
            for img, label in dataloader:
                # get specific label
                spe_label = deepcopy(label)
                # make instance label to share label
                # only use share label for evaluation
                if not label.any() == 0:
                    label = torch.ones_like(label)
                img = img.detach().cuda()

                # *** normal test *** #
                if not TTA:
                    output = model.forward(img, train=False)
                    pred_sp = (output[0]
                               .data.max(1, keepdim=True)[1]
                               .detach()
                               .cpu()
                               .numpy()
                               .flatten()
                               .tolist()
                               )
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
                        y_pred.extend(pred_sh)  # 0:spe, 1:sha
                        y_true.extend(true_sh)
                else:
                    # TODO implement this with specific acc
                    # *** TTA *** #
                    tmp_pred, tmp_true = [], []
                    for transformer in transforms:  # custom transforms or e.g. tta.aliases.d4_transform()
                        # augment image
                        augmented_image = transformer.augment_image(img)
                        # pass to model
                        output = model.forward(augmented_image, False)
                        # 0:spe, 1:sha
                        tmp_pred.extend(output[1].sigmoid().flatten().tolist())
                        tmp_true.extend(label.flatten().tolist())
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
                        y_pred.extend(pred)
                        y_true.extend(true)
                    torch.cuda.empty_cache()

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    spe_true, spe_pred = np.array(spe_true), np.array(spe_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=1)
    AUC = cal_auc(fpr, tpr)

    idx_real = np.where(y_true == 0)[0]
    idx_fake = np.where(y_true == 1)[0]

    r_acc = accuracy_score(y_true[idx_real], y_pred[idx_real] > 0.5)
    f_acc = accuracy_score(y_true[idx_fake], y_pred[idx_fake] > 0.5)

    spe_acc = accuracy_score(spe_true, spe_pred)

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
