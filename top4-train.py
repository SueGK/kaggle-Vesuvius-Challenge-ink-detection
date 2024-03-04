#!/usr/bin/env python
# coding: utf-8

# ## summary
# 
# * 2.5d segmentation
#     *  segmentation_models_pytorch 
#     *  Unet
# * use only 6 slices in the middle
# * slide inference

# In[1]:

# from resnet3d import generate_model

import sys
sys.path.append('/root/autodl-tmp/VCInkDectection/input/resnet')
sys.path.append('/root/autodl-tmp/VCInkDectection/input/Efficient-3DCNNs-master')


from resnet import generate_model
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, log_loss
import pickle
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import warnings
import sys
import os
import gc
import sys
import math
import time
import random
import shutil
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter
import cv2

import scipy as sp
import numpy as np

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from functools import partial

import argparse
import importlib
import torch
import torch.nn as nn
from torch.optim import Adam, SGD, AdamW

import datetime
import segmentation_models_pytorch as smp



import numpy as np
from torch.utils.data import DataLoader, Dataset
import cv2
import datetime
import torch
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform


# ## config

# In[7]:


import os
import albumentations as A
from albumentations.pytorch import ToTensorV2

import argparse


parser = argparse.ArgumentParser()
parser.add_argument(
    "--valid_id",
    type=int,
    default=1,
    help='valid_id'
    )

parser.add_argument(
    "--model_name",
    type=str,
    default='Unet',
    help='model_name'
    )

parser.add_argument(
    "--image_size",
    type=int,
    default=256,
    help='image_size'
    )


parser.add_argument(
    "--backbone_name",
    type=str,
    default='efficientnet-b1',
    help='model_name'
    )

parser.add_argument(
    "--batch_size",
    type=int,
    default=16,
    help='batch_size'
    )

parser.add_argument(
    "--epochs",
    type=int,
    default=15,
    help='epochs'
    )

parser.add_argument(
    "--diceloss",
    type=int,
    default=0,
    help='diceloss'
    )

parser.add_argument(
    "--tverskyloss",
    type=int,
    default=0,
    help='tverskyloss'
    )

parser.add_argument(
    "--in_chans",
    type=int,
    default=22,
    help='in_chans'
    )

parser.add_argument(
    "--resnet_depth",
    type=int,
    default=152,
    help='resnet_depth'
    )

parser.add_argument(
    "--resnet_weight",
    type=str,
    default='r3d152_KM_200ep.pth',
    help='resnet_weight'
    )

parser.add_argument(
    "--slicing_num",
    type=int,
    default=4300,
    help='balance in mask pixel for 4Fold Training'
    )

parser.add_argument(
    "--cropping_num_min",
    type=int,
    default=12,
    help='balance in mask pixel for 4Fold Training'
    )

parser.add_argument(
    "--cropping_num_max",
    type=int,
    default=22,
    help='balance in mask pixel for 4Fold Training'
    )

parser.add_argument(
    "--fbeta_gamma",
    type=int,
    default=2,
    help='balance in mask pixel for 4Fold Training'
    )

parser.add_argument(
    "--clip_min",
    type=int,
    default=50,
    help='balance in mask pixel for 4Fold Training'
    )

parser.add_argument(
    "--clip_max",
    type=int,
    default=200,
    help='balance in mask pixel for 4Fold Training'
    )

parser.add_argument(
    "--ls",
    type=float,
    default=0.3,
    help='balance in mask pixel for 4Fold Training'
    )


parser.add_argument(
    "--model",
    type=str,
    default='resnet152',
    help='choose between resnet152, resnet200, resnext101'
    )


args = parser.parse_args(args=[])

d = datetime.datetime.now()
year, month, day, hour, minute, second = d.year, d.month, d.day, d.hour, d.minute, d.second
if len(str(month)) == 1:
    month = '0' + str(month)
if len(str(day)) == 1:
    day = '0' + str(day)
if len(str(hour)) == 1:
    hour = '0' + str(hour)

current_day = f'{year}_{month}_{day}'

class CFG:
    # ============== comp exp name =============
    comp_name = 'vesuvius'

    # comp_dir_path = './'
    comp_dir_path = "/root/autodl-tmp/VCInkDectection/input" # '/kaggle/input'
    comp_folder_name = 'vesuvius-challenge-ink-detection'
    # comp_dataset_path = f'{comp_dir_path}datasets/{comp_folder_name}/'
    comp_dataset_path = f'{comp_dir_path}/{comp_folder_name}/'


    # ============== pred target =============
    target_size = 1

    # ============== model cfg =============
    model_name = args.model_name
    backbone = args.backbone_name
    #backbone = 'se_resnext50_32x4d'

    in_chans = args.in_chans # 65
    # ============== training cfg =============
    size = args.image_size
    tile_size = args.image_size
    stride_rate = 3
    stride = tile_size // stride_rate

    # cropping_num = args.cropping_num

    # exp_name = f'{current_day}_{model_name}_{backbone}_{tile_size}_{in_chans}_batchsize{args.batch_size}_diceloss{args.diceloss}_tverskyloss{args.tverskyloss}_3DCNN_depth{args.resnet_depth}'
    exp_name = f'{current_day}_{tile_size}_{in_chans}_batchsize{args.batch_size}_diceloss{args.diceloss}_tverskyloss{args.tverskyloss}_3DCNN_depth{args.resnet_depth}_stride{stride_rate}_cropping_num{args.cropping_num_min}-{args.cropping_num_max}_ls{args.ls}_clip{args.clip_min}_{args.clip_max}_CosineAnnealingWarmUpRestarts(optimizer, T_0=9, T_mult=1, eta_max=1e-4, T_up=3, gamma=0.5)'

    train_batch_size = args.batch_size # 32
    valid_batch_size = train_batch_size * 2
    use_amp = True

    scheduler = 'GradualWarmupSchedulerV2'
    # scheduler = 'CosineAnnealingLR'
    epochs = args.epochs # 30

    # adamW warmupあり
    warmup_factor = 10
    # lr = 1e-4 / warmup_factor
    lr = 1e-3 / warmup_factor

    # ============== fold =============
    valid_id = args.valid_id

    # objective_cv = 'binary'  # 'binary', 'multiclass', 'regression'
    metric_direction = 'maximize'  # maximize, 'minimize'
    # metrics = 'dice_coef'

    # ============== fixed =============
    pretrained = True
    inf_weight = 'best'  # 'best'

    min_lr = 1e-7
    weight_decay = 1e-6
    max_grad_norm = 1000

    print_freq = 50
    num_workers = 8

    seed = 42

    # ============== set dataset path =============
    print('set dataset path')
    output_save_folder = 'checkpoints'
    outputs_path = f'/root/autodl-tmp/VCInkDectection/working/{output_save_folder}/{current_day}/{exp_name}/{args.valid_id}/'

    submission_dir = outputs_path + 'submissions/'
    submission_path = submission_dir + f'submission_{exp_name}.csv'

    model_dir = outputs_path + \
        f'{comp_name}-models/'

    mask_dir = outputs_path + 'mask_pred'

    figures_dir = outputs_path + 'figures/'

    log_dir = outputs_path + 'logs/'
    log_path = log_dir + f'{exp_name}.txt'

    # ============== augmentation =============
    train_aug_list = [
        # A.RandomResizedCrop(
        #     size, size, scale=(0.85, 1.0)),
        #A.Resize(size, size),
        A.HorizontalFlip(p=0.6),
        A.VerticalFlip(p=0.6),
        A.RandomGamma(gamma_limit=(50, 150), p=0.6),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=360, interpolation=0, border_mode=0, p=0.6),
        A.OneOf([
                A.GaussNoise(var_limit=[10, 30]),
                A.GaussianBlur(),
                # A.MotionBlur(),
                ], p=0.6),
        # A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        A.CoarseDropout(max_holes=4, max_width=int(size * 0.2), max_height=int(size * 0.2),
                        mask_fill_value=0, p=0.6),
        # A.Cutout(max_h_size=int(size * 0.6),
        #          max_w_size=int(size * 0.6), num_holes=1, p=1.0),
        A.Normalize(
            mean= [0] * args.in_chans,
            std= [1] * args.in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]

    valid_aug_list = [
        A.Resize(size, size),
        A.Normalize(
            mean= [0] * args.in_chans,
            std= [1] * args.in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]


# ## helper

# In[8]:


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# In[9]:


def init_logger(log_file):
    from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

def set_seed(seed=None, cudnn_deterministic=True):
    if seed is None:
        seed = 42

    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = False


# In[10]:


def make_dirs(cfg):
    for dir in [cfg.model_dir, cfg.figures_dir, cfg.submission_dir, cfg.log_dir, cfg.mask_dir]:
        os.makedirs(dir, exist_ok=True)


# In[11]:


def cfg_init(cfg, mode='train'):
    set_seed(cfg.seed)
    # set_env_name()
    # set_dataset_path(cfg)

    if mode == 'train':
        make_dirs(cfg)


# In[12]:


cfg_init(CFG)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Logger = init_logger(log_file=CFG.log_path)

Logger.info('\n\n-------- exp_info -----------------')
# Logger.info(datetime.datetime.now().strftime('%Y年%m月%d日 %H:%M:%S'))




def read_image_mask(fragment_id):

    images = []

    # idxs = range(65)
    mid = 65 // 2
    start = mid - CFG.in_chans // 2
    end = mid + CFG.in_chans // 2
    idxs = list(range(start, end)) # (21, 43)
    # idxs = [14, 15] + idxs


    for i in tqdm(idxs):
        
        image = cv2.imread(CFG.comp_dataset_path + f"train/{fragment_id}/surface_volume/{i:02}.tif", 0)

        pad0 = (CFG.tile_size - image.shape[0] % CFG.tile_size)
        pad1 = (CFG.tile_size - image.shape[1] % CFG.tile_size)

        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)

        images.append(image)
    images = np.stack(images, axis=2)
    #images = np.media`n(images, axis=2)[..., None].astype('uint8')

    
    mask = cv2.imread(CFG.comp_dataset_path + f"train/{fragment_id}/inklabels.png", 0)
    mask = np.pad(mask, [(0, pad0), (0, pad1)], constant_values=0)

    mask = mask.astype('float32')
    mask /= 255.0

    mask_prag = cv2.imread(CFG.comp_dataset_path + f"train/{fragment_id}/mask.png", 0)
    mask_prag = np.pad(mask_prag, [(0, pad0), (0, pad1)], constant_values=0)

    mask_prag = mask_prag.astype('float32')
    mask_prag /= 255.0

    return images, mask, mask_prag


# In[14]:


def get_train_valid_dataset():
    train_images = []
    train_masks = []

    valid_images = []
    valid_masks = []
    valid_xyxys = []

    valid_masks_frag = []

    for fragment_id in range(1, 5): # 1,2(fragment2a),3,4(fragment2b)

        if fragment_id == 2:
            image, mask, mask_frag = read_image_mask(2)
            image, mask, mask_frag = image[:, :args.slicing_num, :], mask[:, :args.slicing_num], mask_frag[:, :args.slicing_num]
            # image:(14848, 4300, 22), mask(inklabels): (14848, 4300), mask_frag(mask): (14848, 4300)
        elif fragment_id == 4:
            image, mask, mask_frag = read_image_mask(2)
            image, mask, mask_frag = image[:, args.slicing_num:, :], mask[:, args.slicing_num:], mask_frag[:, args.slicing_num:]
             
        else:
            image, mask, mask_frag = read_image_mask(fragment_id) # image shape (8192, 6400, 22)

        x1_list = list(range(0, image.shape[1]-CFG.tile_size+1, CFG.stride)) #range(0, 6145, 85) len 73
        y1_list = list(range(0, image.shape[0]-CFG.tile_size+1, CFG.stride)) #range(0, 7937, 85) len 94

        for y1 in y1_list:
            for x1 in x1_list:
                y2 = y1 + CFG.tile_size
                x2 = x1 + CFG.tile_size
                # xyxys.append((x1, y1, x2, y2))
        
                if fragment_id == CFG.valid_id:
                    valid_images.append(image[y1:y2, x1:x2])
                    valid_masks.append(mask[y1:y2, x1:x2, None]) # None add 1 dimension ==> (256, 256, 1)

                    valid_xyxys.append([x1, y1, x2, y2])

                    valid_masks_frag.append(mask_frag[y1:y2, x1:x2, None])
                else:

                    tmp = mask_frag[y1:y2, x1:x2] # (256, 256)
                    if tmp.min() == 1: # only choose areas with label 1
                        train_images.append(image[y1:y2, x1:x2])
                        train_masks.append(mask[y1:y2, x1:x2, None])

    return train_images, train_masks, valid_images, valid_masks, valid_xyxys, valid_masks_frag

train_images, train_masks, valid_images, valid_masks, valid_xyxys, valid_masks_frag = get_train_valid_dataset()
# train_images: 15068, (256, 256, 22)
# train_masks: 15068, (256, 256, 1)
# valid_images/masks: 6862, (256, 256, 1)
# valid_id 1: (6330, 8181)
# valid_xyxys: len 6862 [[0, 0, 256, 256], [85, 0, 341, 256], [170, 0, 426, 256], ,....,[6120, 7905, 6376, 8161]]
# (x1, y1, x2, y2) image[y1:y2, x1:x2] stride < image tile size, so valid tiles有重叠 
valid_xyxys = np.stack(valid_xyxys)


import numpy as np
from torch.utils.data import DataLoader, Dataset
import cv2
import torch
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform


# In[18]:


def get_transforms(data, cfg):
    if data == 'train':
        aug = A.Compose(cfg.train_aug_list)
    elif data == 'valid':
        aug = A.Compose(cfg.valid_aug_list)

    # print(aug)
    return aug

class CustomDataset(Dataset):
    def __init__(self, images, cfg, labels=None, mask_frag=None, transform=None, mode=None):
        self.images = images
        self.cfg = cfg
        self.labels = labels
        self.mask_frag = mask_frag
        self.transform = transform
        self.mode = mode

    def __len__(self):
        # return len(self.df)
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = np.clip(image, args.clip_min, args.clip_max) # min 50, max 200 


        label = self.labels[idx]

        if self.mask_frag:
            mask_frag = self.mask_frag[idx]

        image_tmp = np.zeros_like(image)

        if self.transform:

            if self.mode == 'train':

                # cropping_num = CFG.cropping_num
                cropping_num = random.randint(args.cropping_num_min, args.cropping_num_max)

                start_idx = random.randint(0, args.in_chans - cropping_num)
                crop_indices = np.arange(start_idx, start_idx + cropping_num)

                start_paste_idx = random.randint(0, args.in_chans - cropping_num)

                tmp = np.arange(start_paste_idx, cropping_num)
                np.random.shuffle(tmp)

                cutout_idx = random.randint(0, 2)
                temporal_random_cutout_idx = tmp[:cutout_idx]

                image_tmp[..., start_paste_idx : start_paste_idx + cropping_num] = image[..., crop_indices]

                if random.random() > 0.4:
                    image_tmp[..., temporal_random_cutout_idx] = 0
                image = image_tmp


        data = self.transform(image=image, mask=label)

        image = data['image'].unsqueeze(0)
        label = data['mask']

        if self.mode == 'train':
            return image, label
        else:
            return image, label, mask_frag


train_dataset = CustomDataset(
    train_images, CFG, labels=train_masks, mask_frag=None, transform=get_transforms(data='train', cfg=CFG), mode='train')
valid_dataset = CustomDataset(
    valid_images, CFG, labels=valid_masks, mask_frag=valid_masks_frag, transform=get_transforms(data='valid', cfg=CFG), mode='valid')

train_loader = DataLoader(train_dataset,
                          batch_size=CFG.train_batch_size,
                          shuffle=True,
                          num_workers=CFG.num_workers, drop_last=True,
                          )
valid_loader = DataLoader(valid_dataset,
                          batch_size=CFG.valid_batch_size,
                          shuffle=False,
                          num_workers=CFG.num_workers, drop_last=False)





class CustomModel(nn.Module):
    def __init__(self, cfg, weight=None):
        super().__init__()
        self.cfg = cfg

        if args.model_name == 'Unet':

            self.encoder = smp.Unet(
                encoder_name=cfg.backbone,
                encoder_weights=weight,
                in_channels=cfg.in_chans,
                classes=cfg.target_size,
                activation=None,
            )

        if args.model_name == 'UnetPlusPlus':

            self.encoder = smp.UnetPlusPlus(
                encoder_name=cfg.backbone,
                encoder_weights=weight,
                in_channels=cfg.in_chans,
                classes=cfg.target_size,
                activation=None,
            )

    def forward(self, image):
        output = self.encoder(image)
        # output = output.squeeze(-1)
        return output


def build_model(cfg, weight="imagenet"):
    print('model_name', cfg.model_name)
    print('backbone', cfg.backbone)

    model = CustomModel(cfg, weight)

    return model


class Decoder(nn.Module):
    def __init__(self, encoder_dims, upscale):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(encoder_dims[i]+encoder_dims[i-1], encoder_dims[i-1], 3, 1, 1, bias=False),
                nn.BatchNorm2d(encoder_dims[i-1]),
                nn.ReLU(inplace=True)
            ) for i in range(1, len(encoder_dims))])

        self.logit = nn.Conv2d(encoder_dims[0], 2, 1, 1, 0)
        self.up = nn.Upsample(scale_factor=upscale, mode="bilinear")
    def forward(self, feature_maps):
        for i in range(len(feature_maps)-1, 0, -1):
            f_up = F.interpolate(feature_maps[i], scale_factor=2, mode="bilinear")
            f = torch.cat([feature_maps[i-1], f_up], dim=1)
            f_down = self.convs[i-1](f)
            feature_maps[i-1] = f_down

        x = self.logit(feature_maps[0])
        mask = self.up(x)
        return mask


class SegModel(nn.Module):
    def __init__(self):
        super().__init__()
        # original kaggle code
        # self.encoder = generate_model(model_depth=args.resnet_depth, n_input_channels=1)

        # original paper code
        self.encoder = generate_model(model_depth=args.resnet_depth,
                                      n_input_channels=1,
                                      shortcut_type='B',
                                      conv1_t_size=7,
                                      conv1_t_stride=1,
                                      widen_factor=1.0,
                                      n_classes=1039,
                                      no_max_pool=True)

        # original kaggle code
        # self.decoder = Decoder(encoder_dims=[64, 128, 256, 512], upscale=4)

        # original paper code
        self.decoder = Decoder(encoder_dims=[256, 512, 1024, 2048], upscale=4)


    def forward(self, x):
        feat_maps = self.encoder(x)
        feat_maps_pooled = [torch.mean(f, dim=2) for f in feat_maps]
        pred_mask = self.decoder(feat_maps_pooled)
        return pred_mask
    
    def load_pretrained_weights(self, state_dict):
        # Convert 3 channel weights to single channel
        # ref - https://timm.fast.ai/models#Case-1:-When-the-number-of-input-channels-is-1
        conv1_weight = state_dict['conv1.weight']
        state_dict['conv1.weight'] = conv1_weight.sum(dim=1, keepdim=True)
        print(self.encoder.load_state_dict(state_dict, strict=False))



# ## scheduler

# In[24]:


import torch.nn as nn
import torch
import math
import time
import numpy as np
import torch

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau



import math
from torch.optim.lr_scheduler import _LRScheduler
class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr) * self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (
                        1 + math.cos(math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.eta_max = self.base_eta_max * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


            
            
            


from models.resnext import resnext101

class SegModel_resnext101(nn.Module):
    def __init__(self):
        super().__init__()
        # original kaggle code
        # self.encoder = generate_model(model_depth=args.resnet_depth, n_input_channels=1)

        # original paper code
        self.encoder = resnext101(sample_size=112,
                                  sample_duration=16,
                                  shortcut_type='B',
                                  cardinality=32,
                                  num_classes=600)

        # original paper code
        self.decoder = Decoder(encoder_dims=[256, 512, 1024, 2048], upscale=4)

    def forward(self, x):
        feat_maps = self.encoder(x)
        feat_maps_pooled = [torch.mean(f, dim=2) for f in feat_maps]
        pred_mask = self.decoder(feat_maps_pooled)
        return pred_mask

            
            
        
if args.model == 'resnet152':          

    model = SegModel()
    model.load_pretrained_weights(torch.load('/root/autodl-tmp/VCInkDectection/input/weights/r3d152_KM_200ep.pth')["state_dict"])
    model = model.to(device)
    
    
elif args.model == 'resnet200':          

    model = SegModel()
    model.load_pretrained_weights(torch.load('/root/autodl-tmp/VCInkDectection/input/weights/r3d200_KM_200ep.pth')["state_dict"])
    model = model.to(device)
    
    
elif args.model == 'resnext101':          

    model = SegModel_resnext101()

    checkpoint = '/root/autodl-tmp/VCInkDectection/input/weights/kinetics_resnext_101_RGB_16_best.pth'
    state_dict = torch.load(checkpoint)['state_dict']

    from collections import OrderedDict
    checkpoint_custom = OrderedDict()
    for key_model, key_checkpoint in zip(model.encoder.state_dict().keys(), state_dict.keys()):
        checkpoint_custom.update({f'{key_model}' : state_dict[f'{key_checkpoint}']})

    model.encoder.load_state_dict(checkpoint_custom, strict=True)
    model.encoder.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
    model = model.to(device)

    
    

optimizer = AdamW(model.parameters(), lr=0)
scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=9, T_mult=1, eta_max=1e-4,  T_up=3, gamma=0.5)


# DiceLoss = smp.losses.DiceLoss(mode='binary')
# BCELoss = smp.losses.SoftBCEWithLogitsLoss()
CELoss = nn.CrossEntropyLoss(label_smoothing=args.ls)

# alpha = 0.5
# beta = 1 - alpha
# TverskyLoss = smp.losses.TverskyLoss(
#     mode='binary', log_loss=False, alpha=alpha, beta=beta)


def fbeta_loss(preds, targets, beta=0.5, smooth=1e-5):
    """
    https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/397288
    """
    
    preds = torch.sigmoid(preds)

    y_true_count = targets.sum()
    ctp = preds[targets==1].sum()
    cfp = preds[targets==0].sum()
    beta_squared = beta * beta

    c_precision = ctp / (ctp + cfp + smooth)
    c_recall = ctp / (y_true_count + smooth)
    dice = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall + smooth)

    return (1 - dice)**(args.fbeta_gamma)



def focal_tversky_loss(preds, targets, alpha=0.7, beta=0.3, epsilon=1e-6, gamma=1):

    preds = torch.sigmoid(preds)

    preds = preds.reshape(-1)
    targets = targets.reshape(-1)

    TP = (preds * targets).sum()
    FP = ((1-targets) * preds).sum()
    FN = (targets * (1-preds)).sum()
    Tversky = (TP + epsilon) / (TP + alpha*FP + beta*FN + epsilon)
    FocalTversky = (1 - Tversky)**gamma

    return FocalTversky



def criterion(y_pred, y_true):
    return CELoss(y_pred, y_true)



def rand_bbox(size, lam):
    W = size[-2]
    H = size[-1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    #uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

cutmix = True
beta = 1

def train_fn(train_loader, model, criterion, optimizer, device):
    model.train()

    scaler = GradScaler(enabled=CFG.use_amp)
    losses = AverageMeter()

    for step, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        images = images.to(device)
        labels = labels[:, 0, ...].long().to(device)
        batch_size = labels.size(0)
            
        if cutmix and random.random() > 0.4:
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(images.size()[0]).cuda()
            bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)

            images[:, :, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, :, bbx1:bbx2, bby1:bby2]
            labels[:, bbx1:bbx2, bby1:bby2] = labels[rand_index, bbx1:bbx2, bby1:bby2]



        with autocast(CFG.use_amp):
            y_preds = model(images)
            loss = criterion(y_preds, labels)

        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), CFG.max_grad_norm)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return losses.avg

def valid_fn(valid_loader, model, criterion, device, valid_xyxys, valid_mask_gt):
    mask_pred = np.zeros(valid_mask_gt.shape)
    mask_count = np.zeros(valid_mask_gt.shape)

    model.eval()
    losses = AverageMeter()

    for step, (images, labels, mask_frag) in tqdm(enumerate(valid_loader), total=len(valid_loader)):

        if mask_frag.max() == 0:
            continue

        images = images.to(device)
        labels = labels[:, 0, ...].long().to(device)
        batch_size = labels.size(0)

        with torch.no_grad():
            y_preds = model(images)
            loss = criterion(y_preds, labels)
        losses.update(loss.item(), batch_size)

        # make whole mask
        y_preds = torch.softmax(y_preds, 1)[:, 1, ...].to('cpu').numpy()
        start_idx = step*CFG.valid_batch_size
        end_idx = start_idx + batch_size
        for i, (x1, y1, x2, y2) in enumerate(valid_xyxys[start_idx:end_idx]):
            mask_pred[y1:y2, x1:x2] += y_preds[i]#.squeeze(0)
            mask_count[y1:y2, x1:x2] += np.ones((CFG.tile_size, CFG.tile_size))

            
            
    # print(f'mask_count_min: {mask_count.min()}')
    mask_pred /= mask_count
    return losses.avg, mask_pred




from sklearn.metrics import fbeta_score

def fbeta_numpy(targets, preds, beta=0.5, smooth=1e-5):
    """
    https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/397288
    """
    y_true_count = targets.sum()
    ctp = preds[targets==1].sum()
    cfp = preds[targets==0].sum()
    beta_squared = beta * beta

    c_precision = ctp / (ctp + cfp + smooth)
    c_recall = ctp / (y_true_count + smooth)
    dice = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall + smooth)

    return dice, ctp, cfp, (y_true_count - ctp)




def calc_fbeta(mask, mask_pred):
    mask = mask.astype(int).flatten()
    mask_pred = mask_pred.flatten()

    best_th = 0
    best_dice = 0
    for th in np.array(range(10, 95+1, 5)) / 100:
        
        # dice = fbeta_score(mask, (mask_pred >= th).astype(int), beta=0.5)
        dice, ctp, cfp, cfn = fbeta_numpy(mask, (mask_pred >= th).astype(int), beta=0.5)
        Logger.info(f'th: {th}, fbeta: {dice}')

        if dice > best_dice:
            best_dice = dice
            best_th = th

    dice, ctp, cfp, cfn = fbeta_numpy(mask, (mask_pred >= 0.5).astype(int), beta=0.5)
    Logger.info(f'best_th: {best_th}, fbeta: {best_dice}')
    return dice, 0.5, ctp, cfp, cfn


def calc_cv(mask_gt, mask_pred):
    best_dice, best_th, ctp, cfp, cfn = calc_fbeta(mask_gt, mask_pred)

    return best_dice, best_th, ctp, cfp, cfn



fragment_id = CFG.valid_id

if fragment_id == 2:
    valid_mask_gt = cv2.imread(CFG.comp_dataset_path + f"train/2/inklabels.png", 0)
    valid_mask_gt = valid_mask_gt[:, :args.slicing_num]

elif fragment_id == 4:
    valid_mask_gt = cv2.imread(CFG.comp_dataset_path + f"train/2/inklabels.png", 0)
    valid_mask_gt = valid_mask_gt[:, args.slicing_num:]

else:
    valid_mask_gt = cv2.imread(CFG.comp_dataset_path + f"train/{fragment_id}/inklabels.png", 0)

valid_mask_gt = valid_mask_gt / 255
pad0 = (CFG.tile_size - valid_mask_gt.shape[0] % CFG.tile_size)
pad1 = (CFG.tile_size - valid_mask_gt.shape[1] % CFG.tile_size)
valid_mask_gt = np.pad(valid_mask_gt, [(0, pad0), (0, pad1)], constant_values=0)




if fragment_id == 2:
    valid_mask_gt_frag = cv2.imread(CFG.comp_dataset_path + f"train/2/mask.png", 0)
    valid_mask_gt_frag = valid_mask_gt_frag[:, :args.slicing_num]

elif fragment_id == 4:
    valid_mask_gt_frag = cv2.imread(CFG.comp_dataset_path + f"train/2/mask.png", 0)
    valid_mask_gt_frag = valid_mask_gt_frag[:, args.slicing_num:]

else:
    valid_mask_gt_frag = cv2.imread(CFG.comp_dataset_path + f"train/{fragment_id}/mask.png", 0)

valid_mask_gt_frag = valid_mask_gt_frag / 255
pad0 = (CFG.tile_size - valid_mask_gt_frag.shape[0] % CFG.tile_size)
pad1 = (CFG.tile_size - valid_mask_gt_frag.shape[1] % CFG.tile_size)
valid_mask_gt_frag = np.pad(valid_mask_gt_frag, [(0, pad0), (0, pad1)], constant_values=0)
valid_mask_gt_frag = valid_mask_gt_frag.astype('bool')



fold = CFG.valid_id

if CFG.metric_direction == 'minimize':
    best_score = np.inf
elif CFG.metric_direction == 'maximize':
    best_score = -1

best_loss = np.inf

for epoch in range(CFG.epochs):

    scheduler.step()
    Logger.info(f'Lr : {scheduler.get_lr()}')


    start_time = time.time()

    # train
    avg_loss = train_fn(train_loader, model, criterion, optimizer, device)
    Logger.info(f'Train Loss : {avg_loss}')


    if epoch >= 20:
        # eval
        avg_val_loss, mask_pred = valid_fn(
            valid_loader, model, criterion, device, valid_xyxys, valid_mask_gt)

        mask_pred = mask_pred * valid_mask_gt_frag


        best_dice, best_th, ctp, cfp, cfn = calc_cv(valid_mask_gt, mask_pred)

        # score = avg_val_loss
        score = best_dice

        elapsed = time.time() - start_time

        Logger.info(
            f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        # Logger.info(f'Epoch {epoch+1} - avgScore: {avg_score:.4f}')
        Logger.info(
            f'Epoch {epoch+1} - avgScore: {score:.4f}')

        if CFG.metric_direction == 'minimize':
            update_best = score < best_score
        elif CFG.metric_direction == 'maximize':
            update_best = score > best_score

        if update_best:
            best_loss = avg_val_loss
            best_score = score

            Logger.info(
                f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            Logger.info(
                f'Epoch {epoch+1} - Save Best Loss: {best_loss:.4f} Model')

        torch.save({'model': model.state_dict()},
                    CFG.model_dir + f'{CFG.model_name}_fold{fold}_epoch{epoch+1}_score{score}.pth')


        mask_pred_png = (mask_pred > 0.5).astype('uint8')*255
        cv2.imwrite(f'{CFG.mask_dir}/Fold{args.valid_id}_epoch{epoch+1}_score{score}_ctp{ctp}_cfp{cfp}_cfn{cfn}.png', mask_pred_png)

        Logger.info('\n')


