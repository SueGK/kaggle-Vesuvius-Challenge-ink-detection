
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, log_loss
import pickle
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import warnings
import sys
import pandas as pd
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
import pandas as pd

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from functools import partial

import argparse
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW

import datetime

import segmentation_models_pytorch as smp
from einops import rearrange, reduce, repeat
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder, DecoderBlock
from timm.models.resnet import *

import numpy as np
from torch.utils.data import DataLoader, Dataset
import cv2
import torch
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform


class CFG:
    # ============== comp exp name =============
    comp_name = 'vesuvius'

    # comp_dir_path = './'
    comp_dir_path = '/root/autodl-tmp/VCInkDectection/input/'
    comp_folder_name = 'vesuvius-challenge-ink-detection'
    # comp_dataset_path = f'{comp_dir_path}datasets/{comp_folder_name}/'
    comp_dataset_path = f'{comp_dir_path}{comp_folder_name}/'
    
    exp_name = 'vesuvius_3d_slice_resume2'

    # ============== pred target =============
    target_size = 1

    # ============== model cfg =============
    model_name = 'Unet'
    backbone = 'efficientnet-b0'
    # backbone = 'se_resnext50_32x4d'
    mode = 'train'
    crop_fade  = 32 # 32
    crop_size = 256
    crop_depth = 5 #5, 12
    in_chans = 16
    infer_fragment_z = [24, 40]
    train_fragment_depth = 11
    infer_fragment_depth = 9
    # ============== training cfg =============
    size = 256
    tile_size = 256
    #stride = tile_size // 2

    train_batch_size = 16 # 32
    valid_batch_size = train_batch_size * 2
    use_amp = True

    scheduler = 'GradualWarmupSchedulerV2'
    # scheduler = 'CosineAnnealingLR'
    epochs = 35

    # adamW warmupあり
    warmup_factor = 10
    # lr = 1e-4 / warmup_factor
    lr = 1e-4 
    is_ckpt = True
    # ============== fold =============
    valid_id = 1

    # objective_cv = 'binary'  # 'binary', 'multiclass', 'regression'
    metric_direction = 'maximize'  # maximize, 'minimize'
    # metrics = 'dice_coef'

    # ============== fixed =============
    pretrained = True
    inf_weight = 'best'  # 'best'

    min_lr = 1e-6
    weight_decay = 1e-6
    max_grad_norm = 1000

    print_freq = 50
    num_workers = 4

    seed = 42

    # ============== set dataset path =============
    print('set dataset path')

    outputs_path = f'/root/autodl-tmp/VCInkDectection/working/outputs/{comp_name}/{exp_name}/'

    submission_dir = outputs_path + 'submissions/'
    submission_path = submission_dir + f'submission_{exp_name}.csv'

    model_dir = outputs_path + \
        f'{comp_name}-models/'

    figures_dir = outputs_path + 'figures/'

    log_dir = outputs_path + 'logs/'
    log_path = log_dir + f'{exp_name}.txt'

    # ============== augmentation =============
    train_aug_list = [
        # A.RandomResizedCrop(
        #     size, size, scale=(0.85, 1.0)),
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.7),
        A.RandomBrightnessContrast(p=0.75),
        A.ShiftScaleRotate(p=0.75),
        A.OneOf([
                A.GaussNoise(var_limit=[10, 50]),
                A.GaussianBlur(),
                A.MotionBlur(),
                ], p=0.4),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        A.CoarseDropout(max_holes=1, max_width=int(size * 0.3), max_height=int(size * 0.3), 
                        mask_fill_value=0, p=0.5),
        # A.Cutout(max_h_size=int(size * 0.6),
        #          max_w_size=int(size * 0.6), num_holes=1, p=1.0),
        A.Normalize(
            mean= [0] * in_chans,
            std= [1] * in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]

    valid_aug_list = [
        A.Resize(size, size),
        A.Normalize(
            mean= [0] * in_chans,
            std= [1] * in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]


parser = argparse.ArgumentParser()

parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
    help='batch_size'
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
    default=10,
    help='balance in mask pixel for 4Fold Training'
    )

parser.add_argument(
    "--cropping_num_max",
    type=int,
    default=16,
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

def make_dirs(cfg):
    for dir in [cfg.model_dir, cfg.figures_dir, cfg.submission_dir, cfg.log_dir]:
        os.makedirs(dir, exist_ok=True)
        
def cfg_init(cfg, mode='train'):
    set_seed(cfg.seed)
    # set_env_name()
    # set_dataset_path(cfg)

    if mode == 'train':
        make_dirs(cfg)
        
def metric_to_text(ink, label, mask):
	text = []

	p = ink.reshape(-1)
	t = label.reshape(-1)
	pos = np.log(np.clip(p,1e-7,1))
	neg = np.log(np.clip(1-p,1e-7,1))
	bce = -(t*pos +(1-t)*neg).mean()
	text.append(f'bce={bce:0.5f}')


	mask_sum = mask.sum()
	#print(f'{threshold:0.1f}, {precision:0.3f}, {recall:0.3f}, {fpr:0.3f},  {dice:0.3f},  {score:0.3f}')
	text.append('p_sum  th   prec   recall   fpr   dice   score')
	text.append('-----------------------------------------------')
	for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
		p = ink.reshape(-1)
		t = label.reshape(-1)
		p = (p > threshold).astype(np.float32)
		t = (t > 0.5).astype(np.float32)

		tp = p * t
		precision = tp.sum() / (p.sum() + 0.0001)
		recall = tp.sum() / t.sum()

		fp = p * (1 - t)
		fpr = fp.sum() / (1 - t).sum()

		beta = 0.5
		#  0.2*1/recall + 0.8*1/prec
		score = beta * beta / (1 + beta * beta)  / (recall+0.001) + 1 / (1 + beta * beta)  / (precision+0.001)
		score = 1 / score

		dice = 2 * tp.sum() / (p.sum() + t.sum())
		p_sum = p.sum()/mask_sum

		# print(fold, threshold, precision, recall, fpr,  score)
		text.append( f'{p_sum:0.2f}, {threshold:0.2f}, {precision:0.3f}, {recall:0.3f}, {fpr:0.3f},  {dice:0.3f},  {score:0.3f}')
	text = '\n'.join(text)
	return text

def make_infer_mask():
	s = CFG.crop_size
	f = CFG.crop_fade
	x = np.linspace(-1, 1, s)
	y = np.linspace(-1, 1, s)
	xx, yy = np.meshgrid(x, y)
	d = 1 - np.maximum(np.abs(xx), np.abs(yy))
	d1 = np.clip(d, 0, f / s * 2)
	d1 = d1 / d1.max()
	infer_mask = d1
	return infer_mask

cfg_init(CFG)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

Logger = init_logger(log_file=CFG.log_path)

Logger.info('\n\n-------- exp_info -----------------')
# Logger.info(datetime.datetime.now().strftime('%Y年%m月%d日 %H:%M:%S'))

if 'train' in CFG.mode:  #todo: try stride different for train/test
	CFG.stride = CFG.crop_size//4 # 2
if 'test' in CFG.mode:
	CFG.stride = CFG.crop_size//2 # 8
    
def read_image_mask(fragment_id):

    images = []

    # idxs = range(65)
    mid = 65 // 2
    start = mid - CFG.in_chans // 2
    end = mid + CFG.in_chans // 2
    idxs = range(start, end)

    for i in tqdm(idxs):
        
        image = cv2.imread(CFG.comp_dataset_path + f"train/{fragment_id}/surface_volume/{i:02}.tif", 0)

        pad0 = (CFG.tile_size - image.shape[0] % CFG.tile_size)
        pad1 = (CFG.tile_size - image.shape[1] % CFG.tile_size)

        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)

        images.append(image)
    images = np.stack(images, axis=2)

    mask = cv2.imread(CFG.comp_dataset_path + f"train/{fragment_id}/inklabels.png", 0)
    mask = np.pad(mask, [(0, pad0), (0, pad1)], constant_values=0)

    mask = mask.astype('float32')
    mask /= 255.0
    
    mask_prag = cv2.imread(CFG.comp_dataset_path + f"train/{fragment_id}/mask.png", 0)
    mask_prag = np.pad(mask_prag, [(0, pad0), (0, pad1)], constant_values=0)

    mask_prag = mask_prag.astype('float32')
    mask_prag /= 255.0
    
    return images, mask, mask_prag # (14848, 9728, 16)

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
            image, mask, mask_frag = image[:9456], mask[:9456], mask_frag[:9456]
            # image:(14848, 4300, 22), mask(inklabels): (14848, 4300), mask_frag(mask): (14848, 4300)
        elif fragment_id == 4:
            image, mask, mask_frag = read_image_mask(2)
            image, mask, mask_frag = image[9456:], mask[9456:], mask_frag[9456:]
             
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
                    if tmp.sum()==0:
                        continue # only choose areas with label 1
                    train_images.append(image[y1:y2, x1:x2])
                    train_masks.append(mask[y1:y2, x1:x2, None])

    return train_images, train_masks, valid_images, valid_masks, valid_xyxys, valid_masks_frag



train_images, train_masks, valid_images, valid_masks, valid_xyxys, valid_masks_frag = get_train_valid_dataset()
valid_xyxys = np.stack(valid_xyxys)

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


#         if self.mode == 'train':
            
#             # cropping_num = CFG.cropping_num
#             cropping_num = random.randint(args.cropping_num_min, args.cropping_num_max)

#             start_idx = random.randint(0, CFG.in_chans - cropping_num)
#             crop_indices = np.arange(start_idx, start_idx + cropping_num)

#             start_paste_idx = random.randint(0, CFG.in_chans - cropping_num)

#             tmp = np.arange(start_paste_idx, cropping_num)
#             np.random.shuffle(tmp)

#             cutout_idx = random.randint(0, 2)
#             temporal_random_cutout_idx = tmp[:cutout_idx]

#             image_tmp[..., start_paste_idx : start_paste_idx + cropping_num] = image[..., crop_indices]

#             if random.random() > 0.4:
#                 image_tmp[..., temporal_random_cutout_idx] = 0
#             image = image_tmp
            
        if self.transform:
                data = self.transform(image=image, mask=label)
                image = data['image']
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
                          num_workers=CFG.num_workers, pin_memory=True, drop_last=True,
                          )
valid_loader = DataLoader(valid_dataset,
                          batch_size=CFG.valid_batch_size,
                          shuffle=False,
                          num_workers=CFG.num_workers, pin_memory=True, drop_last=False)

#======> model <=========
class SmpUnetDecoder(nn.Module):
	def __init__(self,
	             in_channel,
	             skip_channel,
	             out_channel,
	             ):
		super().__init__()
		self.center = nn.Identity()

		i_channel = [in_channel, ] + out_channel[:-1]
		s_channel = skip_channel
		o_channel = out_channel
		block = [
			DecoderBlock(i, s, o, use_batchnorm=True, attention_type=None)
			for i, s, o in zip(i_channel, s_channel, o_channel)
		]
		self.block = nn.ModuleList(block)

	def forward(self, feature, skip):
		d = self.center(feature)
		decode = []
		for i, block in enumerate(self.block):
			s = skip[i]
			d = block(d, s)
			decode.append(d)

		last = d
		return last, decode


#######################################################################################

class Net(nn.Module):
	def __init__(self, ):
		super().__init__()
		self.output_type = ['inference', 'loss']

		# --------------------------------
		self.d = 5

		conv_dim = 64
		encoder_dim = [conv_dim, 64, 128, 256, 512, ]
		decoder_dim = [256, 128, 64, 32, 16]

		self.encoder = resnet34d(pretrained=False, in_chans=self.d)

		self.decoder = SmpUnetDecoder(
			in_channel=encoder_dim[-1],
			skip_channel=encoder_dim[:-1][::-1] + [0],
			out_channel=decoder_dim,
		)
		self.logit = nn.Conv2d(decoder_dim[-1], 1, kernel_size=1)



	def forward(self, images):
		v = images
		B, C, H, W = v.shape
		vv = [
			v[:, i:i + self.d] for i in range(0,C-self.d+1,2)
		]
		K = len(vv)
		x = torch.cat(vv, 0)

		# ---------------------------------

		encoder = []
		e = self.encoder

		x = e.conv1(x)
		x = e.bn1(x)
		x = e.act1(x); encoder.append(x)
		x = F.avg_pool2d(x, kernel_size=2, stride=2)
		x = e.layer1(x); encoder.append(x)
		x = e.layer2(x); encoder.append(x)
		x = e.layer3(x); encoder.append(x)
		x = e.layer4(x); encoder.append(x)
		##[print('encoder',i,f.shape) for i,f in enumerate(encoder)]

		for i in range(len(encoder)):
			e = encoder[i]
			_, c, h, w = e.shape
			e = rearrange(e, '(K B) c h w -> K B c h w', K=K, B=B, h=h, w=w)
			encoder[i] = e.mean(0)

		last, decoder = self.decoder(feature = encoder[-1], skip = encoder[:-1][::-1]  + [None])


		# ---------------------------------
		logit = self.logit(last)

		output = {}
		if 1:
			if logit.shape[2:]!=(H, W):
				logit = F.interpolate(logit, size=(H, W), mode='bilinear', align_corners=False, antialias=True)
			output['ink'] = logit

		return output




from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from warmup_scheduler import GradualWarmupScheduler


class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    """
    https://www.kaggle.com/code/underwearfitting/single-fold-training-of-resnet200d-lb0-965
    """
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(
            optimizer, multiplier, total_epoch, after_scheduler)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

def get_scheduler(cfg, optimizer):
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg.epochs, eta_min=1e-6)
    scheduler = GradualWarmupSchedulerV2(
        optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)

    return scheduler

def scheduler_step(scheduler, avg_val_loss, epoch):
    scheduler.step(epoch)





BCE = torch.nn.BCEWithLogitsLoss()

if CFG.is_ckpt == True:  
    model = Net().cuda()
    optimizer = AdamW(model.parameters(), lr=CFG.lr)
    # Load the checkpoint file
    checkpoint = torch.load('/root/autodl-tmp/VCInkDectection/working/outputs/vesuvius/vesuvius_3d_slice_resume2/vesuvius-models/Unet_fold1_slice16_crop_depth5_best.pth')
    
    # Load the state dictionary into the model
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler = get_scheduler(CFG, optimizer)
    
else:  
    model = Net().cuda()
    optimizer = AdamW(model.parameters(), lr=CFG.lr)
    scheduler = get_scheduler(CFG, optimizer)

def criterion(y_pred, y_true):
    # return 0.5 * BCELoss(y_pred, y_true) + 0.5 * DiceLoss(y_pred, y_true)
    return BCE(y_pred, y_true)
    # return 0.5 * BCELoss(y_pred, y_true) + 0.5 * TverskyLoss(y_pred, y_true)
    
def train_fn(train_loader, model, criterion, optimizer, device):
    model.train()
    gc.collect()
    torch.cuda.empty_cache()
    scaler = GradScaler(enabled=CFG.use_amp)
    losses = AverageMeter()

    for step, (images, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        images = images.float().cuda()
        labels = labels.cuda()
        batch_size = labels.size(0)
        #new_labels = F.interpolate(labels, size=(CFG.stride, CFG.stride), mode='bilinear', align_corners=False, antialias=True)
        
        with autocast(CFG.use_amp):
            output = model(images)
            #loss = criterion(y_preds, labels)
            logit = output['ink']
            loss = criterion(logit, labels)

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

    for step, (images, labels, masks) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
        images = images.float().to(device) # ([32, 16, 256, 256])
        labels = labels.float().to(device) # ([32, 1, 256, 256])
        masks = masks.permute((0,3,1,2))                      # ([32, 1, 256, 256])
        batch_size = labels.size(0)

        with torch.no_grad():
            output = model(images)
            y_preds = output['ink']
            loss = criterion(y_preds, labels)
        losses.update(loss.item(), batch_size)

        # make whole mask
        y_preds = torch.sigmoid(y_preds).to('cpu').numpy()
        start_idx = step*CFG.valid_batch_size
        end_idx = start_idx + batch_size
        for i, (x1, y1, x2, y2) in enumerate(valid_xyxys[start_idx:end_idx]):
            mask_pred[y1:y2, x1:x2] += y_preds[i].squeeze(0)
            mask_count[y1:y2, x1:x2] += np.ones((CFG.tile_size, CFG.tile_size))

    print(f'mask_count_min: {mask_count.min()}')
    mask_pred /= mask_count
    mask_pred = mask_pred
    return losses.avg, mask_pred, masks.detach().numpy(), labels.cpu().numpy()
 #                   (8192, 6400) ([15, 1, 256, 256]))


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

    return dice

def calc_fbeta(mask, mask_pred):
    mask = mask.astype(int).flatten()
    mask_pred = mask_pred.flatten()

    best_th = 0
    best_dice = 0
    for th in np.array(range(40, 65+1, 5)) / 100:
        
        # dice = fbeta_score(mask, (mask_pred >= th).astype(int), beta=0.5)
        dice = fbeta_numpy(mask, (mask_pred >= th).astype(int), beta=0.5)
        print(f'th: {th}, fbeta: {dice}')

        if dice > best_dice:
            best_dice = dice
            best_th = th
    
    Logger.info(f'best_th: {best_th}, fbeta: {best_dice}')
    return best_dice, best_th


def calc_cv(mask_gt, mask_pred):
    best_dice, best_th = calc_fbeta(mask_gt, mask_pred)

    return best_dice, best_th


fragment_id = CFG.valid_id

valid_mask_gt = cv2.imread(CFG.comp_dataset_path + f"train/{fragment_id}/inklabels.png", 0)
valid_mask_gt = valid_mask_gt / 255
pad0 = (CFG.tile_size - valid_mask_gt.shape[0] % CFG.tile_size)
pad1 = (CFG.tile_size - valid_mask_gt.shape[1] % CFG.tile_size)
valid_mask_gt = np.pad(valid_mask_gt, [(0, pad0), (0, pad1)], constant_values=0)

valid_mask1_gt = cv2.imread(CFG.comp_dataset_path + f"train/{fragment_id}/mask.png", 0)
valid_mask1_gt = valid_mask1_gt / 255
pad0 = (CFG.tile_size - valid_mask1_gt.shape[0] % CFG.tile_size)
pad1 = (CFG.tile_size - valid_mask1_gt.shape[1] % CFG.tile_size)
valid_mask1_gt = np.pad(valid_mask1_gt, [(0, pad0), (0, pad1)], constant_values=0)


fold = CFG.valid_id

if CFG.metric_direction == 'minimize':
    best_score = np.inf
elif CFG.metric_direction == 'maximize':
    best_score = -1

best_loss = np.inf

for epoch in range(CFG.epochs):

    start_time = time.time()

    # train
    avg_loss = train_fn(train_loader, model, criterion, optimizer, device)

    # eval
    avg_val_loss, mask_pred, masks, labels = valid_fn(
        valid_loader, model, criterion, device, valid_xyxys, valid_mask_gt)

    scheduler_step(scheduler, avg_val_loss, epoch)
    
    text = metric_to_text(mask_pred, valid_mask_gt, valid_mask1_gt)
    Logger.info(text)
    best_dice, best_th = calc_cv(valid_mask_gt, mask_pred)

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
        lr = optimizer.param_groups[0]['lr']
        torch.save({'model': model.state_dict(),
                    #'preds': mask_pred,
                    'epoch': epoch + 1,
                    'optimizer': optimizer.state_dict(),
                    'lr': lr},
                    CFG.model_dir + f'{CFG.model_name}_fold{fold}_slice{CFG.in_chans}_crop_depth{CFG.crop_depth}_stride_{CFG.stride}_best.pth')