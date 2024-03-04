my_lib_dir ='/kaggle/input/ink-00/my_lib'
import pdb
import sys
import time
import tqdm
import gc

sys.path.append(my_lib_dir)
sys.path.append('/kaggle/input/pretrainedmodels/pretrainedmodels-0.7.4')
sys.path.append('/kaggle/input/efficientnet-pytorch/EfficientNet-PyTorch-master')
sys.path.append('/kaggle/input/timm-pytorch-image-models/pytorch-image-models-master')
sys.path.append('/kaggle/input/segmentation-models-pytorch/segmentation_models.pytorch-master')
sys.path.append('/kaggle/input/einops/einops-master')

from helper import *
import hashlib
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform
from collections import defaultdict
from glob import glob
import PIL.Image as Image
Image.MAX_IMAGE_PIXELS = 10000000000  # Ignore PIL warnings about large images

import cv2
import wandb
import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp

from einops import rearrange, reduce, repeat
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder, DecoderBlock
from timm.models.resnet import *


import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use('TkAgg')
#%matplotlib inline 

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from warmup_scheduler import GradualWarmupScheduler
  
print('import ok !!!')

class Config(object):
    #==============>> model <<=================
    mode = 'train' # 'test', 'skip_fake_test'
    crop_fade  = 32 # 32
    crop_size = 256
    crop_depth = 12
    in_chans = 16
    infer_fragment_z = [24, 40]
    train_fragment_depth = 11
    infer_fragment_depth = 9
    # stride = 192 / 56🔥

    #==============>> training cfg <<=================
    seed = 42  
    num_worker = 0 # debug => 0
    batch_size = 32 # 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # optimizer
    scheduler = 'GradualWarmupSchedulerV2'
    # scheduler = 'CosineAnnealingLR'
    epochs = 15
    lr = 1e-4 
    weight_decay = 4e-6
    lr_drop = 15
    labelSmoothing = 0.3

    # infer
    train_threshold = 0.5
    valid_threshold = 0.8

    is_tta = True
    
    # augmentation
    train_aug_list = [
        # A.RandomResizedCrop(
        #     size, size, scale=(0.85, 1.0)),
        A.Resize(crop_size, crop_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.75),
        A.RandomBrightnessContrast(p=0.5, brightness_limit=.2, contrast_limit=.2),
        A.ChannelDropout(channel_drop_range=(1,2), p=.25),
        A.ShiftScaleRotate(rotate_limit=180, p=0.1),
        A.OneOf([
                A.GaussNoise(var_limit=[10, 50]),
                A.GaussianBlur(),
                A.MotionBlur(),
                ], p=0.2),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.1),
        #A.CoarseDropout(max_holes=1, max_width=int(crop_size * 0.05), max_height=int(crop_size * 0.05), 
        #                mask_fill_value=0, p=0.1),
        # A.Cutout(max_h_size=int(size * 0.6),
        #          max_w_size=int(size * 0.6), num_holes=1, p=1.0),
        #A.Normalize(
        #    mean= [0] * in_chans,
        #    std= [1] * in_chans
        #),
        ToTensorV2(transpose_mask=True),
    ]

    valid_aug_list = [
        A.Resize(crop_size, crop_size),
        A.Normalize(
            mean= [0] * in_chans,
            std= [1] * in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]  

CFG = Config()


if 'train' in CFG.mode:  #todo: try stride different for train/test
	CFG.stride = CFG.crop_size//2 # 2
if 'test' in CFG.mode:
	CFG.stride = CFG.crop_size//2 # 8

def cfg_to_text():
    d = Config.__dict__
    text = [f'\t{k} : {v}' for k,v in d.items() if not (k.startswith('__') and k.endswith('__'))]
    d = CFG.__dict__
    text += [f'\t{k} : {v}' for k,v in d.items() if not (k.startswith('__') and k.endswith('__'))]
    return 'CFG\n'+'\n'.join(text)

print(cfg_to_text())

## dataset ##
if 'train' in CFG.mode:
    data_dir = '/root/autodl-tmp/VCInkDectection/input/vesuvius-challenge-ink-detection/train' #todo change
    train_id = ['3', '2a', '2b']
    valid_id =['1']

if 'test' in CFG.mode: 
	data_dir = '/root/autodl-tmp/VCInkDectection/input/vesuvius-challenge-ink-detection/test'
	valid_id = glob(f'{data_dir}/*')
	valid_id = sorted(valid_id)
	valid_id = [f.split('/')[-1] for f in valid_id]

    # https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/410985
	a_file = f'{data_dir}/a/mask.png'
	with open(a_file,'rb') as f:
		hash_md5 = hashlib.md5(f.read()).hexdigest()
	is_skip_test = hash_md5 == '0b0fffdc0e88be226673846a143bb3e0'
	print('is_skip_test:',is_skip_test)

print('data_dir', data_dir)
print('valid_id', valid_id)



def do_binarise(m, threshold=0.5): #binary mask
    m = m-m.min()
    m = m/(m.max()+1e-7)
    m = (m>threshold).astype(np.float32)
    return m

def read_data(fragment_id, z0, z1):
    volume = []
    start_timer = time.time()
    for i in range(z0,z1):
        v = np.array(Image.open(f'{data_dir}/{fragment_id}/surface_volume/{i:02d}.tif'), dtype=np.uint16)
        v = (v >> 8).astype(np.uint8) # v: max 65535 min 0 --> v: max 255 min 0
        #v = (v / 65535.0 * 255).astype(np.uint8)
        volume.append(v)
        print(f'\r @ read_data(): volume-{fragment_id}  {str(time.time() - start_timer)}', end='', flush=True)
    #print('')
    volume = np.stack(volume, -1) # (2727, 6330, 12)
    height, width, depth = volume.shape
    #print(f'fragment_id={fragment_id} volume: {volume.shape}')

    #---
    mask = cv2.imread(f'{data_dir}/{fragment_id}/mask.png',cv2.IMREAD_GRAYSCALE)
    mask = do_binarise(mask)

    if 'train' in CFG.mode:
        ir    = cv2.imread(f'{data_dir}/{fragment_id}/ir.png',cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(f'{data_dir}/{fragment_id}/inklabels.png',cv2.IMREAD_GRAYSCALE)
        ir    = ir/255
        label = do_binarise(label)

    if 'test' in CFG.mode:
        ir = None
        label = None

    d = dotdict(
        fragment_id = fragment_id,
        volume = volume,
        ir     = ir,
        label  = label,
        mask   = mask,
    )
    return d

def read_data1(fragment_id, z0, z1):
    if fragment_id=='2a':
        y = 9456
        d = read_data('2', z0, z1)
        d = dotdict(
            fragment_id=fragment_id,
            volume  = d.volume[:y],
            ir      = d.ir[:y],
            label   = d.label[:y],
            mask    = d.mask[:y],
        )
    elif  fragment_id=='2b':
        y = 9456
        d = read_data('2', z0, z1)
        d = dotdict(
            fragment_id=fragment_id,
            volume  = d.volume[y:],
            ir      = d.ir[y:],
            label   = d.label[y:],
            mask    = d.mask[y:],
        )
    elif  fragment_id=='2aa':
        y0,y1 = 0, 7074
        d = read_data('2', z0, z1)
        d = dotdict(
            fragment_id=fragment_id,
            volume  = d.volume[y0:y1],
            ir      = d.ir[y0:y1],
            label   = d.label[y0:y1],
            mask    = d.mask[y0:y1],
        )
    else:
        d = read_data(fragment_id, z0, z1)
    return d

def run_check_data():
    d=read_data1('1', z0=32-16, z1=32+16)#valid_id[0] # changed
    print('')
    print('fragment_id:', d.fragment_id)
    print('volume:', d.volume.shape, ' min:', d.volume.min(), ' max:', d.volume.max())
    print('mask  :', d.mask.shape, ' min:', d.mask.min(), ' max:', d.mask.max())
    if 'train' in CFG.mode:
        print('ir    :', d.ir.shape, ' min:', d.ir.min(), ' max:', d.ir.max())
        print('label :', d.label.shape, ' min:', d.label.min(), ' max:', d.label.max())

#run_check_data()
print('data ok !!!')





##====================> model <========================##
class SmpUnetDecoder(nn.Module):
	def __init__(self,
	         in_channel,
	         skip_channel,
	         out_channel,
	    ):
		super().__init__()
		self.center = nn.Identity()

		i_channel = [in_channel,]+ out_channel[:-1]
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

		last  = d
		return last, decode

class Net(nn.Module):
	def __init__(self,):
		super().__init__()
		self.output_type = ['inference', 'loss']

		conv_dim = 64
		encoder1_dim  = [conv_dim, 64, 128, 256, 512, ]
		decoder1_dim  = [256, 128, 64, 64,]

		self.encoder1 = resnet34d(pretrained=False, in_chans=CFG.crop_depth)

		self.decoder1 = SmpUnetDecoder(
			in_channel   = encoder1_dim[-1],
			skip_channel = encoder1_dim[:-1][::-1],
			out_channel  = decoder1_dim,
		)
		# -- pool attention weight
		self.weight1 = nn.ModuleList([
			nn.Sequential(
				nn.Conv2d(dim, dim, kernel_size=3, padding=1),
				nn.ReLU(inplace=True),
			) for dim in encoder1_dim
		])
		self.logit1 = nn.Conv2d(decoder1_dim[-1],1,kernel_size=1)

		#--------------------------------
		#
		encoder2_dim  = [64, 128, 256, 512]#
		decoder2_dim  = [128, 64, 32, ]
		self.encoder2 = resnet10t(pretrained=False, in_chans=decoder1_dim[-1])

		self.decoder2 = SmpUnetDecoder(
			in_channel   = encoder2_dim[-1],
			skip_channel = encoder2_dim[:-1][::-1],
			out_channel  = decoder2_dim,
		)
		self.logit2 = nn.Conv2d(decoder2_dim[-1],1,kernel_size=1)

	def forward(self, batch):
		v = batch['volume']
		B,C,H,W = v.shape
		vv = [
			v[:,i:i+CFG.crop_depth] for i in [0,2,4,]
		]
		K = len(vv)
		x = torch.cat(vv,0)
		#x = v

		#----------------------
		encoder = []
		e = self.encoder1
		x = e.conv1(x)
		x = e.bn1(x)
		x = e.act1(x);
		encoder.append(x)
		x = F.avg_pool2d(x, kernel_size=2, stride=2)
		x = e.layer1(x);
		encoder.append(x)
		x = e.layer2(x);
		encoder.append(x)
		x = e.layer3(x);
		encoder.append(x)
		x = e.layer4(x);
		encoder.append(x)
		# print('encoder', [f.shape for f in encoder])

		for i in range(len(encoder)):
			e = encoder[i]
			f = self.weight1[i](e)
			_, c, h, w = e.shape
			f = rearrange(f, '(K B) c h w -> B K c h w', K=K, B=B, h=h, w=w)  #
			e = rearrange(e, '(K B) c h w -> B K c h w', K=K, B=B, h=h, w=w)  #
			w = F.softmax(f, 1)
			e = (w * e).sum(1)
			encoder[i] = e

		feature = encoder[-1]
		skip = encoder[:-1][::-1]
		last, decoder = self.decoder1(feature, skip)
		logit1 = self.logit1(last)

		#----------------------
		x = last #.detach()
		#x = F.avg_pool2d(x,kernel_size=2,stride=2)
		encoder = []
		e = self.encoder2
		x = e.layer1(x); encoder.append(x)
		x = e.layer2(x); encoder.append(x)
		x = e.layer3(x); encoder.append(x)
		x = e.layer4(x); encoder.append(x)

		feature = encoder[-1]
		skip = encoder[:-1][::-1]
		last, decoder = self.decoder2(feature, skip)
		logit2 = self.logit2(last)
		logit2_out = F.interpolate(logit2, size=(H, W), mode='bilinear', align_corners=False, antialias=True)

		output = {
			'ink' : torch.sigmoid(logit2_out),
            'loss1': torch.sigmoid(logit1),
            'loss2': torch.sigmoid(logit2),
		}
		return output

def run_check_net():

	height,width =  CFG.crop_size, CFG.crop_size
	depth = CFG.infer_fragment_z[1]-CFG.infer_fragment_z[0]
	batch_size = 3

	batch = {
		'volume' : torch.from_numpy( np.random.choice(256, (batch_size, depth, height, width))).cuda().float(),
	}
	net = Net().cuda()

	with torch.no_grad():
		with torch.cuda.amp.autocast(enabled=True):
			output = net(batch)

	#---
	print('batch')
	for k, v in batch.items():
		print(f'{k:>32} : {v.shape} ')

	print('output')
	for k, v in output.items():
		print(f'{k:>32} : {v.shape} ')


run_check_net()
print('net ok !!!')


#============> loss <============
def build_loss():
    BCELoss     = smp.losses.SoftBCEWithLogitsLoss()
    DiceLoss    = smp.losses.DiceLoss(mode='binary')
    return {"BCELoss":BCELoss, "DICELoss":DiceLoss}


#============> scheduler <============
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
        optimizer, cfg.epochs, eta_min=1e-7)
    scheduler = GradualWarmupSchedulerV2(
        optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)

    return scheduler

def scheduler_step(scheduler, avg_val_loss, epoch):
    scheduler.step(epoch)

# infer here !!!!
#https://gist.github.com/janpaul123/ca3477c1db6de4346affca37e0e3d5b0
def mask_to_rle(mask):
    m = mask.reshape(-1)
    # m = np.where(mask > threshold, 1, 0).astype(np.uint8)
    s = np.array((m[:-1] == 0) & (m[1:] == 1))
    e = np.array((m[:-1] == 1) & (m[1:] == 0))

    s_index = np.where(s)[0] + 2
    e_index = np.where(e)[0] + 2
    length = e_index - s_index
    rle = ' '.join(map(str, sum(zip(s_index, length), ())))
    return rle

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
		score = beta * beta / (1 + beta * beta) * 1 / recall + 1 / (1 + beta * beta) * 1 / (precision)
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


def train_one(net, d):
    #volume, label = train_augment_v2(d)
	#get coord
    crop_size  = CFG.crop_size
    stride = CFG.stride
    H,W,D  = d.volume.shape # (8181, 6330, 12)
    labels = d.label # (8181, 6330)    
	##pad #assume H,W >size
    px, py = W % stride, H % stride  #
    if (px != 0) or (py != 0): # px=2 py=5
        px = stride - px
        py = stride - py
        pad_volume = np.pad(d.volume, [(0, py), (0, px), (0, 0)], constant_values=0)
        pad_labels = np.pad(d.label, [(0, py), (0, px),], constant_values=0)
    else:
        pad_volume = d.volume
        pad_labels = d.label

    pH, pW, _  = pad_volume.shape
    x = np.arange(0,pW-crop_size+1,stride)
    y = np.arange(0,pH-crop_size+1,stride)
    x,y = np.meshgrid(x,y)
    xy  = np.stack([x,y],-1).reshape(-1,2)
    print('H,W,pH,pW,len(xy)',H,W,pH,pW,len(xy))

    #=============> train_one_epoch <=============
    net.train()
    loss, train_loss = 0, 0

    #---
    batch_iter = np.array_split(xy, np.ceil(len(xy)/CFG.batch_size)) # ceil(27390/32) = 856 todo ??? //6

    # pbar = tqdm(enumerate(batch_iter), total=len(batch_iter), desc="Train")
    for t, xy0 in enumerate(batch_iter): # each batch has 32 = len(xy0)
        #print('\r: ', t, 'len--', len(batch_iter), end='')
        crop_size  = CFG.crop_size

        volume = []
        inklabels = []
        for x0,y0 in xy0 :
            v = pad_volume[y0:y0 + crop_size, x0:x0 + crop_size]
            k = pad_labels[y0:y0 + crop_size, x0:x0 + crop_size]
            # exclude areas with a mask value of 0
            # if np.all(k==0):
            #     continue
            # augmentation
            transform = A.Compose(CFG.train_aug_list)
            data = transform(image=v, mask=k)
            aug_image = data['image']
            aug_image = aug_image.permute((1,2,0))
            aug_mask = data['mask']
            volume.append(aug_image)
            inklabels.append(aug_mask)                                  # each volume (224, 224, 12)
        if volume == []:
            continue
        volume = np.stack(volume)                                # (32, 224, 224, 12)
        volume = np.ascontiguousarray(volume.transpose(0,3,1,2)) # (32, 12，224, 224)
        volume = volume/255
        volume = torch.from_numpy(volume).float().cuda()                
                                                                              # each inklabel (224, 224)
        inklabels = np.stack(inklabels)                                       # (32, 224, 224)
        inklabels = np.ascontiguousarray(inklabels)                           # (32, 224, 224)
        inklabels = torch.from_numpy(inklabels).unsqueeze(1).float().cuda()                # (32, 224, 224)      
        print('volume shape     :', volume.shape)
        print('inklabels shape  :', inklabels.shape)

        
        batch = { 'volume': volume, 'inklabels': inklabels } # [32, 12, 224, 224] [32, 1, 224, 224]
        
        # model output
        # input batch: volume [32, 12, 224, 224]
        # output ink mask: [32, 1, 224, 224] 0~1 min 0.0975 max 0.7783
        optimizer.zero_grad()
        with amp.autocast():
            output = net(batch)
            pred_labels = output['ink']   # [32, 1, 224, 224]
            labels = batch['inklabels']
        #middle loss
        new_labels = F.interpolate(labels, size=(CFG.stride, CFG.stride), mode='bilinear', align_corners=False, antialias=True)
        logit1 = output['loss1']
        logit2 = output['loss2']
        loss1 = criterion(logit1, new_labels)
        loss2 = criterion(logit2, new_labels)

        #BCE_loss = criterion(pred_labels, labels) # tensor(0.6158, device='cuda:0')loss.
        #Dice_loss = loss_dict["DICELoss"](pred_labels, labels)
        loss = loss1 + loss2 #+ BCE_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.detach().item()
        del loss1, loss2, loss, pred_labels, new_labels # BCE_loss, 
        gc.collect()
        torch.cuda.empty_cache()
        train_loss = train_loss / volume.shape[0]
        
        wandb.log({"train_loss":train_loss})
    epoch_train_loss = train_loss
    gc.collect()
    torch.cuda.empty_cache()
    wandb.log({"epoch_train_loss":epoch_train_loss})
    print("epoch: {epoch} /", "lr: {:.2E}".format(scheduler.get_last_lr()[0]), flush=True)
    print("epoch: {epoch} /", "BCE loss: {:.3f}".format(epoch_train_loss), flush=True)

    

  
# ==========> validation <=============
def valid_one(net, d):

	#get coord
    crop_size  = CFG.crop_size
    stride = CFG.stride
    H,W,D  = d.volume.shape # (9456, 9506, 12)
    labels = d.label # (9456, 9506)   
    mask = d.mask 
	##pad #assume H,W >size
    px, py = W % stride, H % stride
    if (px != 0) or (py != 0):
        px = stride - px
        py = stride - py
        pad_volume = np.pad(d.volume, [(0, py), (0, px), (0, 0)], constant_values=0)
        pad_labels = np.pad(d.label, [(0, py), (0, px),], constant_values=0)
    else:
        pad_volume = d.volume
        pad_labels = d.label

    pH, pW, _  = pad_volume.shape
    x = np.arange(0,pW-crop_size+1,stride)
    y = np.arange(0,pH-crop_size+1,stride)
    x,y = np.meshgrid(x,y)
    xy  = np.stack([x,y],-1).reshape(-1,2)
    print('H,W,pH,pW,len(xy)',H,W,pH,pW,len(xy))

    #--
    infer_mask = make_infer_mask()
    probability = np.zeros((pH,pW))
    count = np.zeros((pH,pW))    
   
    start_timer = time.time()
    #---
    net.eval()

    loss, bce_loss, valid_loss = 0, 0, 0

    batch_iter_valid = np.array_split(xy, np.floor(len(xy)/CFG.batch_size)) # ceil(27390/32) = 856
    
    for t, xy0 in enumerate(batch_iter_valid):
        #print('\r: ', t, 'len--', len(batch_iter), end='')
        crop_size  = CFG.crop_size

        volume = []
        inklabels = []
        for x0,y0 in xy0 :
            v = pad_volume[y0:y0 + crop_size, x0:x0 + crop_size]
            k = pad_labels[y0:y0 + crop_size, x0:x0 + crop_size]
            # if k.max()==0:
            #     continue
            # augmentation
            # transform = A.Compose(CFG.valid_aug_list)
            # data = transform(image=v, mask=k)
            # aug_image = data['image']
            # aug_image = aug_image.permute((1,2,0))
            # aug_mask = data['mask']
            volume.append(v)
            inklabels.append(k)
        # if volume == []:
        #     continue
        volume = np.stack(volume)  # (32, 224, 224, 12)
        volume = np.ascontiguousarray(volume.transpose(0,3,1,2)) # (32, 12，224, 224)
        volume = volume/255
        volume = torch.from_numpy(volume).float().cuda()

        inklabels = np.stack(inklabels)  # (32, 224, 224)
        inklabels = np.ascontiguousarray(inklabels) # (32, 224, 224)
        inklabels = torch.from_numpy(inklabels).unsqueeze(1).float().cuda()        
        print('volume shape     :', volume.shape)
        print('inklabels shape  :', inklabels.shape)

        batch_valid = { 'volume': volume, 'inklabels': inklabels } # [32, 12, 224, 224]
        

        with torch.no_grad():
            output = net(batch_valid)
            pred_labels = output['ink']   
            valid_labels = batch_valid['inklabels']
            bce_loss += criterion(pred_labels, valid_labels) 
            valid_loss = bce_loss.detach().item() / volume.shape[0]
            wandb.log({"valid_loss":valid_loss})    
   
        k = pred_labels.data.cpu().numpy()
        ##print(k.shape) # [32, 1, 224, 224]
        
        for b in range(CFG.batch_size):
            x0,y0 = xy0[b]
            probability[y0:y0 + crop_size, x0:x0 + crop_size] += k[b,0]*infer_mask # (224, 224)*(224, 224)
            count[y0:y0 + crop_size, x0:x0 + crop_size] += infer_mask
        print(f'\r @infer_one(): {t} / {len(batch_iter_valid)} : {time_to_str(time.time() - start_timer, "sec")}', end='', flush=True)
    probability = probability/(count+0.000001)
    probability = probability[:H,:W]
    probability = probability*d.mask
    
    epoch_valid_loss = valid_loss
    #wandb.log({"epoch_valid_loss":epoch_valid_loss})       
    return probability, epoch_valid_loss
    print("@valid epoch: {epoch} /", "BCE loss: {:.3f}".format(epoch_valid_loss), flush=True)

    # text = metric_to_text(probability, d.label, d.mask)
    # print(text)

#==================================================
#============>>  TRAINING  <<====================
model = [
    #'./ckpts/meanpool-resnet34d-unet_fragment_3_epoch_15.pth'

]

#----
# if CFG.mode == 'test':
#     net_all = []
#     for i, checkpoint in enumerate(model):
#         print(checkpoint)
#         net = Net()
#         f = torch.load(checkpoint)
#         net.load_state_dict(f)
#         net_all.append(net)
#     net = net_all[0].cuda()
# else:
#     net = Net().cuda()
    #ckpt = torch.load('./ckpts/meanpool-unet_frag_3_epoch_11.pth')
    #net.load_state_dict(ckpt)
net = Net().cuda()
scaler = amp.GradScaler()
optimizer = torch.optim.AdamW(net.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
#loss_dict = build_loss()
criterion = torch.nn.BCEWithLogitsLoss()
# scheduler = 'GradualWarmupSchedulerV2'
scheduler = get_scheduler(CFG, optimizer)


all_text = []


wandb.init(project='ink-detection-comp',
            name=f'meanpool-resnet34d-unet_valid_{valid_id}',
        group="meanpool-resnet34d-unet-aug", 
        job_type="train")


for i, fragment_id in enumerate(train_id):
    torch.cuda.empty_cache()
    gc.collect()
    d = read_data1(fragment_id, z0=22, z1=38)
    if i != 0:
        d.update(d)
print('')
print('fragment_id:', d.fragment_id)
print('volume:', d.volume.shape, ' min:', d.volume.min(), ' max:', d.volume.max())
print('mask  :', d.mask.shape, ' min:', d.mask.min(), ' max:', d.mask.max())
if 'train' in CFG.mode:
    print('ir    :', d.ir.shape, ' min:', d.ir.min(), ' max:', d.ir.max())
    print('label :', d.label.shape, ' min:', d.label.min(), ' max:', d.label.max())

for epoch in range(1, CFG.epochs+1):
    
    all_text.append(f'======Train epoch {epoch}\n')
    train_one(net, d)
    all_text.append(f'======Valid epoch {epoch} on fragment_{valid_id}\n')
    d_valid = read_data1(valid_id[0], z0=22, z1=38)
    probability, epoch_valid_loss = valid_one(net, d_valid)
    scheduler_step(scheduler, epoch_valid_loss, epoch)
    predict = (probability>0.50).astype(np.uint8)

    all_text.append(f'epoch_{epoch}\n')
    text = metric_to_text(probability, d_valid.label, d_valid.mask)
    all_text.append(text)
    print(text)

    if epoch >= 10:
        torch.save(net.state_dict(), f"./ckpts/meanpool-unet_frag_{fragment_id}_epoch_{epoch}.pth")
wandb.finish()

# for epoch in range(1, CFG.epochs+1):
#     for fragment_id in train_id:
#         all_text.append(f'======Train epoch {epoch} on fragment_{fragment_id}\n')
#         torch.cuda.empty_cache()
#         gc.collect()
#         d = read_data1(fragment_id, z0=22, z1=38)
        
#     train_one(net, d_train)
#     all_text.append(f'======Valid epoch {epoch} on fragment_{valid_id}\n')
#     d_valid = read_data1(valid_id[0], z0=22, z1=38)
#     probability, epoch_valid_loss = valid_one(net, d_valid)
#     scheduler_step(scheduler, epoch_valid_loss, epoch)
#     predict = (probability>0.50).astype(np.uint8)

#     all_text.append(f'epoch_{epoch}\n')
#     text = metric_to_text(probability, d_valid.label, d_valid.mask)
#     all_text.append(text)
#     print(text)

#     if epoch >= 10:
#         torch.save(net.state_dict(), f"./ckpts/meanpool-unet_frag_{fragment_id}_epoch_{epoch}.pth")
# wandb.finish()



# Convert the list to a NumPy array
my_array = np.array(all_text)

# Save the array as a text file
file_path = "train3.txt"
np.savetxt(file_path, my_array, fmt="%s")



#==================================================

# submission = defaultdict(list)
# for t,fragment_id in enumerate(valid_id):
#     d = read_data1(fragment_id)
    
#     print('==================================')
#     print('fragment_id', d.fragment_id)
#     print('\tmask', d.mask.shape)
#     print('\tlabel', d.label.shape)
#     print('\tvolume', d.volume.shape)
#     print('CFG.stride', CFG.stride)
#     print('CFG.crop_size', CFG.crop_size)  
#     print('')

#     probability = train_one(net, d)
#     print('probability', probability.shape)

#     probability = d.mask*probability
#     predict = (probability>0.5).astype(np.uint8)
    
#     #----
#     submission['Id'].append(fragment_id)
#     submission['Predicted'].append(mask_to_rle(predict))
    
#     #----
#     probability8 = (probability * 255).astype(np.uint8)
#     plt.figure(t), plt.imshow(probability8, cmap='gray')
#     #plt.waitforbuttonpress()
#     if 'train' in CFG.mode:
#         text = metric_to_text(probability, d.label)
#         print(text)
#     print('')

# print('')
# print('CFG.mode', CFG.mode)
# submit_df = pd.DataFrame.from_dict(submission)