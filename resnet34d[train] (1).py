my_lib_dir ='/kaggle/input/ink-00/my_lib'
import pdb
import sys
import time
import tqdm
sys.path.append(my_lib_dir)
sys.path.append('/kaggle/input/pretrainedmodels/pretrainedmodels-0.7.4')
sys.path.append('/kaggle/input/efficientnet-pytorch/EfficientNet-PyTorch-master')
sys.path.append('/kaggle/input/timm-pytorch-image-models/pytorch-image-models-master')
sys.path.append('/kaggle/input/segmentation-models-pytorch/segmentation_models.pytorch-master')
sys.path.append('/kaggle/input/einops/einops-master')

from helper import *

import numpy as np
import pandas as pd

from collections import defaultdict
from glob import glob
import PIL.Image as Image
Image.MAX_IMAGE_PIXELS = 10000000000  # Ignore PIL warnings about large images

import cv2
import wandb
import torch
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
  
print('import ok !!!')

class Config(object):
    #==============>> model <<=================
    mode = ['train']
    crop_size = 224
    crop_depth = 8+4
    one_depth = 8 #6+4
    #==============>> #todo: ??? <<================= 
    model_name = 'Unet'
    #backbone = 'efficientnet-b5'
    #backbone = 'mit_b5'
    backbone = 'resnet34d'
    #backbone = 'resnext50_32x4d'
    pretrained = True
    # ==== data

    #==============>> training cfg <<=================
    seed = 42  
    num_worker = 2 # debug => 0
    batch_size = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # optimizer
    epochs = 20
    lr = 1e-4
    wd = 1e-5
    lr_drop = 15

    # infer
    thr = 0.5

    is_tta = True

CFG = Config()
CFG.fragment_z0 = 22 #-1 #todo: how to choose the number?
CFG.fragment_z1 = CFG.fragment_z0+CFG.crop_depth #+2
#WANDB_RUN_NAME = f'{CFG.backbone}_ep{CFG.backbon}'

if 'train' in CFG.mode:  #todo: try stride different for train/test
	CFG.stride = CFG.crop_size//4
if 'test' in CFG.mode:
	CFG.stride = CFG.crop_size//8


## dataset ##
if 'train' in CFG.mode:
    data_dir = '/root/autodl-tmp/VCInkDectection/input/vesuvius-challenge-ink-detection/train' #todo change
    train_id = ['1', '3']
    valid_id =['2a']

if 'test' in CFG.mode: 
	data_dir = '/root/autodl-tmp/VCInkDectection/input/vesuvius-challenge-ink-detection/test'
	valid_id = glob(f'{data_dir}/*')
	valid_id = sorted(valid_id)
	valid_id = [f.split('/')[-1] for f in valid_id]

print('data_dir', data_dir)
print('valid_id', valid_id)



def do_binarise(m, threshold=0.5): #binary mask
    m = m-m.min()
    m = m/(m.max()+1e-7)
    m = (m>threshold).astype(np.float32)
    return m

def read_data(fragment_id, z0=CFG.fragment_z0, z1=CFG.fragment_z1):
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

def read_data1(fragment_id):
	if fragment_id=='2a':
		y = 9456
		d = read_data('2') 
		d = dotdict(
			fragment_id='2a',
			volume  = d.volume[:y],
			ir      = d.ir[:y],
			label   = d.label[:y],
			mask    = d.mask[:y],
		)
	elif  fragment_id=='2b':
		y = 9456
		d = read_data('2') 
		d = dotdict(
			fragment_id='2b',
			volume  = d.volume[y:],
			ir      = d.ir[y:],
			label   = d.label[y:],
			mask    = d.mask[y:],
		)
	else:
		d = read_data(fragment_id)
	return d

def run_check_data():
    d=read_data1(train_id)#valid_id[0] # changed
    print('')
    print('fragment_id:', d.fragment_id)
    print('volume:', d.volume.shape, 'min:', d.volume.min(), 'max:', d.volume.max())
    print('mask  :', d.mask.shape, 'min:', d.mask.min(), 'max:', d.mask.max())
    if 'train' in CFG.mode:
        print('ir    :', d.ir.shape, 'min:', d.ir.min(), 'max:', d.ir.max())
        print('label :', d.label.shape, 'min:', d.label.min(), 'max:', d.label.max())

#run_check_data()
#print('data ok !!!')

## model ##


class SmpUnetDecoder(UnetDecoder):
	def __init__(self, **kwargs):
		super(SmpUnetDecoder, self).__init__(
			**kwargs)

	def forward(self, encoder):
		feature = encoder[::-1]  # reverse channels to start from head of encoder
		head = feature[0]           # head: [32, 512, 7, 7]
		skip = feature[1:] + [None] # skip[list]: len 5 [[32, 256, 14, 14], [32, 128, 28, 28], [32, 64, 56, 56], [32, 64, 112, 112], [None]]
		d = self.center(head)       # [32, 512, 7, 7]

		decoder = []
		for i, decoder_block in enumerate(self.blocks): # 5 blocks
			# print(i, d.shape, skip[i].shape if skip[i] is not None else 'none')
			# print(decoder_block.conv1[0])
			# print('')
			s = skip[i]              # 5个 [32, 256, 14, 14] ...
			d = decoder_block(d, s)  # 5个 [32, 256, 14, 14] ...
			decoder.append(d)

		last  = d # [32, 16, 112, 112]
		return last, decoder
        # decoder
        # 0: [32, 256, 14, 14]
        # 1: [32, 128, 28, 28]
        # 2: [32, 64, 56, 56]
        # 3: [32, 32, 112, 112]
        # 4: [32, 16, 224, 224]

class Net(nn.Module):
	def __init__(self,):
		super().__init__()

		conv_dim=64
		encoder_dim  = [conv_dim] + [64, 128, 256, 512 ] # [64, 64, 128, 256, 512]
		self.encoder = resnet34d(pretrained=False,in_chans=CFG.one_depth)

		self.decoder = SmpUnetDecoder(
			encoder_channels=[0] + encoder_dim, # [0, 64, 64, 128, 256, 512]
			decoder_channels=[256, 128, 64, 32, 16],
			n_blocks=5,
			use_batchnorm=True,
			center=False,
			attention_type=None,
		)
		self.logit = nn.Conv2d(16,1,kernel_size=1)

		#-- pool attention weight
		self.weight = nn.ModuleList([
			nn.Sequential(
				nn.Conv2d(dim, dim, kernel_size=3, padding=1),
				nn.ReLU(inplace=True),
			) for dim in encoder_dim  # Encoder_dim = [64, 64, 128, 256, 512]
		])

	def forward(self, batch):
		v = batch['volume'] # v: torch.Size([32, 12, 224, 224])
		B,C,H,W = v.shape
		vv = [
			v[:,i:i+CFG.one_depth] for i in [0,2,4] # v[:,0:8] v[:,2:10] v[:,4:12]
		]  # [torch.Size([32, 8, 224, 224]), [32, 8, 224, 224], [32, 8, 224, 224]]
		K = len(vv)
		x = torch.cat(vv,0) # torch.Size([96, 8, 224, 224])
        # v batch_size 32 channels 12  ==> x batchsize 96 channels 8 
        # channel 12中取overlapping的3部分
        # todo: why change shape?
		# x = v
		# ---- new input volume size = [96, 8, 224, 224]
		encoder = []
		x = self.encoder.conv1(x)
		x = self.encoder.bn1(x)
		x = self.encoder.act1(x)   ; encoder.append(x)
		x = F.avg_pool2d(x,kernel_size=2,stride=2)
		x = self.encoder.layer1(x) ; encoder.append(x)
		x = self.encoder.layer2(x) ; encoder.append(x)
		x = self.encoder.layer3(x) ; encoder.append(x)
		x = self.encoder.layer4(x) ; encoder.append(x)
		#print('encoder', [f.shape for f in encoder])
        # 0:torch.Size([96, 64, 112, 112])
        # 1:torch.Size([96, 64, 56, 56])
        # 2:torch.Size([96, 128, 28, 28])
        # 3:torch.Size([96, 256, 14, 14])
        # 4:torch.Size([96, 512, 7, 7])

		#encode pooling -------   softmax(encoder[i]*pool attention weight[i])
		#<todo> add positional encode (z slice no.)
		for i in range(len(encoder)):
			e = encoder[i]                    # [96, 64, 112, 112]
			f = self.weight[i](encoder[i])    # [96, 64, 112, 112]
			_, c, h, w = f.shape
                        # K=3, B=32, c=64, h&w=112 
                        # f: [96, 64, 112, 112] ==> [32, 3, 64, 112, 112]
			f = rearrange(f, '(K B) c h w -> B K c h w', K=K, B=B, h=h, w=w) #f.reshape(B, K, c, h, w)
                        # e: [96, 64, 112, 112] ==> [32, 3, 64, 112, 112]
			e = rearrange(e, '(K B) c h w -> B K c h w', K=K, B=B, h=h, w=w) #e.reshape(B, K, c, h, w)
			w = F.softmax(f, 1)# [32, 3, 64, 112, 112] dim=1 we have k=3 3个batch, c=64是encoder conv后的输出通道，刚开始是8（12切成重叠的3份） so every slice along dim will sum to 1
			e = (w * e).sum(1) #
			encoder[i] = e
        # encoder[list]: len 5 --- [torch.Size([32, 64, 112, 112]), torch.Size([32, 64, 56, 56]), torch.Size([32, 128, 28, 28]), torch.Size([32, 256, 14, 14]), torch.Size([32, 512, 7, 7])]
		# decoder[list]: len 5 --- [torch.Size([32, 256, 14, 14]), torch.Size([32, 128, 28, 28]), torch.Size([32, 64, 56, 56]), torch.Size([32, 32, 112, 112]), torch.Size([32, 16, 224, 224])]
		last, decoder = self.decoder(encoder) # encoder[list]: len 5
		#print('decoder',[f.shape for f in decoder])
		#print('last',last.shape) [32, 16, 224, 224]
		logit = self.logit(last)  # [32, 1, 224, 224] conv 16-->1

		output = {
			'ink' : torch.sigmoid(logit),
		}
		return output

#============>>  TRAINING  <<====================
net = Net().cuda()
scaler = amp.GradScaler()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(net.parameters(), lr=CFG.lr)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CFG.lr,
                                                steps_per_epoch=10, epochs=CFG.epochs//10,
                                                pct_start=0.1)



#============

def run_check_net():

    height,width =  CFG.crop_size, CFG.crop_size
    depth = CFG.crop_depth
    batch_size = 2

    batch = {
        'volume' : torch.from_numpy( np.random.choice(256, (batch_size, depth, height, width))).float(),#.cuda()
    }
    net = Net()#.cuda()

    with torch.no_grad():       # input batch: volume [2, 12, 224, 224]
        with torch.cuda.amp.autocast(enabled=True):
            output = net(batch) # output ink mask: [2, 1, 224, 224] 0~1 min 0.0975 max 0.7783

    #---
    print('batch')
    for k, v in batch.items():
        print(f'{k:>32} : {v.shape} ')

    print('output')
    for k, v in output.items():
        print(f'{k:>32} : {v.shape} ')

#run_check_net()
print('net ok !!!')

#============> loss <============
def build_loss():
    BCELoss     = smp.losses.SoftBCEWithLogitsLoss()
    DiceLoss    = smp.losses.DiceLoss(mode='binary')
    return {"BCELoss":BCELoss, "DiceLoss":DiceLoss}

#============> metrics <============
# ref - https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/397288
def fbeta_score(preds, targets, threshold, beta=0.5, smooth=1e-5):
    preds_t = torch.where(preds > threshold, 1.0, 0.0).float()
    y_true_count = targets.sum()
    
    ctp = preds_t[targets==1].sum()
    cfp = preds_t[targets==0].sum()
    beta_squared = beta * beta

    c_precision = ctp / (ctp + cfp + smooth)
    c_recall = ctp / (y_true_count + smooth)
    dice = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall + smooth)

    return dice

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
		score = beta * beta / (1 + beta * beta) * 1 / recall + 1 / (1 + beta * beta) * 1 / precision
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


def train_one(net, d_train):

    #get coord
    size   = CFG.crop_size
    stride = CFG.stride
    H,W,D  = d_train.volume.shape # (9456, 9506, 12)
    labels = d_train.label # (9456, 9506)

    x = np.arange(0,W-size+1,stride) # x len[165]: [0,56,112,...,9240]
    y = np.arange(0,H-size+1,stride) # y len[166]: [0,56,112,...,9184]
    x,y = np.meshgrid(x,y) # (165, 166)
    xy  = np.stack([x,y],-1).reshape(-1,2) # xy len[165*166=27390]
    print('H,W,len(xy)',H,W,len(xy)) # H,W,len(xy) 9456 9506 27390

    net.train()
    
    
    loss, train_loss = 0, 0

    #---
    batch_iter = np.array_split(xy, np.ceil(len(xy)/CFG.batch_size)) # ceil(27390/32) = 856

    # pbar = tqdm(enumerate(batch_iter), total=len(batch_iter), desc="Train")
    for t, xy0 in enumerate(batch_iter): # each batch has 32 = len(xy0)
        #print('\r: ', t, 'len--', len(batch_iter), end='')
        crop_size  = CFG.crop_size

        volume = []
        inklabels = []
        for x0,y0 in xy0 :
            v = d_train.volume[y0:y0 + crop_size, x0:x0 + crop_size]
            k = d_train.label[y0:y0 + crop_size, x0:x0 + crop_size]
            volume.append(v)
            inklabels.append(k)                                  # each volume (224, 224, 12)
        volume = np.stack(volume)                                # (32, 224, 224, 12)
        volume = np.ascontiguousarray(volume.transpose(0,3,1,2)) # (32, 12，224, 224)
        volume = volume/255
        volume = torch.from_numpy(volume).float().cuda()                
                                                                              # each inklabel (224, 224)
        inklabels = np.stack(inklabels)                                       # (32, 224, 224)
        inklabels = np.ascontiguousarray(inklabels)                           # (32, 224, 224)
        inklabels = torch.from_numpy(inklabels).float().cuda()                # (32, 224, 224)      
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
            bce_loss = 0.5 * loss_dict["BCELoss"](pred_labels, labels) # tensor(0.6158, device='cuda:0')loss.
            Dice_loss = 0.5 * loss_dict["DICELoss"](pred_labels, labels)
            loss = bce_loss + Dice_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.detach().item()
            train_loss = train_loss / (t+1)
        
        wandb.log({"train_loss":train_loss})
    epoch_train_loss = train_loss
    wandb.log({"epoch_train_loss":epoch_train_loss})
    print("epoch: {epoch} /", "lr: {:.2E}".format(scheduler.get_last_lr()[0]), flush=True)
    print("epoch: {epoch} /", "BCE loss: {:.3f}".format(train_loss), flush=True)

    scheduler.step()
  
# ==========> validation <=============
def valid_one(net, d_valid):
    size   = CFG.crop_size
    stride = CFG.stride
    #get valid dataset
    H,W,D  = d_valid.volume.shape # (9456, 9506, 12)
    labels = d_valid.label # (9456, 9506)

    x = np.arange(0,W-size+1,stride) # x len[165]: [0,56,112,...,9240]
    y = np.arange(0,H-size+1,stride) # y len[166]: [0,56,112,...,9184]
    x,y = np.meshgrid(x,y) # (165, 166)
    xy  = np.stack([x,y],-1).reshape(-1,2) # xy len[165*166=27390]
    print('H,W,len(xy)',H,W,len(xy)) # H,W,len(xy) 9456 9506 27390

    probability = np.zeros((H,W))
    count = np.zeros((H,W))
    start_timer = time.time()
    #---
    net.eval()

    loss, valid_loss = 0, 0

    batch_iter_valid = np.array_split(xy, np.ceil(len(xy)/CFG.batch_size)) # ceil(27390/32) = 856
    
    for t, xy0 in enumerate(batch_iter_valid):
        #print('\r: ', t, 'len--', len(batch_iter), end='')
        crop_size  = CFG.crop_size

        volume = []
        inklabels = []
        for x0,y0 in xy0 :
            v = d_valid.volume[y0:y0 + crop_size, x0:x0 + crop_size]
            k = d_valid.label[y0:y0 + crop_size, x0:x0 + crop_size]
            volume.append(v)
            inklabels.append(k)
        volume = np.stack(volume)  # (32, 224, 224, 12)
        volume = np.ascontiguousarray(volume.transpose(0,3,1,2)) # (32, 12，224, 224)
        volume = volume/255
        volume = torch.from_numpy(volume).float().cuda()

        inklabels = np.stack(inklabels)  # (32, 224, 224)
        inklabels = np.ascontiguousarray(inklabels) # (32, 224, 224)
        inklabels = torch.from_numpy(inklabels).float().cuda()        
        print('volume shape     :', volume.shape)
        print('inklabels shape  :', volume.shape)

        batch_valid = { 'volume': volume, 'inklabels': inklabels } # [32, 12, 224, 224]
        

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                output = net(batch_valid)
                pred_labels = output['ink']   
                valid_labels = batch_valid['inklabels']
                bce_loss = 0.5 * loss_dict["BCELoss"](pred_labels, valid_labels) 
                Dice_loss = 0.5 * loss_dict["DICELoss"](pred_labels, valid_labels)
                loss = bce_loss + Dice_loss
                valid_loss += loss.detach().item()
                valid_loss = valid_loss / (t+1)
            wandb.log({"valid_loss":valid_loss})    
   
        k = pred_labels.data.cpu().numpy()
        ##print(k.shape) # [32, 224, 224]
        
        for b in range(CFG.batch_size):
            x0,y0 = xy0[b]
            probability[y0:y0 + crop_size, x0:x0 + crop_size] += k[b,0] #第b个的
            count[y0:y0 + crop_size, x0:x0 + crop_size] += 1
        print(f'\r @infer_one(): {t} / {len(batch_iter_valid)} : {time_to_str(time.time() - start_timer, "sec")}', end='', flush=True)
    probability = probability/(count+0.000001)

    epoch_valid_loss = valid_loss
    wandb.log({"epoch_valid_loss":epoch_valid_loss})       

    text = metric_to_text(probability, valid_labels)
    print(text)

    
    for threshold in np.arange(0.2, 0.65, 0.05):
        fbeta = fbeta_score(probability, valid_labels, threshold)
        print(f"Threshold : {threshold:.2f}\tFBeta : {fbeta:.6f}")
    
    return probability

#==================================================
is_debug = True
if not is_debug:
    wandb.init(project='ink detection',
            group="exp_1", 
            job_type="train")
    
for epoch in range(1, CFG.epochs+1):
    for fragment_id in train_id:
        d_train = read_data1(fragment_id)
        d_valid = read_data1(valid_id[0])
        loss_dict = build_loss()
        probability = train_one(net, d_train, d_valid)
        print('probability', probability.shape)

    if epoch >= 10:
        torch.save(net.state_dict(), f"./ckpts/resnet34d_epoch_{epoch}.pt")



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