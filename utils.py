import os
import time
import shutil
import math

import torch
import numpy as np
from torch.optim import SGD, Adam
from tensorboardX import SummaryWriter


class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)


_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                or input('{} exists, remove? (y/[n]): '.format(path)) == 'y'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def set_save_path(save_path, remove=True):
    ensure_path(save_path, remove=remove)
    set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    return log, writer


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def make_optimizer(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret

#将LR上的坐标映射到HR网格上，不归一化
def LR_coord_map_HR(shape,coordinate, ranges=None):
    ret=torch.zeros([coordinate.shape[0],coordinate.shape[1]])
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        # seq = v0 + r + (2 * r) * torch.arange(n).float()
        seq = v0 + r + (2 * r) * coordinate[:,i]
        ret[:,i]=seq
    
    
    return ret

def to_pixel_samples_unnormalize(img,range):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
        with LR coordinate map to HR
    """
    coord = make_coord(img.shape[-2:],range)
    rgb = img.view(3, -1).permute(1, 0)
    return coord, rgb

  

def to_pixel_samples(img):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """
    coord = make_coord(img.shape[-2:])
    rgb = img.view(3, -1).permute(1, 0)
    return coord, rgb

def coordinate_normalize(src,length, ranges=None):
    """ normalize coordinates.
        length: scalar ,used to calculate r
        src: orign coordinates
    """
    # coord_seqs = []
    # for i, n in enumerate(shape):
    if ranges is None:
        v0, v1 = -1, 1
    else:
        for i in length(ranges):
            v0, v1 = ranges[i]
        
    r = (v1 - v0) / (2 * length)
    seq = v0 + r + (2 * r) * src
    
    # if flatten:
    #     ret = ret.view(-1, ret.shape[-1])
    return seq

def calc_psnr(sr, hr, dataset=None, scale=1, rgb_range=1):
    diff = (sr - hr) / rgb_range
    if dataset is not None:
        if dataset == 'benchmark':
            shave = scale
            if diff.size(1) > 1:
                gray_coeffs = [65.738, 129.057, 25.064]
                convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
                diff = diff.mul(convert).sum(dim=1)
        elif dataset == 'div2k':
            shave = scale + 6
        else:
            raise NotImplementedError
        valid = diff[..., shave:-shave, shave:-shave]
        # valid=diff
    else:
        valid = diff
    mse = valid.pow(2).mean()
    return -10 * torch.log10(mse)

#Y通道计算PSNR
def calc_psnr_Y(sr, hr, dataset=None, scale=1, rgb_range=1):
    # diff = (sr - hr) / rgb_range       
    if sr.size(1) > 1:
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = sr.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        sr = sr.mul(convert).sum(dim=1)
        hr = hr.mul(convert).sum(dim=1)
        # sr1=torch.tensor((sr+16/256)*255,dtype=torch.uint8)
        # hr1=torch.tensor((hr+16/256)*255,dtype=torch.uint8)
        sr=((sr+16/256)*255).clamp(0,255)
        hr=((hr+16/256)*255).clamp(0,255)
        sr1=sr.type(torch.ByteTensor) #注意 负值则会变为255而不是0
        hr1=hr.type(torch.ByteTensor)
    valid=sr1.float()-hr1.float()
    mse = valid.pow(2).mean()
    return -10 * torch.log10(mse/(255**2))

def compute_psnr_uint8(clean_img, img):
    """compute the psnr
    """
    clean_img = np.array(clean_img).clip(0,255).astype(np.uint8)
    img = np.array(img).clip(0,255).astype(np.uint8)
    mse = np.mean((clean_img.astype(np.float32) - img.astype(np.float32)) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))