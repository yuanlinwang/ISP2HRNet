import functools
import random
import math
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register
from utils import to_pixel_samples,coordinate_normalize,to_pixel_samples_unnormalize,make_coord
from utils import LR_coord_map_HR
import cv2

@register('sr-implicit-paired')
class SRImplicitPaired(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]

        s = img_hr.shape[-2] // img_lr.shape[-2] # assume int scale
        if self.inp_size is None:
            h_lr, w_lr = img_lr.shape[-2:]
            img_hr = img_hr[:, :h_lr * s, :w_lr * s]
            crop_lr, crop_hr = img_lr, img_hr
        else:
            w_lr = self.inp_size
            x0 = random.randint(0, img_lr.shape[-2] - w_lr)
            y0 = random.randint(0, img_lr.shape[-1] - w_lr)
            crop_lr = img_lr[:, x0: x0 + w_lr, y0: y0 + w_lr]
            w_hr = w_lr * s
            x1 = x0 * s
            y1 = y0 * s
            crop_hr = img_hr[:, x1: x1 + w_hr, y1: y1 + w_hr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }


def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, Image.BICUBIC)(
            transforms.ToPILImage()(img)))


@register('sr-implicit-downsampled')
class SRImplicitDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        s = random.uniform(self.scale_min, self.scale_max)

        if self.inp_size is None:
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)
            img = img[:, :round(h_lr * s), :round(w_lr * s)] # assume round int
            img_down = resize_fn(img, (h_lr, w_lr))
            crop_lr, crop_hr = img_down, img
        else:
            w_lr = self.inp_size
            w_hr = round(w_lr * s)
            x0 = random.randint(0, img.shape[-2] - w_hr)
            y0 = random.randint(0, img.shape[-1] - w_hr)
            crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            crop_lr = resize_fn(crop_hr, w_lr)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }
        
#不规则采样——非x2x3x4的超分factor
@register('sr-implicit-downsampled-sample')
class SRImplicitDownsampled_sample(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q
        self.ratio=0.5

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        s = random.uniform(self.scale_min, self.scale_max)

        
        h_lr = math.floor(img.shape[-2] / s + 1e-9)
        w_lr = math.floor(img.shape[-1] / s + 1e-9)
        img = img[:, :round(h_lr * s), :round(w_lr * s)] # assume round int
        img_down = resize_fn(img, (h_lr, w_lr))
        crop_lr, crop_hr = img_down, img

        lr_coord, lr_rgb = to_pixel_samples(crop_lr.contiguous())
        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())
        #crop_lr 采样 形成输入
        #采样点数
        sample_points=round(h_lr*w_lr*self.ratio)
    
        #对crop_lr采样
        np.random.seed(5)
        regular_coordinate=np.zeros([sample_points,2])
        ####选规则输入点
        point_idxs=np.random.choice(np.arange(h_lr*w_lr), sample_points, replace=False)
        regular_coordinate[:,1]=point_idxs%w_lr #col
        regular_coordinate[:,0]=point_idxs//w_lr #row
        map_rgb=lr_rgb[point_idxs,:] #[sample_points,3]

        #将LR坐标投影到HR网格上作为输入
        #测试时不用归一化，将LR坐标点映射到HR即可 map_coord是先row 再col
        map_coord= LR_coord_map_HR(crop_lr.shape[-2:],torch.from_numpy(regular_coordinate),ranges=[(-0.5,crop_hr.shape[-2]-0.5),(-0.5,crop_hr.shape[-1]-0.5)])
        #注意xy顺序，先拼接x(col),再拼接y(row)
        irregular_input=torch.concat([map_coord[:,1].unsqueeze(dim=1),map_coord[:,0].unsqueeze(dim=1),map_rgb],dim=1)
        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': irregular_input,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb,
            'hr_h': crop_hr.shape[1],
            'hr_w': crop_hr.shape[2],
        }

@register('sr-implicit-uniform-varied')
class SRImplicitUniformVaried(Dataset):

    def __init__(self, dataset, size_min, size_max=None,
                 augment=False, gt_resize=None, sample_q=None):
        self.dataset = dataset
        self.size_min = size_min
        if size_max is None:
            size_max = size_min
        self.size_max = size_max
        self.augment = augment
        self.gt_resize = gt_resize
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]
        p = idx / (len(self.dataset) - 1)
        w_hr = round(self.size_min + (self.size_max - self.size_min) * p)
        img_hr = resize_fn(img_hr, w_hr)

        if self.augment:
            if random.random() < 0.5:
                img_lr = img_lr.flip(-1)
                img_hr = img_hr.flip(-1)

        if self.gt_resize is not None:
            img_hr = resize_fn(img_hr, self.gt_resize)

        hr_coord, hr_rgb = to_pixel_samples(img_hr)

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / img_hr.shape[-2]
        cell[:, 1] *= 2 / img_hr.shape[-1]

        return {
            'inp': img_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }


@register('irregular-sampled')
class IrregularSampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None, normalize=True):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q
        self.normalize=normalize

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        s = random.uniform(self.scale_min, self.scale_max)

        if self.inp_size is None:
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)
            img = img[:, :round(h_lr * s), :round(w_lr * s)] # assume round int
            # img_down = resize_fn(img, (h_lr, w_lr))
            # crop_lr, crop_hr = img_down, img
            crop_hr = img
        else:
            w_lr = self.inp_size
            h_lr = self.inp_size
            w_hr = round(w_lr * s)
            x0 = random.randint(0, img.shape[-2] - w_hr)
            y0 = random.randint(0, img.shape[-1] - w_hr)
            crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            # crop_lr = resize_fn(crop_hr, w_lr)
            s=w_hr/w_lr

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            # crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)
        
        im=crop_hr.permute(1,2,0)
        #生成LR GT
        # width=48
        # height=48
        # mapx = np.array([[i for i in range(width)] for j in range(height)], dtype=np.float32)
        # mapx=mapx*s+s/2-0.5
        # mapy = np.array([[j for i in range(width)] for j in range(height)], dtype=np.float32)
        # mapy=mapy*s+s/2-0.5
        # LR = cv2.remap(im.numpy(), mapx, mapy, cv2.INTER_CUBIC)  # 复制
        # LR=torch.tensor(LR)
        # LR.clamp_(0,1) #[h,w,3]

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())
        #记得删除，用于测试
        # np.random.seed(0)
        #选离散输入点
        #np.random.uniform 左闭右开
        mapx = np.random.uniform(-0.5, im.shape[1]-0.5, (h_lr ,w_lr))
        mapx=mapx.astype(np.float32)
        mapy = np.random.uniform(-0.5, im.shape[0]-0.5, (h_lr ,w_lr))
        mapy=mapy.astype(np.float32)
        #亚像素值
        irregular_value = cv2.remap(im.numpy(), mapx, mapy, cv2.INTER_CUBIC) 
        irregular_value=np.clip(irregular_value,0,1)
        #将坐标值映射到[-1,1]
        mapx= mapx.reshape(h_lr *w_lr,1)
        mapy= mapy.reshape(h_lr *w_lr,1)
        if self.normalize:
            mapx=coordinate_normalize(mapx,im.shape[1])
            mapy=coordinate_normalize(mapy,im.shape[0])
        #拼接 注意拼接顺序(mapx——>width)(mapy——>height)
        irregular_input=np.concatenate([mapx,mapy,irregular_value.reshape(h_lr *w_lr,3)],axis=1)
        irregular_input=torch.tensor(irregular_input)

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': irregular_input,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb,
            'hr_h': crop_hr.shape[1],
            'hr_w': crop_hr.shape[2],
            
        }
        

        
        
        
#用于测试亚像素点输入
@register('irregular-sampled-test')
class IrregularSampled_test(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None, normalize=True,seed=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q
        self.normalize=normalize
        self.seed =seed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        s = random.uniform(self.scale_min, self.scale_max)#乘的比例的平方

        
        h_lr = math.ceil(img.shape[-2] *math.sqrt(s) + 1e-9)
        w_lr = math.ceil(img.shape[-1] *math.sqrt(s) + 1e-9)

        crop_hr = img
        
        im=crop_hr.permute(1,2,0)#[h,w,3]

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())
        #seed
        if self.seed is not None:
            np.random.seed(self.seed)
        mapx = np.random.uniform(-0.5, im.shape[1]-0.5, (h_lr ,w_lr))
        mapx=mapx.astype(np.float32)
        if self.seed is not None:
            np.random.seed(self.seed+1)
        mapy = np.random.uniform(-0.5, im.shape[0]-0.5, (h_lr ,w_lr))
        mapy=mapy.astype(np.float32)
        #亚像素值
        irregular_value = cv2.remap(im.numpy(), mapx, mapy, cv2.INTER_CUBIC) 
        irregular_value=np.clip(irregular_value,0,1)
        #将坐标值映射到[-1,1]
        mapx= mapx.reshape(h_lr *w_lr,1)
        mapy= mapy.reshape(h_lr *w_lr,1)
        if self.normalize:
            mapx=coordinate_normalize(mapx,im.shape[1])
            mapy=coordinate_normalize(mapy,im.shape[0])
        #拼接 注意拼接顺序(mapx——>width)(mapy——>height)
        irregular_input=np.concatenate([mapx,mapy,irregular_value.reshape(h_lr *w_lr,3)],axis=1)
        irregular_input=torch.tensor(irregular_input)


        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': irregular_input,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb,
            'hr_h': crop_hr.shape[1],
            'hr_w': crop_hr.shape[2],
            
        }
        
#用于测试亚像素点输入,任意分辨率重建
@register('irregular-sampled-test-arbitrary')
class IrregularSampled_test_arbitrary(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None, normalize=True,seed=None,scale_large=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q
        self.normalize=normalize
        self.seed =seed
        self.scale_large=scale_large
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        s = random.uniform(self.scale_min, self.scale_max)#乘的比例的平方

        
        h_lr = math.ceil(img.shape[-2] *math.sqrt(s) + 1e-9)
        w_lr = math.ceil(img.shape[-1] *math.sqrt(s) + 1e-9)

        crop_hr = img

        im=crop_hr.permute(1,2,0)#[h,w,3]
        
        hr_h=int(img.shape[-2]*self.scale_large)
        hr_w=int(img.shape[-1]*self.scale_large)
        hr_coord=make_coord((hr_h, hr_w))
        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / hr_h
        cell[:, 1] *= 2 / hr_w
        # hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())
        
        #seed
        if self.seed is not None:
            np.random.seed(self.seed)
        mapx = np.random.uniform(-0.5, im.shape[1]-0.5, (h_lr ,w_lr))
        mapx=mapx.astype(np.float32)
        if self.seed is not None:
            np.random.seed(self.seed+1)
        mapy = np.random.uniform(-0.5, im.shape[0]-0.5, (h_lr ,w_lr))
        mapy=mapy.astype(np.float32)
        #亚像素值
        irregular_value = cv2.remap(im.numpy(), mapx, mapy, cv2.INTER_CUBIC) 
        irregular_value=np.clip(irregular_value,0,1)
        #将坐标值映射到HR
        mapx= mapx.reshape(h_lr *w_lr,1)
        mapy= mapy.reshape(h_lr *w_lr,1)
        #将LR坐标投影到HR网格上作为输入
        #测试时不用归一化，将LR坐标点映射到HR即可 map_coord是先row 再col
        map_coord= LR_coord_map_HR(crop_hr.shape[-2:],torch.from_numpy(np.concatenate([mapy,mapx],axis=-1)),ranges=[(-0.5,hr_h-0.5),(-0.5,hr_w-0.5)])
        if self.normalize:
            mapx=coordinate_normalize(mapx,im.shape[1])
            mapy=coordinate_normalize(mapy,im.shape[0])
        #拼接 注意拼接顺序(mapx——>width)(mapy——>height)
        irregular_input=np.concatenate([map_coord[:,1][...,np.newaxis],map_coord[:,0][...,np.newaxis],irregular_value.reshape(h_lr *w_lr,3)],axis=1)
        irregular_input=torch.tensor(irregular_input)
        
        hr_rgb=torch.ones_like(hr_coord)

        return {
            'inp': irregular_input,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb,
            'hr_h': hr_h,
            'hr_w': hr_w,
            
        }


    
#RRS任务数据生成
@register('regular-sampled')
class RegularSampled(Dataset):

    def __init__(self, dataset, inp_size=None, ratio_min=0, ratio_max=None,
                 augment=False, sample_q=None, normalize=True,seed=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.ratio_min = ratio_min
        if ratio_max is None:
            ratio_max = ratio_min
        self.ratio_max = ratio_max
        self.augment = augment
        self.sample_q = sample_q
        self.normalize=normalize
        self.seed=seed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        data_name=self.dataset.files[idx].split("/")[-1].split(".")[0]
        s = random.uniform(self.ratio_min, self.ratio_max)

        if self.inp_size is None:
            h = img.shape[-2]
            w = img.shape[-1]
            #采样点数
            sample_points=round(h*w*s)
            crop_hr = img#[3,h,w] 0~1
        else:
            w_lr = self.inp_size
            h_lr = self.inp_size
            w_hr = round(w_lr * s)
            x0 = random.randint(0, img.shape[-2] - w_hr)
            y0 = random.randint(0, img.shape[-1] - w_hr)
            crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            # crop_lr = resize_fn(crop_hr, w_lr)
            s=w_hr/w_lr

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            # crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)
        
        # im=crop_hr.permute(1,2,0)
        #生成LR GT
        # width=48
        # height=48
        # mapx = np.array([[i for i in range(width)] for j in range(height)], dtype=np.float32)
        # mapx=mapx*s+s/2-0.5
        # mapy = np.array([[j for i in range(width)] for j in range(height)], dtype=np.float32)
        # mapy=mapy*s+s/2-0.5
        # LR = cv2.remap(im.numpy(), mapx, mapy, cv2.INTER_CUBIC)  # 复制
        # LR=torch.tensor(LR)
        # LR.clamp_(0,1)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())#[h*w,2] hr_rgb:0~1
        #记得删除，用于测试
        np.random.seed(self.seed)
        assert self.seed is not None
        regular_coordinate=np.zeros([sample_points,2])
        #选规则输入点
        point_idxs=np.random.choice(np.arange(h*w), sample_points, replace=False)
        regular_coordinate[:,0]=point_idxs%w #col
        regular_coordinate[:,1]=point_idxs//w #row
        regular_value=hr_rgb[point_idxs,:]
        #画出输入图像并保存
        # input=torch.zeros([3,h,w])
        # mask=torch.zeros([3,h,w])
        # input[...,regular_coordinate[:,1],regular_coordinate[:,0]]=crop_hr[...,regular_coordinate[:,1],regular_coordinate[:,0]]
        # mask[...,regular_coordinate[:,1],regular_coordinate[:,0]]=1
        # # inp_save_name="./test_inp_oup/"+str(s)+"/input/input_"+str(idx)+".png"
        # # mask_save_name="./test_inp_oup/"+str(s)+"/input/mask_"+str(idx)+".png"
        # inp_save_name=str(idx)+"_inp.png"
        # mask_save_name=str(idx)+"_mask.png"
        # transforms.ToPILImage()(input).save(inp_save_name)
        # transforms.ToPILImage()(mask).save(mask_save_name)

        regular_input=np.concatenate([regular_coordinate,regular_value],axis=1)#[sample_points,5]
        regular_input=torch.tensor(regular_input)

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': regular_input,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb,
            'hr_h': crop_hr.shape[1],
            'hr_w': crop_hr.shape[2],
            'data_name':data_name
            
        }
    

#RRS+SR任务
@register('regular-sampled-and-map')
class RegularSampledandMap(Dataset):

    def __init__(self, dataset, ratio=1,inp_size=None, augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q
        self.ratio=ratio

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]

        s = img_hr.shape[-2] // img_lr.shape[-2] # assume int scale
        if self.inp_size is None:
            h_lr, w_lr = img_lr.shape[-2:]
            #采样点数
            sample_points=round(h_lr*w_lr*self.ratio)

            img_hr = img_hr[:, :h_lr * s, :w_lr * s]
            crop_lr, crop_hr = img_lr, img_hr
        else:
            w_lr = self.inp_size
            x0 = random.randint(0, img_lr.shape[-2] - w_lr)
            y0 = random.randint(0, img_lr.shape[-1] - w_lr)
            crop_lr = img_lr[:, x0: x0 + w_lr, y0: y0 + w_lr]
            w_hr = w_lr * s
            x1 = x0 * s
            y1 = y0 * s
            crop_hr = img_hr[:, x1: x1 + w_hr, y1: y1 + w_hr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        lr_coord, lr_rgb = to_pixel_samples(crop_lr.contiguous())
        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())
        #对crop_lr采样
        np.random.seed(5)
        regular_coordinate=np.zeros([sample_points,2])
        ####选规则输入点
        point_idxs=np.random.choice(np.arange(h_lr*w_lr), sample_points, replace=False)
        regular_coordinate[:,1]=point_idxs%w_lr #col
        regular_coordinate[:,0]=point_idxs//w_lr #row
        map_rgb=lr_rgb[point_idxs,:] #[sample_points,3]

        #将LR坐标投影到HR网格上作为输入
        #测试时不用归一化，将LR坐标点映射到HR即可 map_coord是先row 再col
        map_coord= LR_coord_map_HR(crop_lr.shape[-2:],torch.from_numpy(regular_coordinate),ranges=[(-0.5,crop_hr.shape[-2]-0.5),(-0.5,crop_hr.shape[-1]-0.5)])
        #注意xy顺序，先拼接x(col),再拼接y(row)
        irregular_input=torch.concat([map_coord[:,1].unsqueeze(dim=1),map_coord[:,0].unsqueeze(dim=1),map_rgb],dim=1)

        # #将crop_lr变为不规则输入,
        # #测试时不用归一化，将LR坐标点映射到HR即可
        # lr_coord, lr_rgb = to_pixel_samples_unnormalize(crop_lr.contiguous(),range=[(-0.5,crop_hr.shape[-2]-0.5),(-0.5,crop_hr.shape[-1]-0.5)])
        # #注意xy顺序，先拼接x,再拼接y
        # irregular_input=torch.concat([lr_coord[:,1].unsqueeze(dim=1),lr_coord[:,0].unsqueeze(dim=1),lr_rgb],dim=1)

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': irregular_input,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb,
            'hr_h': crop_hr.shape[1],
            'hr_w': crop_hr.shape[2],
            
        }