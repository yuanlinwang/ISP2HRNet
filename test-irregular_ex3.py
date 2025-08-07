import argparse
import os
import math
from functools import partial

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils
import numpy as np
from utils import make_coord,coordinate_normalize,calc_psnr_Y,compute_psnr_uint8
from torchvision import transforms
from skimage.metrics import structural_similarity as compute_ssim
from PIL import Image

def batched_predict_irregular_test(model, coordinate,value, coord, cell, bsize, ratio,hr_h,hr_w):   
    #将图划分为多个patch,组成batch进行运算
    #1.计算取多大的patch能得到48*48的点 
    hr_w=int(hr_w.cpu().numpy())
    hr_h=int(hr_h.cpu().numpy())
    
    # recons_size=48
    # patch_points=2304
    recons_size=48 #点多了 计算k近邻的矩阵就太大
    patch_points=recons_size*recons_size
    # patch_size=int(np.ceil(recons_size/np.sqrt(ratio)))
    patch_size=int(np.ceil(48.0/np.sqrt(ratio)))
    stride=np.round(patch_size/2)#patch移动步长
    # stride=patch_size-32#patch移动步长
    
    grid_x=int(np.ceil(float(hr_w - patch_size) / stride) + 1)
    grid_y=int(np.ceil(float(hr_h - patch_size) / stride) + 1)
    padding=0.5
    BATCH_SIZE=16
    #cell归一化
    hr_coord = make_coord([patch_size,patch_size])
    torch_hr = torch.Tensor(hr_coord)
    torch_hr = torch_hr.float().cuda()

    cell_local = torch.ones_like(hr_coord)
    cell_local[:, 0] *= 2 / patch_size
    cell_local[:, 1] *= 2 / patch_size
    torch_cell = torch.Tensor(cell_local)
    torch_cell = torch_cell.float().cuda()
    #2.循环每一个patch，得到对应的离散坐标点
    coordinate_room,value_room = np.array([]), np.array([])
    value_pool = np.zeros((hr_h,hr_w, 3))
    value_num=np.zeros((hr_h,hr_w))
    value_avg=np.zeros((hr_h,hr_w, 3))

    coordinate=coordinate.squeeze(0).cpu().numpy()
    value=value.squeeze(0).cpu().numpy()
    

    for index_y in range(0, grid_y):
        for index_x in range(0, grid_x):
            s_x = index_x * stride
            e_x = min(s_x + patch_size, hr_w)
            s_x = e_x - patch_size
            s_y = index_y * stride
            e_y = min(s_y + patch_size, hr_h)
            s_y = e_y - patch_size
            point_idxs = np.where((coordinate[..., 0] >= s_x - padding) & (coordinate[..., 0] <= e_x - 1 + padding) & (coordinate[..., 1] >= s_y - padding) & \
                                                            (coordinate[..., 1] <= e_y - 1 + padding))[0]
            if point_idxs.size == 0:
                print("采样不均匀")
                continue
            num_batch = 1
            point_size = int(num_batch * patch_points)
            if point_size>point_idxs.size :
                replace = False if (point_size - point_idxs.size <= point_idxs.size) else True
                point_idxs_repeat = np.random.choice(point_idxs, point_size - point_idxs.size, replace=replace)
                point_idxs = np.concatenate((point_idxs, point_idxs_repeat))
            elif point_size<point_idxs.size:
                point_idxs=np.random.choice(point_idxs, point_size, replace=False)
            # np.random.shuffle(point_idxs)
            data_batch = coordinate[point_idxs, :]
            value_batch = value[point_idxs,:]
            
            #对data_batch进行归一化
            data_batch[:,0]=data_batch[:,0]-s_x
            data_batch[:,1]=data_batch[:,1]-s_y
            
            data_batch[:,0]=coordinate_normalize(data_batch[:,0],patch_size)
            data_batch[:,1]=coordinate_normalize(data_batch[:,1],patch_size)

            coordinate_room = np.vstack([coordinate_room, data_batch]) if coordinate_room.size else data_batch
            value_room = np.vstack([value_room, value_batch]) if value_room.size else value_batch

    coordinate_room = coordinate_room.reshape((-1, patch_points, coordinate_room.shape[1]))
    value_room = value_room.reshape((-1, patch_points, value_room.shape[1]))

    #3.按batch对patches进行处理
    num_blocks = coordinate_room.shape[0]
    s_batch_num = (num_blocks + BATCH_SIZE - 1) // BATCH_SIZE

    batch_coordinate = np.zeros((BATCH_SIZE, patch_points, 2))
    batch_value = np.zeros((BATCH_SIZE, patch_points,3))
    preds = torch.tensor([]).cuda()
    initials = torch.tensor([]).cuda()
    gradients=torch.tensor([]).cuda()
    gradients_coord=torch.tensor([]).cuda()
    with torch.no_grad():
        for sbatch in range(s_batch_num):
            start_idx = sbatch * BATCH_SIZE
            end_idx = min((sbatch + 1) * BATCH_SIZE, num_blocks)
            real_batch_size = end_idx - start_idx
            batch_coordinate[0:real_batch_size, ...] = coordinate_room[start_idx:end_idx, ...]
            batch_value[0:real_batch_size, ...] = value_room[start_idx:end_idx, ...]

            # batch_coordinate = coordinate_room[start_idx:end_idx, ...]
            # batch_value = value_room[start_idx:end_idx, ...]

            torch_coordinate = torch.Tensor(batch_coordinate)
            torch_coordinate = torch_coordinate.float().cuda()
            torch_value = torch.Tensor(batch_value)
            torch_value = torch_value.float().cuda()

            # pred = model(torch_coordinate,torch_value,torch_hr.unsqueeze(0).repeat(real_batch_size,1,1),
            #                 torch_cell.unsqueeze(0).repeat(real_batch_size,1,1))
            pred,initial,gradient = model(torch_coordinate,torch_value,torch_hr.unsqueeze(0).repeat(BATCH_SIZE,1,1),
                            torch_cell.unsqueeze(0).repeat(BATCH_SIZE,1,1))

            preds=torch.cat([preds,pred],dim=0)
            initials=torch.cat([initials,initial[:,0:3,:,:]],dim=0)#
            gradients=torch.cat([gradients,gradient.reshape(-1,6)],dim=0)
            gradients_coord=torch.cat([gradients_coord,torch_coordinate.reshape(-1,2)],dim=0)
    #4.将预测值映射到HR图像上
    preds=preds.cpu().numpy()
    batch_num=0
    for index_y in range(0, grid_y):
        for index_x in range(0, grid_x):
            s_x = index_x * stride
            e_x = min(s_x + patch_size, hr_w)
            s_x = e_x - patch_size
            s_y = index_y * stride
            e_y = min(s_y + patch_size, hr_h)
            s_y = e_y - patch_size
            #对边缘像素给予少的权重
            border=2
            weight=np.ones((patch_size,patch_size))*0.2
            weight[border:-border,border:-border]=1
            pred_cur=preds[batch_num].reshape(patch_size,patch_size,3)
            weight_3=np.repeat(weight[...,np.newaxis],3,axis=2)
            pred_cur=pred_cur*weight_3
            # value_pool[int(s_y):int(e_y),int(s_x):int(e_x),:] += preds[batch_num].reshape(patch_size,patch_size,3)
            # value_num[int(s_y):int(e_y),int(s_x):int(e_x)] +=1
            value_pool[int(s_y):int(e_y),int(s_x):int(e_x),:] += pred_cur
            value_num[int(s_y):int(e_y),int(s_x):int(e_x)] +=weight
            batch_num +=1
    for i in range(3):
        value_avg[:,:,i]=value_pool[:,:,i]/value_num
    value_avg=torch.from_numpy(value_avg).cuda()
    #5. 将初始重建值映到HR图像上
    h_lr = math.ceil(hr_h *math.sqrt(ratio) + 1e-9)
    w_lr = math.ceil(hr_w *math.sqrt(ratio) + 1e-9)
    value_pool_initial = np.zeros((h_lr,w_lr, 3))
    value_num_initial=np.zeros((h_lr,w_lr))
    value_avg_initial=np.zeros((h_lr,w_lr, 3))
    initials=initials.permute(2,3,1,0).cpu().numpy()
    batch_num=0
    stride=24
    patch_size=48
    for index_y in range(0, grid_y):
        for index_x in range(0, grid_x):
            s_x = index_x * stride
            e_x = min(s_x + patch_size, w_lr)
            s_x = e_x - patch_size
            s_y = index_y * stride
            e_y = min(s_y + patch_size,h_lr)
            s_y = e_y - patch_size

            value_pool_initial[int(s_y):int(e_y),int(s_x):int(e_x),:] += initials[...,batch_num]
            value_num_initial[int(s_y):int(e_y),int(s_x):int(e_x)] +=1
            
            batch_num +=1
    for i in range(3):
        value_avg_initial[:,:,i]=value_pool_initial[:,:,i]/value_num_initial#[h,w,3]
    
    
    return value_avg

def eval_psnr_irregular_test(loader, model,scale=None, data_norm=None, eval_type=None, eval_bsize=None,
              verbose=False,out_save=None,ratio=None,seed=None):
    model.eval()
    ratio=scale
    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, 1,-1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    if eval_type is None:
        metric_fn = utils.calc_psnr
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='benchmark', scale=scale)
    else:
        raise NotImplementedError

    val_res = utils.Averager()
    val_res_2 = utils.Averager()
    pbar = tqdm(loader, leave=False, desc='val')
    idx=0
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()

        coordinate=batch['inp'][:,:,:2] #[1,sample_points,5]
        value=batch['inp'][:,:,2:5]
        value = (value - inp_sub) / inp_div
        if eval_bsize is None:
            with torch.no_grad():
                pred = model(coordinate,value, batch['coord'], batch['cell'])
        else:
            pred = batched_predict_irregular_test(model, coordinate,value,
                batch['coord'], batch['cell'], eval_bsize,ratio,batch['hr_h'],batch['hr_w'])
        pred = pred * gt_div + gt_sub
        pred.clamp_(0, 1) #[h,w,3]

        if eval_type is not None: # reshape for shaving-eval
            # ih, iw = batch['inp'].shape[-2:]
            # s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
            # shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            # pred = pred.view(*shape) \
            #     .permute(0, 3, 1, 2).contiguous()
            pred=pred.unsqueeze(0).permute(0, 3, 1, 2).contiguous()#[1,3,h,w]
            shape = [1, batch['hr_h'], batch['hr_w'], 3]
            batch['gt'] = batch['gt'].view(*shape) \
                .permute(0, 3, 1, 2).contiguous() #[1,3,h,w]
        else:
            pred=pred.reshape([1,batch['hr_h']*batch['hr_w'],3])
        # res_2 = metric_fn(pred, batch['gt'])
        # res=metric_fn(pred, batch['gt'])

        #转为uint8后比较
        pred=(pred.squeeze().permute(1,2,0).cpu().numpy()*255).astype(np.uint8) #[3,h,w]
        GT=(batch['gt'].squeeze().permute(1,2,0).squeeze().cpu().numpy()*255).astype(np.uint8) #[3,h,w]
        ########
        psnr=compute_psnr_uint8(pred, GT) 
        ssim=compute_ssim(pred, GT, data_range=255, channel_axis=-1)
        #保存输出图像
        # out_save_name=os.path.join(out_save,str(idx)+"_"+str(ratio)+".png")
        # Image.fromarray(pred).save(out_save_name)
        
        # val_res.add(res.item(), batch['inp'].shape[0])


        val_res.add(psnr, batch['inp'].shape[0])
        val_res_2.add(ssim, batch['inp'].shape[0])
        if verbose:
            pbar.set_description('val {:.4f}'.format(val_res.item()))
        idx=idx+1
    print(val_res_2.item())
    return val_res.item(),val_res_2.item()



#EX3
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=str, default="configs/test-irregular/test-div2k-2_draw.yaml")
    parser.add_argument('--model',type=str, default="save/a13_modify/epoch-1000.pth")
    parser.add_argument('--gpu', default='6')
    parser.add_argument('--ratio', type=float, default=0.2)
    parser.add_argument('--dataset_name', type=str, default='set5')
    parser.add_argument('--root',type=str, default="/home/ylwang/dataset/benchmark/Set5/HR/")
    parser.add_argument('--seed',type=int, default=5)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config['test_dataset']['wrapper']['args']['scale_min']=args.ratio
    config['test_dataset']['dataset']['args']['root_path']=args.root
    config['test_dataset']['wrapper']['args']['seed']=args.seed
    
    #保存图片路径
    if not os.path.exists("EX3"):
        os.mkdir("EX3")
    out_save=os.path.join("EX3", args.dataset_name)
    if not os.path.exists(out_save):
        os.mkdir(out_save)

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        num_workers=8, pin_memory=False)

    model_spec = torch.load(args.model)['model']
    model_spec['args']['irregular_spec']['args']['knn_k_prime']=12
    model = models.make(model_spec, load_sd=True).cuda()

    psnr,ssim = eval_psnr_irregular_test(loader, model,scale=config.get('test_dataset').get('wrapper').get('args').get('scale_min'),
        data_norm=config.get('data_norm'),
        eval_type=config.get('eval_type'),
        eval_bsize=config.get('eval_bsize'),
        verbose=True,out_save=out_save,ratio=args.ratio,seed=args.seed)

    print( 'PSNR avg:%.2f, SSIM avg:%.4f'%(psnr,ssim))
    file = open('EX3.txt', 'a')
    print( '%s      ———ratio:%.2f'%(args.dataset_name,args.ratio),file=file)
    print( 'PSNR avg:%.2f, SSIM avg:%.4f\n'%(psnr,ssim),file=file)
    
