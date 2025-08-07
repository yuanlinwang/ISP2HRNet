"""
Author: wyl
modify from :https://github.com/hancyran/RepSurf/blob/main/classification/modules/pointnet2_utils.py
"""

import torch
def square_distance(src, dst):
    """
    Calculate Squared distance between each two points.

    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return torch.abs(dist)




def index_points(points, idx, cuda=False, is_group=False,batch_indices=None):
    
    new_points = points[batch_indices, idx, :]
    return new_points


def square_distance_double(src, dst):
    """
    Calculate Squared distance between each two points. in double type

    """
    src = src.double()
    dst = dst.double()
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return torch.abs(dist)

def query_knn_point_double(k, xyz, new_xyz, cuda=False):
    # if cuda:
    #     if not xyz.is_contiguous():
    #         xyz = xyz.contiguous()
    #     if not new_xyz.is_contiguous():
    #         new_xyz = new_xyz.contiguous()
    #     return knnquery(k, xyz, new_xyz)
    dist = square_distance_double(new_xyz, xyz)
    group_idx = dist.sort(descending=False, dim=-1)[1][:, :, :k]
    #判断是否会出现非中心点出现index 0处的情况
    index = torch.arange(2304).unsqueeze(dim=0).repeat(new_xyz.shape[0],1).cuda()
    assert group_idx[:,:,0].equal(index)
    return group_idx

def query_knn_point(k, xyz, new_xyz, cuda=False):
    # if cuda:
    #     if not xyz.is_contiguous():
    #         xyz = xyz.contiguous()
    #     if not new_xyz.is_contiguous():
    #         new_xyz = new_xyz.contiguous()
    #     return knnquery(k, xyz, new_xyz)
    dist = square_distance(new_xyz, xyz)
    group_idx = dist.sort(descending=False, dim=-1)[1][:, :, :k]
    return group_idx

def cal_normal(group_xyz, random_inv=False, is_group=False):
    """
    Calculate Normal Vector (Unit Form + First Term Positive)

    :param group_xyz: [B, N, K=3, 3] / [B, N, G, K=3, 3]
    :param random_inv:
    :param return_intersect:
    :param return_const:
    :return: [B, N, 3]
    """
    edge_vec1 = group_xyz[..., 1, :] - group_xyz[..., 0, :]  # [B, N, 3] [16,2304,5,3]
    edge_vec2 = group_xyz[..., 2, :] - group_xyz[..., 0, :]  # [B, N, 3]

    nor = torch.cross(edge_vec1, edge_vec2, dim=-1)#[16,2304,5,3]
    
    
    unit_nor = nor / torch.norm(nor, dim=-1, keepdim=True)  # [B, N, 3] / [B, N, G, 3] [16,2304,5,3]
    # check nan
    idx = torch.norm(nor, dim=-1, keepdim=True) == 0
    gradient=unit_nor[...,0:2]#[16,2304,5,2]
    idx=torch.cat((idx,idx),dim=-1)
    gradient[idx]=0

    # if not is_group:
    #     pos_mask = (unit_nor[..., 0] > 0).float() * 2. - 1.  # keep x_n positive
    # else:
    #     pos_mask = (unit_nor[..., 0:1, 0] > 0).float() * 2. - 1. #[-1,1]
    # unit_nor = unit_nor * pos_mask.unsqueeze(-1) #第1维梯度值为正值

    # batch-wise random inverse normal vector (prob: 0.5)
    if random_inv:
        random_mask = torch.randint(0, 2, (group_xyz.size(0), 1, 1)).float() * 2. - 1.
        random_mask = random_mask.to(unit_nor.device)
        if not is_group:
            unit_nor = unit_nor * random_mask
        else:
            unit_nor = unit_nor * random_mask.unsqueeze(-1)

    return gradient

def cal_normal_3d(group_xyz, random_inv=False, is_group=False):
    """
    Calculate Normal Vector (Unit Form + First Term Positive)

    :param group_xyz: [B, N, K=3, 3] / [B, N, G, K=3, 3]
    :param random_inv:
    :param return_intersect:
    :param return_const:
    :return: [B, N, 3]
    # return 3D gradient
    """#group_xyz[16,2304,6,3,3]
    edge_vec1 = group_xyz[..., 1, :] - group_xyz[..., 0, :]  # [B, N, 3] [16,2304,6,3]
    edge_vec2 = group_xyz[..., 2, :] - group_xyz[..., 0, :]  # [B, N, 3]

    nor = torch.cross(edge_vec1, edge_vec2, dim=-1)#[16,2304,6,3]
     
    unit_nor = nor / torch.norm(nor, dim=-1, keepdim=True)  # [B, N, 3] / [B, N, G, 3] [16,2304,6,3]
    # check nan
    idx = torch.norm(nor, dim=-1, keepdim=True) == 0 #[16,2304,6,1]
    gradient=unit_nor#[16,2304,5,3]
    idx=torch.cat((idx,idx,idx),dim=-1)#[16,2304,6,3]
    gradient[idx]=0

    return gradient

def cal_area(group_xyz):
    """
    Calculate Area of Triangle

    :param group_xyz: [B, N, K, 3] / [B, N, G, K, 3]; K = 3
    :return: [B, N, 1] / [B, N, G, 1]
    """
    pad_shape = group_xyz[..., 0, None].shape
    det_xy = torch.det(torch.cat([group_xyz[..., 0, None], group_xyz[..., 1, None], torch.ones(pad_shape).cuda()], dim=-1))#
    det_yz = torch.det(torch.cat([group_xyz[..., 1, None], group_xyz[..., 2, None], torch.ones(pad_shape).cuda()], dim=-1))
    det_zx = torch.det(torch.cat([group_xyz[..., 2, None], group_xyz[..., 0, None], torch.ones(pad_shape).cuda()], dim=-1))
    area = torch.sqrt(det_xy ** 2 + det_yz ** 2 + det_zx ** 2).unsqueeze(-1)
    return area

def cal_area_2D(group_xyz):
    """
    Calculate Area of Triangle

    :param group_xyz: [B, N, K, 3] / [B, N, G, K, 3]; K = 3
    :return: [B, N, 1] / [B, N, G, 1]
    """
    pad_shape = group_xyz[..., 0, None].shape
    det_xy = torch.det(torch.cat([group_xyz, torch.ones(pad_shape).cuda()], dim=-1))#[16,2304,5]
    # det_yz = torch.det(torch.cat([group_xyz[..., 1, None], group_xyz[..., 2, None], torch.ones(pad_shape).cuda()], dim=-1))
    # det_zx = torch.det(torch.cat([group_xyz[..., 2, None], group_xyz[..., 0, None], torch.ones(pad_shape).cuda()], dim=-1))
    area = 0.5*torch.abs(det_xy)
    return area



def cal_area_2D(group_xyz,pad):
    """
    Calculate Area of Triangle

    :param group_xyz: [B, N, K, 3] / [B, N, G, K, 3]; K = 3
    :return: [B, N, 1] / [B, N, G, 1]
    """
    # pad_shape = group_xyz[..., 0, None].shape
    det_xy = torch.det(torch.cat([group_xyz, pad], dim=-1))#[16,2304,5]
    # det_yz = torch.det(torch.cat([group_xyz[..., 1, None], group_xyz[..., 2, None], torch.ones(pad_shape).cuda()], dim=-1))
    # det_zx = torch.det(torch.cat([group_xyz[..., 2, None], group_xyz[..., 0, None], torch.ones(pad_shape).cuda()], dim=-1))
    area = 0.5*torch.abs(det_xy)
    return area
