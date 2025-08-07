import torch
from torch import nn
from models.point_utils.pointnet2_utils import query_knn_point, index_points,\
    cal_area_2D,cal_normal_3d
import torch.nn.functional as F
from utils import coordinate_normalize
import numpy as np
import copy
import torch.nn.functional as F
from models.point_utils.polar_utils import xyz2sphere

def regular_xy_normalize(width,height):
    '''
    normalize regular coordinates

    '''
    src_x = np.array([[i for i in range(width)] for j in range(height)], dtype=np.float32)
    src_x=src_x.reshape(width*height,1)
    src_y = np.array([[j for i in range(width)] for j in range(height)], dtype=np.float32)
    src_y=src_y.reshape(width*height,1)
    
    coordinate_nor_x=torch.tensor(coordinate_normalize(src_x,width))
    coordinate_nor_y=torch.tensor(coordinate_normalize(src_y,height))

    regular_xy=torch.cat([coordinate_nor_x,coordinate_nor_y],dim=-1)

    return regular_xy

def sample_and_group(k, src,dst, value,gradient, feature,idx=None,src_value=False, \
                     return_normal=False, return_polar=False, cuda=False,batch_indices=None):
   
    # src=src.to(dst.device)
    # group
    if idx==None: #整像素位置处的搜索
        idx = query_knn_point(k,dst,src, cuda=cuda)
    else:
        idx=idx[:, :, :k]

    #中心点绝对坐标,绝对像素值
    # absolute coordinate and pixel value of the center pixel
    center_xy=src.unsqueeze(dim=-2).repeat(1,1,k-1,1)  # [B, npoint, nsample, 2]
    center_value=value.unsqueeze(dim=-2).repeat(1,1,k-1,1) # [B, npoint, nsample, 3]
    center_gradient=gradient.unsqueeze(dim=-2).repeat(1,1,k-1,1)
    # group coordinate
    group_xy = index_points(dst, idx, cuda=cuda, is_group=True,batch_indices=batch_indices)  # [B, npoint, nsample, 2]
    
    # group value
    group_value = index_points(value, idx, cuda=cuda, is_group=True,batch_indices=batch_indices)  # [B, npoint, nsample, 3]
    
    # group gradient
    group_gradient = index_points(gradient, idx, cuda=cuda, is_group=True,batch_indices=batch_indices)  # [B, npoint, nsample, 6]
    

    if src_value:
        relative_value=group_value-value.unsqueeze(2)# [B, N, K, 3]
        relative_value=relative_value[:,:,1:,:]
        group_xy =group_xy[:,:,1:,:] #删除中心点信息
        group_value =group_value[:,:,1:,:]
        group_gradient =group_gradient[:,:,1:,:]

    #计算相对坐标,相对像素值差异，欧式距离
    #Compute the relative coordinates, relative pixel value differences, and Euclidean distances.
    relative_xyz=group_xy-src.unsqueeze(2)# [B, N, K, 2]
  
    distance=relative_xyz.norm(p=2,dim=3).unsqueeze(3)# [B, N, K, 1]

    if feature is not None:
        center_feature=feature.unsqueeze(dim=-2).repeat(1,1,k-1,1)
        group_feature = index_points(feature, idx, cuda=cuda, is_group=True,batch_indices=batch_indices)
        
        if src_value:
            group_feature=group_feature[:,:,1:,:]
            new_feature = torch.cat([center_xy,center_value,center_gradient,center_feature,
                                     group_xy,group_value,group_gradient,group_feature,relative_xyz,relative_value,distance], dim=-1)
        else:
            center_xy=src.unsqueeze(dim=-2).repeat(1,1,k,1)
            new_feature = torch.cat([center_xy,
                                     group_xy,group_value,group_gradient, group_feature,relative_xyz,distance], dim=-1)
    else:
        if src_value:
            new_feature = torch.cat([center_xy,center_value,center_gradient,
                                     group_xy,group_value,group_gradient,relative_xyz,relative_value,distance], dim=-1)
        else:
            new_feature = torch.cat([center_xy,
                                    group_xy,group_value,group_gradient,relative_xyz,distance], dim=-1)

    return new_feature



def resort_points(points, idx,batch_indices,batch_indices_N):
    """
    Resort Set of points along G dim

    """
    # device = points.device
    B, N, G, _ = points.shape #[16,2304,5,2]

    # view_shape = [B, 1, 1]
    # repeat_shape = [1, N, G]
    # b_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)#[16,2304,5]
    b_indices = batch_indices[...,:-1]
    
    view_shape = [1, N, 1]
    repeat_shape = [B, 1, G]
    # n_indices = torch.arange(N, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    n_indices = batch_indices_N.view(view_shape).repeat(repeat_shape)

    new_points = points[b_indices, n_indices, idx, :]

    return new_points


def group_by_umbrella_v2(xy, value,gradient=None, k=6, cuda=False, idx=None,batch_indices=None,batch_indices_N=None):
    """
    计算三角形顶点的光度和几何信息
    Compute the photometric and geometric information at the triangle's vertices.
    """
    if idx==None: #整像素位置处的搜索
        idx = query_knn_point(k,xy,xy, cuda=cuda)
    else:
        idx=idx[:, :, :k]
    group_xyz = index_points(xy, idx, cuda=cuda, is_group=True,batch_indices=batch_indices)[:, :, 1:]  # [B, N', K-1, 3] #近邻点 不包含中心点 [16,2304,8,2]
    group_value=index_points(value, idx, cuda=cuda, is_group=True,batch_indices=batch_indices)[:, :, 1:] #[16,2304,8,3]

    group_value_norm = group_value - value.unsqueeze(-2)#像素值相对值 relative pixel value
    group_xyz_norm = group_xyz - xy.unsqueeze(-2)#坐标相对值 relative coordinate
    group_phi = xyz2sphere(group_xyz_norm).squeeze() # [16,2304,8]
    # sort_idx = group_phi.argsort(dim=-1)  # [B, N', K-1] #[16,2304,8]

    group_phi,sort_idx=group_phi.sort()
    group_phi_roll= torch.roll(group_phi,-1,dims=-1)

    # Coordinates
    #坐标与其逆时针roll
    # [B, N', K-1, 1, 3] [16,2304,5,1,2]
    sorted_group_xyz = resort_points(group_xyz_norm, sort_idx,batch_indices=batch_indices,batch_indices_N=batch_indices_N)#[16,2304,8,2]
    sorted_group_xyz_roll = torch.roll(sorted_group_xyz, -1, dims=-2)#[16,2304,8,2]
    group_centriod = torch.zeros_like(sorted_group_xyz.unsqueeze(-2))#[16,2304,8,1,2] all=0
    umbrella_group_xyz = torch.cat([group_centriod, sorted_group_xyz.unsqueeze(-2), sorted_group_xyz_roll.unsqueeze(-2)], dim=-2)#[16,2304,8,3,2]
    
    # relative pixel value
    #相对像素值 与逆时针roll
    sorted_group_value = resort_points(group_value_norm, sort_idx,batch_indices=batch_indices,batch_indices_N=batch_indices_N)#[16,2304,8,2]
    sorted_group_value_roll = torch.roll(sorted_group_value, -1, dims=-2)#[16,2304,8,2]
    group_centriod_value = torch.zeros_like(sorted_group_value.unsqueeze(-2))
    umbrella_group_value=torch.cat([group_centriod_value, sorted_group_value.unsqueeze(-2), sorted_group_value_roll.unsqueeze(-2)], dim=-2)#[16,2304,5,3,3]
    
    # relative coordinate
    #绝对坐标值
    sorted_group_xyz_un = resort_points(group_xyz, sort_idx,batch_indices=batch_indices,batch_indices_N=batch_indices_N)#[16,2304,8,2]
    sorted_group_xyz_un_roll = torch.roll(sorted_group_xyz_un, -1, dims=-2)#[16,2304,8,2]

    # absolute pixel value
    #绝对像素值
    sorted_group_value_un = resort_points(group_value, sort_idx,batch_indices=batch_indices,batch_indices_N=batch_indices_N)#[16,2304,8,2]
    sorted_group_value_un_roll = torch.roll(sorted_group_value_un, -1, dims=-2)#[16,2304,8,2]

    EurDis_xyz=torch.norm(sorted_group_xyz,dim=-1).unsqueeze(dim=-1)#[16,2304,8,1]
    EurDis_xyz_roll=torch.norm(sorted_group_xyz_roll,dim=-1).unsqueeze(dim=-1)#[16,2304,8,1]
    # umbrella_group_xyz = torch.cat([sorted_group_xyz,EurDis_xyz, sorted_group_xyz_roll,EurDis_xyz_roll], dim=-1)#[16,2304,5,6]
    
    #[16,2304,8,29]
    umbrella_group_relative=torch.cat([xy.unsqueeze(dim=2).repeat(1,1,k-1,1),value.unsqueeze(dim=2).repeat(1,1,k-1,1),\
        sorted_group_xyz_un,sorted_group_value_un,\
        sorted_group_xyz,sorted_group_value,\
            sorted_group_xyz_un_roll,sorted_group_value_un_roll,\
            sorted_group_xyz_roll,sorted_group_value_roll,EurDis_xyz,EurDis_xyz_roll,\
                group_phi.unsqueeze(3),group_phi_roll.unsqueeze(3)], dim=-1)#[16,2304,8,29]

    
    return umbrella_group_xyz,umbrella_group_value,umbrella_group_relative,sort_idx


    

    
class GradientCalculation_CP_Delaunay_weight(nn.Module):
    """
    Gradient Calculation Module: through cross product
    Area-weighted Gradient Estimation
    """

    def __init__(self, k, batch,npoint, aggr_type='sum', return_dist=False, random_inv=False, cuda=False):
        super(GradientCalculation_CP_Delaunay_weight, self).__init__()
        self.k = k
        self.return_dist = return_dist
        self.random_inv = random_inv
        self.aggr_type = aggr_type
        self.cuda = cuda
        self.batch=batch
        self.npoint=npoint
        self.gradient_c=torch.zeros((batch,npoint,k-1,9)).cuda()
        self.gradient=torch.zeros((batch,npoint,6)).cuda()
        # self.index=nn.Parameter(torch.ones(1)*2)
        
        view_shape = list([batch,npoint,k])
        repeat_shape = list([batch,npoint,k])
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape[0] = 1
        self.batch_indices=torch.arange(batch, dtype=torch.long).cuda().view(view_shape).repeat(repeat_shape)
        
        self.batch_indices_N=torch.arange(npoint, dtype=torch.long).cuda()
        self.pad =torch.ones([batch,npoint,k-1,3,1]).cuda()
    def forward(self,coordinate,value):
        gradient_c=self.gradient_c
        gradient=self.gradient
        
        #Found the K-nearest neighbors (KNN) for each discrete point # [B, N, K, C]
        idx=query_knn_point(self.k, coordinate, coordinate, cuda=self.cuda)

        #K-nearest neighbor coordinates sorted in a counterclockwise direction. [16,2304,6,3,2] 
        #dim3 :一维是中心点坐标，全0  1：逆时针排序的坐标 2：roll后的数据
        #group value[16,2304,6,3,3]
        group_xy,group_value, umbrella_group_all,sort_idx= group_by_umbrella_v2(coordinate, value,gradient=None, k=self.k,idx=idx,\
                                batch_indices=self.batch_indices,batch_indices_N=self.batch_indices_N) 
        
        
        for i in range(3): #Compute the gradient for RGB channels separately.
            group_xyz=torch.cat([group_xy,group_value[...,i].unsqueeze(dim=4)],dim=-1)#[16,2304,6,3,3]
            #normal
            group_gradient = cal_normal_3d(group_xyz, random_inv=False, is_group=True)#[16,2304,6,3]     
            #record
            gradient_c[...,i*3:i*3+3]=group_gradient #归一化了 #[-1,1]
        #Compute the area of each formed triangle.
        area=cal_area_2D(group_xy,self.pad) #[16,2304,8]
        area_norm=torch.sum(area,dim=-1).unsqueeze(dim=-1)#[16,2304,1]
        #check nan
        idx_0 = area_norm == 0
        area_norm[idx_0]=10000
        area_norm=(area/area_norm).unsqueeze(dim=-1)#[16,2304,6,1]
        gradient_c=(gradient_c*area_norm).sum(dim=-2)#[16,2304,9]
       
        max=0
        #channel R
        gradient[...,0]=-gradient_c[...,0]/gradient_c[...,2]
        gradient[...,1]=-gradient_c[...,1]/gradient_c[...,2]
    
        idx_c = gradient_c[...,2] == 0
        if (idx_c==1).sum() !=0:
            gradient[...,0][idx_c]=max
            gradient[...,1][idx_c]=max
            
        #channel G
        gradient[...,2]=-gradient_c[...,3]/gradient_c[...,5]
        gradient[...,3]=-gradient_c[...,4]/gradient_c[...,5]
        idx_c = gradient_c[...,5] == 0
        if (idx_c==1).sum() !=0:
            gradient[...,2][idx_c]=max
            gradient[...,3][idx_c]=max

        #channel B
        gradient[...,4]=-gradient_c[...,6]/gradient_c[...,8]
        gradient[...,5]=-gradient_c[...,7]/gradient_c[...,8]
        idx_c = gradient_c[...,8] == 0
        if (idx_c==1).sum() !=0:
            gradient[...,4][idx_c]=max
            gradient[...,5][idx_c]=max
            
        gradient =gradient /10000.
        
        if torch.any(torch.isnan(gradient)) is True:
            print("error")
        assert not torch.any(torch.isnan(gradient))
        
        #angle
        phi=umbrella_group_all[...,-2:] #[16,2304,8,2]
        sin_angle=torch.abs(torch.sin((phi[:,:,:,1]-phi[:,:,:,0]-0.5)*2*torch.pi))
        
        #collect information
        umbrella_group_all=torch.cat([umbrella_group_all[...,:-2],sin_angle.unsqueeze(dim=-1)],dim=-1)
        
        return gradient,idx,umbrella_group_all,sort_idx



def group_by_umbrella_gradient(group_gradient,sort_idx,batch_indices,batch_indices_N):
    """
    Group a set of points into umbrella surfaces

    """

    #gradient
    sorted_group_gradient = resort_points(group_gradient, sort_idx,batch_indices,batch_indices_N)#[16,2304,8,6]

    sorted_group_gradient_roll = torch.roll(sorted_group_gradient, -1, dims=-2)#[16,2304,8,6]

    umbrella_group_gradient=torch.cat([sorted_group_gradient,sorted_group_gradient_roll], dim=-1)#[16,2304,8,12]

    return umbrella_group_gradient

class ImplicitFeatureExtract_umbrella_v2(nn.Module):
    """
    Implicit Gradient Feature Module

    """
    def __init__(self, npoint, batch, nsample, in_channel, pos_channel, mlp, group_all=False,
                 return_normal=False, return_polar=False, cuda=False):
        super(ImplicitFeatureExtract_umbrella_v2, self).__init__()
        self.npoint = npoint

        self.nsample = nsample
        self.return_normal = return_normal
        self.return_polar = return_polar
        self.cuda = cuda
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.pos_channel = pos_channel
        self.group_all = group_all

        last_channel = in_channel
        for out_channel in mlp[0:]:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
            
        view_shape = list([batch,npoint,nsample])
        repeat_shape = list([batch,npoint,nsample])
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape[0] = 1
        self.batch_indices=torch.arange(batch, dtype=torch.long).cuda().view(view_shape).repeat(repeat_shape)
        
        self.batch_indices_N=torch.arange(npoint, dtype=torch.long).cuda()
    def forward(self,gradient, group_all,sort_idx,idx):
 
        group_gradient=index_points(gradient, idx, is_group=True,batch_indices=self.batch_indices)[:, :, 1:]#[16,2304,8,6]
        
        knn=group_gradient.shape[-2]
        #[16,2304,8,12]
        group_gradient = group_by_umbrella_gradient(group_gradient, sort_idx,self.batch_indices,self.batch_indices_N) 
        #group_all[16,2304,8,28]
        new_feature=torch.cat([group_gradient,group_all,gradient.unsqueeze(dim=-2).repeat(1,1,knn,1)],dim=-1)#[16,2304,8,46]
        new_feature=new_feature.permute(0,3,2,1)  #[B,C,nsample,N] [16,46,8,2304]
        
        for i, conv in enumerate(self.mlp_convs):
                bn = self.mlp_bns[i]
                new_feature = F.relu(bn(conv(new_feature)))
  
        new_feature = torch.mean(new_feature, 2) #[B,C,N]
        return new_feature
    


class IRRETORE(nn.Module):
    """
    Irregular To Regular Module

    """

    def __init__(self, npoint, batch, nsample, in_channel, outchannel,inp_size, mlp,color_mlp, group_all=False,
                 return_normal=False, return_polar=False, cuda=False):
        super(IRRETORE, self).__init__()
        self.npoint = npoint
        self.batch = batch
        self.nsample = nsample
        self.return_normal = return_normal
        self.return_polar = return_polar
        self.cuda = cuda
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.outchannel = outchannel
        self.group_all = group_all
        self.inp_size=inp_size

        self.mlp_convs_color = nn.ModuleList()
        self.mlp_bns_color = nn.ModuleList()
        self.color_mlp=color_mlp
        
        #generate filter
        last_channel = in_channel
        for out_channel in mlp[0:]:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.map2= nn.Conv2d(last_channel, last_channel, 1) 
        self.bn2=nn.BatchNorm2d(last_channel)

        if color_mlp is not None:
            last_channel = in_channel
            for out_channel in color_mlp[0:]:
                self.mlp_convs_color.append(nn.Conv2d(last_channel, out_channel, 1))
                self.mlp_bns_color.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
        

        self.color_map2 = nn.Conv2d(last_channel, 3, 1) 
        
        view_shape = list([batch,npoint,nsample])
        repeat_shape = list([batch,npoint,nsample])
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape[0] = 1
        self.batch_indices=torch.arange(batch, dtype=torch.long).cuda().view(view_shape).repeat(repeat_shape)

    def forward(self, regular_coordinate, xy, value, gradient, feature):
        

        new_feature = sample_and_group(self.nsample, regular_coordinate,xy, value, gradient,feature,idx=None,
                                        src_value=False,
                                        return_polar=self.return_polar,
                                        return_normal=self.return_normal,batch_indices=self.batch_indices)#[B,N,nsample,C]
        
        new_feature=new_feature.permute(0,3,2,1)
        fea_filter=new_feature.clone()
        group_value=new_feature[:,4:7,:,:].clone()
        group_feature=new_feature[:,13:13+feature.shape[2],:,:].clone()
       
        #processed by MLP [batch,channel,nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            fea_filter = F.relu(bn(conv(fea_filter)))
            
        # filter for feature
        fea_filter=self.map2(fea_filter)
        out_feature=torch.mul(group_feature,fea_filter)
        out_feature=F.relu(self.bn2(out_feature))
        out_feature=torch.mean(out_feature,dim=2)

        # filter for value channel 
        #[batch,channel,nsample,npoint]
        for i, conv in enumerate(self.mlp_convs_color):
            bn = self.mlp_bns_color[i]
            new_feature = F.relu(bn(conv(new_feature)))

        new_feature=self.color_map2(new_feature)
        color_filter=F.softmax(new_feature,dim=2)

        out_color=torch.mul(group_value,color_filter)
        out_color=torch.sum(out_color,dim=2)
        out=torch.cat([out_color,out_feature],dim=1)
        

        out=out.view(self.batch,self.outchannel,self.inp_size,self.inp_size)

        return out
