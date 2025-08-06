import sys
sys.path.append(".\\..\\")

import torch.nn as nn
import torch
from models import register
from models.point_utils.gradient_utils_v2 import IRRETORE,regular_xy_normalize,GradientCalculation_CP_Delaunay_weight,ImplicitFeatureExtract_umbrella_v2


@register('irrtr_a13')
class IRRTR_a13(nn.Module):

    def __init__(self,npoint, knn_k,knn_k_prime, in_channel,batch,inp_size,outchannel):
        super().__init__()


        regular_coordinate=regular_xy_normalize(inp_size,inp_size)
        regular_coordinate=regular_coordinate.unsqueeze(0)
        regular_coordinate=regular_coordinate.repeat(batch, 1, 1)
        self.regular_coordinate=regular_coordinate.cuda()

        self.gradientcalculation =GradientCalculation_CP_Delaunay_weight(knn_k,batch,npoint)
        self.sf1 = ImplicitFeatureExtract_umbrella_v2(npoint,batch, nsample=knn_k, in_channel=in_channel, pos_channel=0, mlp=[64, 64, 128], group_all=False,
                 return_normal=False, return_polar=False, cuda=False)

        self.retrans=IRRETORE(npoint, batch, nsample=knn_k_prime,in_channel=144, outchannel=outchannel,inp_size=inp_size, mlp=[128],color_mlp=[128], group_all=False,
                 return_normal=False, return_polar=False, cuda=False)


    def forward(self,coordinate,value):

        gradient,idx,group_all,sort_idx=self.gradientcalculation(coordinate,value)
        
        
        feature1 = self.sf1(gradient, group_all,sort_idx,idx)
        

        out=self.retrans(self.regular_coordinate,coordinate, value, gradient, feature1.permute(0,2,1))
        
        

        return out,gradient
    
    



