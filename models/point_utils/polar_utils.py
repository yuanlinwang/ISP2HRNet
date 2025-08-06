"""
Author: Haoxi Ran
Date: 05/10/2022
"""

import torch
import numpy as np


def xyz2sphere(xyz, normalize=True):
    """
    Convert XYZ to Spherical Coordinate

    reference: https://en.wikipedia.org/wiki/Spherical_coordinate_system

    :param xyz: [B, N, 3] / [B, N, G, 3]
    :return: (rho, theta, phi) [B, N, 3] / [B, N, G, 3]
    """
    # rho = torch.sqrt(torch.sum(torch.pow(xyz, 2), dim=-1, keepdim=True))
    # rho = torch.clamp(rho, min=0)  # range: [0, inf] #[batch,nsample,k_sample,1]
    # theta = torch.acos(xyz[..., 2, None] / rho)  # range: [0, pi]
    phi = torch.atan2(xyz[..., 1, None], xyz[..., 0, None])  # range: [-pi, pi] [batch,nsample,k_sample,1]
    # check nan
    # idx = rho == 0
    # theta[idx] = 0

    if normalize:
        # theta = theta / np.pi  # [0, 1]
        phi = phi / (2 * np.pi) + .5  # [0, 1]
    # out = torch.cat([rho, phi], dim=-1) #[batch,nsample,k_sample,2]
    return phi