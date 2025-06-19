import torch 
import numpy as np
from typing import Callable
from torch.autograd import Function
import torch.nn.functional as F



class _TruncExp(Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))
trunc_exp = _TruncExp.apply

def get_activation(name) -> Callable:
    if name is None:
        return lambda x: x
    name = name.lower()
    if name == "none":
        return lambda x: x
    elif name == "lin2srgb":
        return lambda x: torch.where(
            x > 0.0031308,
            torch.pow(torch.clamp(x, min=0.0031308), 1.0 / 2.4) * 1.055 - 0.055,
            12.92 * x,
        ).clamp(0.0, 1.0)
    elif name == "exp":
        return lambda x: torch.exp(x)
    elif name == "shifted_exp":
        return lambda x: torch.exp(x - 1.0)
    elif name == "trunc_exp":
        return trunc_exp
    elif name == "shifted_trunc_exp":
        return lambda x: trunc_exp(x - 1.0)
    elif name == "sigmoid":
        return lambda x: torch.sigmoid(x)
    elif name == "tanh":
        return lambda x: torch.tanh(x)
    elif name == "shifted_softplus":
        return lambda x: F.softplus(x - 1.0)
    elif name == "scale_-11_01":
        return lambda x: x * 0.5 + 0.5
    elif name == "relu":
        return lambda x: torch.relu(x)
    else:
        try:
            return getattr(F, name)
        except AttributeError:
            raise ValueError(f"Unknown activation function: {name}")
        

"""
Full projection Function
"""
def ndc2pixel(v, S):
    return ((v + 1.0) * S - 1.0) * 0.5


def fullproj(point_3d, full_proj_matrix, W, H):
    """
    Project the 3D point cloud into pixel space
    Using the 3DGS projection methods: World_coord --> Camera_coord 
    --> NDC_coord--> Pixel_coord 
    
    """
    hom = torch.matmul(point_3d, full_proj_matrix)
    weight = 1.0/(hom[:,3] + 0.000001)
    return ndc2pixel(hom[:,0]*weight, W), ndc2pixel(hom[:,1]*weight, H)


def project_and_filter(points_3d, P, W, H):
     
    N = points_3d.shape[0]
    # Construct homogeneous coordinates [X, Y, Z, 1]
    ones = torch.ones((N, 1), dtype=points_3d.dtype, device=points_3d.device)
    points_homogeneous = torch.cat([points_3d, ones], dim=1)  # [N, 4]
    
    # Project 3D points into pixel space 
    x,y = fullproj(points_homogeneous, P, 640, 480)
    
    # Keep only the projected pixel that is inside the pixel space ：x ∈ (0, 640), y ∈ (0, 480)
    mask = (x > 0) & (x < W) & (y > 0) & (y < H)
    
    x_filtered = x[mask]
    y_filtered = y[mask]
    points_3d_filtered = points_3d[mask]  # [M, 3]
    
    xy = torch.cat([x_filtered.unsqueeze(1), 
                        y_filtered.unsqueeze(1)], dim=1) 
    
    return mask, xy, points_3d_filtered 
    
    
    



     
