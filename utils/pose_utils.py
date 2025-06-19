import numpy as np
import torch


def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def poses_avg(poses):
    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w

def render_path_spiral(views, focal=30, zrate=0.5, rots=2, N=120):
    poses = []
    for view in views:
        tmp_view = np.eye(4)
        tmp_view[:3] = np.concatenate([view.R.T, view.T[:, None]], 1)
        tmp_view = np.linalg.inv(tmp_view)
        tmp_view[:, 1:3] *= -1
        poses.append(tmp_view)
    poses = np.stack(poses, 0)
    # poses = np.stack([np.concatenate([view.R.T, view.T[:, None]], 1) for view in views], 0)
    c2w = poses_avg(poses)
    up = normalize(poses[:, :3, 1].sum(0))

    # Get radii for spiral path
    rads = np.percentile(np.abs(poses[:, :3, 3]), 90, 0)
    render_poses = []
    rads = np.array(list(rads) + [1.0])

    for theta in np.linspace(0.0, 2.0 * np.pi * rots, N + 1)[:-1]:
        c = np.dot(
            c2w[:3, :4],
            np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0]) * rads,
        )
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))
        render_pose = np.eye(4)
        render_pose[:3] = viewmatrix(z, up, c)
        # render_pose[:3] =  np.array([[ 9.9996626e-01, -7.5253481e-03, -3.2866236e-03, -5.6992844e-02],
        #             [-7.7875191e-03, -9.9601853e-01, -8.8805482e-02, -2.9015102e+00],
        #             [-2.6052459e-03,  8.8828087e-02, -9.9604356e-01, -2.3510060e+00]])
        render_pose[:3, 1:3] *= -1
        render_poses.append(np.linalg.inv(render_pose))
    return render_poses



"""
Use to get the gt 3d coord from query keypoint
"""

def new_calculate_ndc2camera(proj_matrix, xndc, yndc, depth):
    a1 = proj_matrix[0,0]
    a2 = proj_matrix[0,1]
    a3 = proj_matrix[0,2]
    a4 = proj_matrix[0,3]
    
    a5 = proj_matrix[1,0]
    a6 = proj_matrix[1,1]
    a7 = proj_matrix[1,2]
    a8 = proj_matrix[1,3]
    
    
    a13 = proj_matrix[3,0]
    a14 = proj_matrix[3,1]
    a15 = proj_matrix[3,2]
    a16 = proj_matrix[3,3]
    
    A1 = a1-xndc*a13
    B1 = a2-xndc*a14
    C1 = (a3-xndc*a15)*depth+a4-xndc*a16
    
    A2 = a5-yndc*a13
    B2 = a6-yndc*a14
    C2 = (a7-yndc*a15)*depth+a8-yndc*a16
    
    X = (-C1*B2+C2*B1)/(A1*B2-A2*B1)
    Y = (-A1*C2+A2*C1)/(A1*B2-A2*B1)
    
    return X, Y

def pixel2ndc(pixel, S):
    return (((pixel/0.5)+1.0)/S)-1.0

def getGTXYZ(camera2ndc, view2camera, point_2d, depth_map):
    #Get the depth value
    depth_map = depth_map.detach().squeeze(0)
    depth = depth_map[point_2d[:,1].int().to("cpu"), point_2d[:,0].int().to("cpu")] 
    X, Y = new_calculate_ndc2camera(camera2ndc.transpose(0,1), pixel2ndc(point_2d[:,0], 640), pixel2ndc(point_2d[:,1], 480), depth)
    ones = torch.tensor([1.0]).repeat(point_2d.size(0)).to("cuda")
        
    cam_coord_inv = torch.stack([X, Y, depth, ones], dim=1)
    output = torch.matmul(cam_coord_inv.double(), torch.inverse(view2camera).double())
    return output[:, :3]


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



def project_and_filter(points_3d,P, W, H):
    """
    Project the 3D points into 2D pixel spaces with given projection matrix
    The pixel that is projected out of the pixel space [0:W, 0:H] will be rejected
    
    Input:  
         points_3d: [N, 3] Tensor
         points_feat: [N,C] Tensor: the feature associated with each 3D points 
         P: [3, 4] Projection Matrix Tensor

    Return :
         mask 
         result: [M, 5] Tensor, Each line  [x, y, X, Y, Z] contains its pixel coordinates (x, y) and its asscociated 
                 3D point cloud coordinates (X, Y , Z)   
         points_feat_filter: [M, C] the feature of all the projected keypoint that is inside the pixel space 
    """
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


    # Concetenate to [M, 5]： [x, y, X, Y, Z]
    result = torch.cat([x_filtered.unsqueeze(1), 
                        y_filtered.unsqueeze(1), 
                        points_3d_filtered], dim=1)   
    
    return mask, result





