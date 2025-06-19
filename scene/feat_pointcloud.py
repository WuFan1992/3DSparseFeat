import torch
import numpy as np
import os
import torch.nn as nn
from utils.graphics_utils import BasicPointCloud
from utils.system_utils import mkdir_p

from plyfile import PlyData, PlyElement
from scene.colmap_loader import read_points3D_binary, read_points3D_text

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    ids_3d = vertices['id_3d']
    return BasicPointCloud(points=positions, colors=colors, normals=normals, ids_3d=ids_3d)

def storePly(path, xyz, rgb, ids_3d):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('id_3d', 'int32')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb, ids_3d), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


class FeatPointCloud:
    
    def __init__(self):
        self._xyz = torch.empty(0).to("cuda")
        self.xyz_gradient_accum = torch.empty(0)
        self._ids_3d = torch.empty(0).to("cuda")
        
    @property
    def get_xyz(self):
        return self._xyz
    
    
    @property
    def get_ids_3d(self):
        return self._ids_3d 
    
    
    def generate_ply(self, path):
        ply_path = os.path.join(path, "sparse/0/points3D_withidx.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _, ids_3d = read_points3D_binary(bin_path)
            
            except:
                xyz, rgb, _, ids_3d = read_points3D_text(txt_path)
            
            storePly(ply_path, xyz, rgb, ids_3d)
        
    
    def init_feat_pc(self, source_path, semantic_feature_size : int):
        """
        fetch the point cloud 
        """
        # Generate the ply file
        self.generate_ply(source_path)
        
        # Get the point cloud path
        pc_path = os.path.join(source_path,"sparse/0/points3D_withidx.ply")
        # Load the pcd
        pcd = fetchPly(pc_path)
        # set the initial xyz 
        self._xyz = torch.tensor(np.asarray(pcd.points)).float().cuda()
        # Set the point cloud 3D
        self._ids_3d = torch.tensor(np.array(pcd.ids_3d)).float().cuda()
        
    
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        return l
        
    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals), axis=1) 
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)
    
    def load_ply(self, path):
        plydata = PlyData.read(path)

        self._xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        
        
        count = sum(1 for name in plydata.elements[0].data.dtype.names if name.startswith("semantic_"))
    
    def update_ply(self, indices: np.array, kp_feat: torch.Tensor):
        """
            For each matched 3D points, find its closest point in point cloud and update its feature 
            indices :  index in the Point cloud that need to update the feature
            kp_feat: 3D point feature [N, 64] 
        
        """
        
        valid_mask = indices != -1
        filter_indices = torch.tensor(indices[valid_mask])
        filter_feature = kp_feat[valid_mask]

        #Get the index of semantic_feature given the indice and the ids_3d
        np_indices = self._ids_3d.cpu().numpy()
        feat_indices = [np.where(np_indices == j)[0][0].item() for j in filter_indices]
      
            
            
    

    
    
    
    