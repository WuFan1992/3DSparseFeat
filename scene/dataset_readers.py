
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys

from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text, read_points3D_nvm
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from torchvision.transforms import PILToTensor


import torch

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image_path: str
    image_name: str
    seq_num: int
    keypoints: np.array
    point3d_id: np.array


class SceneInfo(NamedTuple):
    train_cameras: list
    test_cameras: list
    ply_path: str
 
 

@torch.inference_mode()
def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width
        image_name = extr.name

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE" or intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        ### elif intr.model=="PINHOLE":
        elif intr.model=="PINHOLE" or intr.model=="OPENCV":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"


        image_path = os.path.join(images_folder, extr.name)

        seq_num = extr.name.split("/")[0]

        
        keypoints = extr.xys  # ex: [[266.845    56.1778 ] [210.445    60.8841 ] [213.72     70.4107 ]]
        point3d_id = extr.point3D_ids # ex  [    -1     -1 143442 138231  52487 ]
      
  
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, 
                            image_path=image_path, image_name=image_name,
                            seq_num=seq_num, keypoints=keypoints, point3d_id=point3d_id)
        
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval): 
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    
    reading_dir = "images" if images == None else images
    
    if os.path.exists(os.path.join(path, "sparse/0", "list_test.txt")):
        # 7scenes
        with open(os.path.join(path, "sparse/0", "list_test.txt")) as f:
            test_images = f.readlines()
            test_images = [x.strip() for x in test_images]
    elif os.path.exists(os.path.join(path, "dataset_test.txt")):
        # cambridge
        with open(os.path.join(path, "dataset_test.txt")) as f:
            test_images = f.readlines()
            test_images = [x.split(" ")[0] for x in test_images if x[0] != '#']
    else:
        test_images = []

    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, 
                                           images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)



    if eval:
        train_cam_infos = []
        test_cam_infos = []
        for cam_info in cam_infos:
            if cam_info.image_name in test_images:
                test_cam_infos.append(cam_info)
            else:
                train_cam_infos.append(cam_info)
        
    else:
        train_cam_infos = []
        test_cam_infos = cam_infos
    
    print(f'test cameras: {len(test_cam_infos)}')
    print(f'train cameras: {len(train_cam_infos)}')


    ply_path = os.path.join(path, "sparse/0/points3D.ply")

    scene_info = SceneInfo(train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           ply_path=ply_path)
     
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo
}