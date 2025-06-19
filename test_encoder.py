import torch
import sys


import cv2
import numpy as np
import time
import torch
import torch.optim as optim

########## Image #############
from PIL import Image
from torchvision.transforms import PILToTensor
##############################




from scene import Scene
from tqdm import tqdm

from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams

from utils.graphics_utils import getWorld2View2, fov2focal
from utils.general_utils import image_process

from XFeat.modules.xfeat import XFeat
from scene.feat_pointcloud import FeatPointCloud
from encoder.feat_encoder import FeatEncoder, Config
from utils.pose_utils import project_and_filter
from utils.graphics_utils import getIntrinsic, getExtrinsic
from utils.loc_utils import calculate_pose_errors

import torch.nn.functional as F

# // For the netvlad global descriptor
from netvlad.netvlad import NetVLAD

""""
This file is used to run the whole test cases and get the average error for xfeat feature
It first gets the test image from the Scene. Apply the Xfeat detector to get the keypoints and descriptors
The test image will then sent to NetVlad to get its global descriptor. This global descriptor will then be
used to find its most similar image in training dataset (each image in training dataset are considered as 
reference image)
Once query image and reference image are found, we get the extrinsic and intrinsic parameters of reference image and then 
Use these two paramters to project(rasterize) the 3DGS-Xfeat. We record each pixel in the projected feature map that correspond with a 
3DGS point. Theses pixel with its feature are then used to match with the keypoint of query image.
Once the matching is done, for each matching points in  projected image, we have its 3D coordinate so that we can directly apply PnP RANSAC 
to 2D(query)-3D (3DGS)        
The methods return the average of all the test image's pose error 

command: 
python test_encoder.py -s ../GSplatLoc/gsplatloc-main/datasets/wholehead/ -m ../GSplatLoc/gsplatloc-main/output_wholescene/img_2000_head --iteration 15000

we need to already train a 3DGS with xfeat feature in 15000 iteration and put it into the "output_wholescene/img_2000_head"
Training image must be put in datasets/wholehead/

If we want to use the netvlad to do the image retrieval, we must launch the getdes.py. Make sure that in the netvlad.py, 
from netvlad.base_model import BaseModel must be 
from base_model import BaseModel

python getdes.py -s datasets/wholehead/ -m output_wholescene/img_2000_head --iteration 15000

Then after get the global descriptor, change the 
from base_model import BaseModel
back to  
from netvlad.base_model import BaseModel 
before runing the 2d_feature_disk_all.py
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import collections as mplcollections
from matplotlib import colors as mcolors

import torch.nn as nn



class InterpolateSparse2d(nn.Module):
    """ Efficiently interpolate tensor at given sparse 2D positions. """ 
    def __init__(self, mode = 'bicubic', align_corners = False): 
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners

    def normgrid(self, x, H, W):
        """ Normalize coords to [-1,1]. """
        return 2. * (x/(torch.tensor([W-1, H-1], device = x.device, dtype = x.dtype))) - 1.

    def forward(self, x, pos, H, W):
        """
        Input
            x: [B, C, H, W] feature tensor
            pos: [B, N, 2] tensor of positions
            H, W: int, original resolution of input 2d positions -- used in normalization [-1,1]

        Returns
            [B, N, C] sampled channels at 2d positions
        """
        grid = self.normgrid(pos, H, W).unsqueeze(-2).to(x.dtype)
        x = F.grid_sample(x, grid, mode = self.mode , align_corners = False)
        return x.permute(0,2,3,1).squeeze(-2)


def pixel2ndc(pixel, S):
    return (((pixel/0.5)+1.0)/S)-1.0

def getOrigPoint(point_in_render_img, W, H, ProjMatrix):
    pixel_x, pixel_y, proj_z, p_w = point_in_render_img[1], point_in_render_img[0], point_in_render_img[2], point_in_render_img[3]
    p_proj_x, p_proj_y = pixel2ndc(pixel_x, W), pixel2ndc(pixel_y, H)
    p_hom_x, p_hom_y, p_hom_z = p_proj_x/p_w, p_proj_y/p_w, proj_z/p_w
    p_hom_w = 1/p_w
    p_hom = np.array([p_hom_x, p_hom_y, p_hom_z,p_hom_w])
    origP = np.matmul(p_hom, np.linalg.inv(ProjMatrix), dtype=np.float32)
    origP = origP[:3]
    print("ori = ", point_in_render_img[4], point_in_render_img[5],  point_in_render_img[6] )
    return origP

def getAllOrigPoints(points_in_render_img, W,H,ProjMatrix):
    match_3d = []
    for piri in points_in_render_img:
        origP = getOrigPoint(piri, W,H,ProjMatrix)
        match_3d.append(origP)
    return match_3d

def getXY(points):
    res = []
    for p in points:
        res.append([p[0],p[1]])
    return res
    
    

def warp_corners_and_draw_matches(ref_points, dst_points, img1, img2):
    dst_points_xy = dst_points[:, [0,1]]
    # Calculate the Homography matrix
    H, mask = cv2.findHomography(ref_points, dst_points, cv2.USAC_MAGSAC, 3.5, maxIters=1_000, confidence=0.999)
    mask = mask.flatten()

    # Get corners of the first image (image1)
    h, w = img1.shape[:2]
    corners_img1 = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32).reshape(-1, 1, 2)

    # Warp corners to the second image (image2) space
    warped_corners = cv2.perspectiveTransform(corners_img1, H)

    # Draw the warped corners in image2
    img2_with_corners = img2.copy()
    for i in range(len(warped_corners)):
        start_point = tuple(warped_corners[i-1][0].astype(int))
        end_point = tuple(warped_corners[i][0].astype(int))
        cv2.line(img2_with_corners, start_point, end_point, (0, 255, 0), 4)  # Using solid green for corners

    # Prepare keypoints and matches for drawMatches function
    keypoints1 = [cv2.KeyPoint(p[0], p[1], 5) for p in ref_points]
    keypoints2 = [cv2.KeyPoint(p[0], p[1], 5) for p in dst_points]
    matches = [cv2.DMatch(i,i,0) for i in range(len(mask)) if mask[i]]

    # Draw inlier matches
   # img_matches = cv2.drawMatches(img1, keypoints1, img2_with_corners, keypoints2, matches, None,
   #                               matchColor=(0, 255, 0), flags=2)
    
    #clean the keypoint with mask
    ref_points_valid = []
    match_3d_points = []
    for i in range(len(mask)):
        if mask[i]:
            ref_points_valid.append(ref_points[i])
            match_3d_points.append(dst_points[i, [4,5,6]])

    return  ref_points_valid, match_3d_points



def getRefImg(query_name):
    #Get the image number
    query_index = int(query_name.split("-")[1])
    if query_index + 15 > 1000:
        ref_index = query_index - 15
    else:
        ref_index = query_index + 15
    if ref_index > 99:
        ref_index = "000" + str(ref_index)
    else:
        ref_index = "0000"+ str(ref_index)
    ref_name = "frame-" + str(ref_index)
    return  ref_name

def createNetVlad():
    conf = {"model_name": "VGG16-NetVLAD-Pitts30K", "whiten": True}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NetVLAD(conf).eval().to(device)
    
    return model

def imageRetrieval(query_img, netvlad_model,global_desc_names):
    
    global_desc, names = torch.squeeze(global_desc_names[0]), global_desc_names[1]
    query_global_desc = netvlad_model(query_img[None])["global_descriptor"]
    
    similarity = torch.mm(query_global_desc, global_desc.t().cuda())
    _, idx = similarity.max(dim=1)
    
    #num_seq = names[idx].split("/")[0]
    #img_name = names[idx].split("/")[1]
    
    #return img_name, num_seq
    return names[idx]

  


def localize_set(scene, args):

    views_test = scene.getTestCameras()
    views_train = scene.getTrainCameras()

    # Keep track of rotation and translation errors for calculation of the median error.
    rErrs = []
    tErrs = []

    prior_rErr = []
    prior_tErr = []
    pnp_p = []
    inliers = []


    #Load the global descriptor
    global_desc_names = torch.load("./netvlad/global_desc.pt")
    netvlad_model = createNetVlad()
    
    featpc = FeatPointCloud()
    featpc.init_feat_pc(args.source_path, 64) 
    feat_pcd = torch.tensor(featpc.get_xyz)
    
    # Load the Encoder model
    encoder_pth = scene.model_path + "/3dsparse_feat_" + str(args.iteration) + ".pth"
    conf = Config()
    encoder = FeatEncoder(conf).cuda()
    encoder.load_state_dict(torch.load(encoder_pth, weights_only=True))
    encoder.eval()
    
        
    xfeat = XFeat(top_k=4096)
        
    for _, view in enumerate(tqdm(views_test, desc="Matching progress")):
        
        # Open the test image
        try:
            query_img = Image.open(view.image_path) 
        
        except:
            print(f"Error opening image: {view.image_path}")
            continue
        
        original_image = image_process(query_img)
        query_image = original_image.cuda()

        
        #Get the reference image name 
        ref_name= imageRetrieval(query_image, netvlad_model,global_desc_names)

        #load reference image
        ref_view = [view for view in views_train if view.image_name == ref_name]
        
        #Get reference image parameters
        ref_K = getIntrinsic(ref_view[0], original_image.shape[2], original_image.shape[1])
        cam_intri = torch.reshape(torch.tensor(ref_K), (1,9))
        ref_R, ref_t = ref_view[0].R, ref_view[0].T
        cam_extri = torch.tensor(getExtrinsic(ref_R, ref_t)).view(1, 16)
        
        
        # Get the reference pose and project a sparse feature map
        mask, proj_sfm = project_and_filter(feat_pcd, ref_view[0].full_proj_transform, query_image.shape[1], query_image.shape[2])
        proj_p_feature = encoder(proj_sfm[:, 2:], cam_intri, cam_extri).squeeze(0)
        
        
        #Get the image R and t and the reference K
        query_R, query_t= view.R, view.T, 
        

        # Use     
        query_keypoints, _, query_feature = xfeat.detectAndCompute(query_image[None], 
                                                                 top_k=4096)[0].values()   #ref_keypoints size = [top_k, 2] x-->W y-->H x and y are display coordinate
        
       
        # Matching
        idxs0, idxs1 = xfeat.match(query_feature.to("cpu"), proj_p_feature.to("cpu"), min_cossim=0 )
        query_matched = query_keypoints[idxs0].cpu().numpy()
        match_3d = proj_sfm[:, 2:][idxs1]

        _, R, t, inl = cv2.solvePnPRansac(match_3d.cpu().numpy(), query_matched, 
                                                      ref_K, 
                                                      distCoeffs=None, 
                                                      flags=cv2.SOLVEPNP_ITERATIVE, 
                                                      iterationsCount=20000
                                                      )
        R, _ = cv2.Rodrigues(R) 
    
        #print("R = ", R  , "  t= ", t)
        #print("query R = ", query_R, "query t = ", query_t)
        rotError, transError = calculate_pose_errors(query_R, query_t, R.T, t)

        # Print the errors
        print(f"Rotation Error: {rotError} deg")
        print(f"Translation Error: {transError} cm")

        if inl is not None:
            prior_rErr.append(rotError)
            prior_tErr.append(transError)
            inliers.append(len(inl))
    
    err_mean_rot =  np.mean(prior_rErr)
    err_mean_trans = np.mean(prior_tErr)
    mean_inliers = np.mean(inliers) 
    print(f"Rotation Average Error: {err_mean_rot} deg ")
    print(f"Translation Average Error: {err_mean_trans} cm ")
    print(f"Mean inliers : {mean_inliers} cm ")
    

    

def launch_inference(dataset : ModelParams, args): 
    
    scene = Scene(dataset, load_iteration=args.iteration, shuffle=False)
    localize_set(scene, args)


if __name__ == "__main__":
# Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    args = parser.parse_args(sys.argv[1:])
    args.data_device="cuda"
    args.eval = True

    launch_inference(model.extract(args), args)











