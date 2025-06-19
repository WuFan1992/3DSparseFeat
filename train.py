
from tqdm import tqdm
from PIL import Image
from random import randint
import torch.nn.functional as F
import torch

from arguments import *
from encoder.feat_encoder import FeatEncoder, Config
from scene import Scene
from scene.feat_pointcloud import FeatPointCloud
from utils.graphics_utils import getIntrinsic, getExtrinsic
from utils.general_utils import image_process, sample_features
from XFeat.modules.xfeat import XFeat
from utils.pose_utils import project_and_filter


"""
python train.py -s ../GSplatLoc/gsplatloc-main/datasets/wholehead/ -m ../GSplatLoc/gsplatloc-main/output_wholescene/img_2000_head --iteration 15000

"""


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def train(dataset):
    
    xfeat = XFeat(top_k=4096)
    scene = Scene(dataset, load_iteration=15000)
    featpc = FeatPointCloud()
    featpc.init_feat_pc(dataset.source_path, 64) 
    feat_pcd = torch.tensor(featpc.get_xyz)
    
    viewpoint_stack = scene.getTrainCameras().copy()
    
    conf = Config()
    encoder = FeatEncoder(conf).cuda()
    
    progress_bar = tqdm(range(0, 15000), desc="Training progress")
    optimizer = torch.optim.SGD(encoder.parameters(), lr=1.e-6)
    
    saving_itr = [1000,2000,5000,7000,10000, 12500,15000]
    
    for iteration in range(0, 15000+1):
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        try:
            image = Image.open(viewpoint_cam.image_path) 
        except:
            print(f"Error opening image: {viewpoint_cam.image_path}")
            continue
        
        original_image = image_process(image)
        gt_image = original_image.cuda()
        
        gt_feature_map = xfeat.get_descriptors(gt_image[None])[0]
        gt_feature_map = F.interpolate(gt_feature_map.unsqueeze(0), size=(gt_image.shape[1], gt_image.shape[2]), mode='bilinear', align_corners=True).squeeze(0)
        
        # Define intrinsic matrix
        K = getIntrinsic(viewpoint_cam, original_image.shape[2], original_image.shape[1])
        cam_intri = torch.reshape(torch.tensor(K), (1,9))
        
        # Define pose matrix
        gt_R, gt_t = viewpoint_cam.R, viewpoint_cam.T
        
        cam_extri = torch.tensor(getExtrinsic(gt_R, gt_t)).view(1, 16)
        
        # Project the point cloud into image space
        mask, proj_sfm_2d = project_and_filter(feat_pcd, viewpoint_cam.full_proj_transform, gt_image.shape[1], gt_image.shape[2])
        
        # For each project point, get its gt feature
        gt_feat = sample_features(proj_sfm_2d[:,:2], gt_feature_map)
        
        pred_feat = encoder(feat_pcd[mask], cam_intri, cam_extri)
        
        loss = l1_loss(gt_feat, pred_feat.cuda())
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        
        with torch.no_grad():
            # Progress bar
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{loss.item():.{7}f}"})
                progress_bar.update(10)
            if iteration in saving_itr:
                torch.save(encoder.state_dict(), scene.model_path + "/3dsparse_feat_" + str(iteration) + ".pth")
    progress_bar.close()

        



if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iteration)
    
    args.eval = True
    args.data_device = "cuda"
    
    train(model.extract(args))
    
    
    