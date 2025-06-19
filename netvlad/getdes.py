import torch
import torch.nn as nn


import cv2
import numpy as np
import os
from PIL import Image



import sys
sys.path.insert(0, 'C:/Users/fwu/Documents/PhD_FanWU/MyCode/3DSparsefeat')

from netvlad.netvlad import NetVLAD

from scene import Scene
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import ModelParams
from utils.general_utils import image_process

###  Command #############
# python getdes.py -s ../../GSplatLoc/gsplatloc-main/datasets/wholehead/ -m ../../GSplatLoc/gsplatloc-main/output_wholescene/img_2000_head
########################################################################



def getNetVladDesc(views, model):   # --->  global descriptor [N, 128] and image name:  [N, 1]
    
    imgs_name = []
    global_desc = []
    
    for _, view in enumerate(tqdm(views, desc="Rendering progress")):
        #Get the image
        img_name = view.image_name  # ex: seq-1/frame-000455
        try:
            query_img = Image.open(view.image_path) 
        except:
            print(f"Error opening image: {view.image_path}")
            continue
        original_image = image_process(query_img)
        image = original_image.cuda() # [3,480,640]
        
        output = model(image[None])["global_descriptor"]
        global_desc.append(output.detach().cpu())
        imgs_name.append(img_name)
        
    return global_desc, imgs_name


# Discard layers at the end of base network
conf = {"model_name": "VGG16-NetVLAD-Pitts30K", "whiten": True}

device = "cuda" if torch.cuda.is_available() else "cpu"
model = NetVLAD(conf).eval().to(device)



#Prepare the dataset 
parser = ArgumentParser(description="Netvlad parameters")
model_params = ModelParams(parser, sentinel=True)
args = parser.parse_args(sys.argv[1:])
args.eval = True
args.data_device = "cuda"
scene = Scene(model_params.extract(args), load_iteration=args.iteration, shuffle=False)

global_desc, imgs_name = getNetVladDesc(scene.getTrainCameras(), model)
desc_names_tensor = [torch.stack(global_desc), imgs_name]
torch.save(desc_names_tensor, 'global_desc.pt')




"""
#Get the image 
img_dir_query = "../datasets/wholehead/images/seq-01"   
query_img_name = "frame-000969.color.png"
query_img_path = os.path.join(img_dir_query, query_img_name)
query_img = cv2.imread(query_img_path) # [H,W,C] = [480,640,3]
query_img_tensor = torch.tensor(query_img).permute(2,0,1)[None].cuda() # [C,H,W]

# This is just toy example. Typically, the number of samples in each classes are 4.
#labels = torch.randint(0, 10, (40, )).long()
#x = torch.rand(40, 3, 128, 128).cuda()
output = model(query_img_tensor.to(torch.float))

#triplet_loss = criterion(output, labels)
"""

"""

conf = {"model_name": "VGG16-NetVLAD-Pitts30K", "whiten": True}

device = "cuda" if torch.cuda.is_available() else "cpu"
model = NetVLAD(conf).eval().to(device)


#Get the image 
img_dir_query = "../datasets/wholehead/images/seq-01"   
query_img_name = "frame-000969.color.png"
query_img_path = os.path.join(img_dir_query, query_img_name)
query_img = cv2.imread(query_img_path) # [H,W,C] = [480,640,3]
query_img_tensor = torch.tensor(query_img).permute(2,0,1)[None].cuda() # [C,H,W]


pred = model(query_img_tensor)
print(pred["global_descriptor"])


"""