import torch
import torch.nn as nn 
import torch.nn.functional as F

from .pos_encoder import PositionEncoder
from .utils import get_activation
from .transformer import LoFTREncoderLayer
from .mlp import MLP

class Config:
    pos_in_channels : int = 3
    kp_in_channels: int = 64
    pos_N_freq: int = 10
    pos_max_freq: int = 9
    fusion_channles: int = 512
    post_embed_channels: int = 256
    density_channel : int = 1
    mass_center_channel : int = 3
    camera_embed_input_channel: int = 25
    transformer_header_num : int = 4
    
def weights_init_uniform(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # apply a uniform distribution to the weights and a bias=0
        m.weight.data.uniform_(0.0, 1.0)
        m.bias.data.fill_(0)

class FeatEncoder(nn.Module):
    def __init__(self, config: Config):
        super(FeatEncoder, self).__init__()
        """
        Reference to "NeRF" Position Embedding 
        input [x,y,z], dim = 3
        output: dim = 60
        """
        self.embed = PositionEncoder(config.pos_in_channels, config.pos_N_freq, config.pos_max_freq) # (3,10,10)
        embed_output_dim = self.embed.out_dim 
        """
        Reference to "NeRF"  backbone module:
          project position embedding to higher dimension
          1 x Linear, No activation
          input : dim = 60
          output : dim = 256 (The same as position embedding in NeRF and in LOFTR )
        """
        self.post_embed = nn.Linear(embed_output_dim, config.post_embed_channels)
        

        """
        Reference to "Triplane meets Gaussian Splatting " camera embedding module:
        project camera extrinsic and intrinsic to higher dimension
         1 x Linear, 1 x activation (silu), 1 x Linear
          input : dim = 25
          output : dim = 256 (The same as position embedding in NeRF and in LOFTR )
        """
        self.camera_embed = MLP(config.camera_embed_input_channel, config.post_embed_channels, config.post_embed_channels, 1, "silu")
        
        """
        Reference to "LOFTR" and "Triplane meets Gaussian"
        4 layers transformer(with modulation)(LOFTR) 
        input : dim = 256
        output: dim = 256
        header = 4
        """
        self.self_atten = LoFTREncoderLayer(config.post_embed_channels, config.post_embed_channels, config.transformer_header_num)
        
 
        
        """
        Regress the final descriptor
        4 x mlp 
        input : dim = 256
        hidden : dim = 1024  (projet to higher dimension)
        output : dim = 64
        """
        self.estim_feature = MLP(config.post_embed_channels, 64,1024, 4, "relu")
        
    
    def forward(self, pos: torch.Tensor, cam_intr: torch.Tensor, cam_extr: torch.Tensor):
        
        
        # position embedding
        pos_embed = self.embed(pos)
 
        #post processing position embedding dim from 60 to 256
        pos_embed = self.post_embed(pos_embed)


        
        # prepare camera embedding input
        cam_data = torch.cat([cam_intr, cam_extr], dim=1).float().cuda()
        cam_data = self.camera_embed(cam_data)

        # self attention condition with 
        token = self.self_atten(pos_embed, pos_embed, cam_data)
        
        #estimate feature
        feature = self.estim_feature(token)

        return feature



