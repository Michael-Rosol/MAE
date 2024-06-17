# Masked Autoencoder remake from paper: 

import torch 
import torch.nn as nn
import torchvision 
from einops.layers.torch import Rearrange 

from timm.models.vision_transformer import PatchEmbed, Block

class MaskedAutoEncoder(nn.Module):
    
    def __init__(self, emb_size=1024, decoder_emb_size=512, patch_size=16, num_heads=16, encoder_depth=24,decoder_num_heads=16, 
                  decoder_depth=8,in_chans=3, img_size=224, mlp_ratio=4, norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        #Patch Embedding 
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, emb_size)
        num_patches = self.patch_embed.num_patches

        #Class tokening and positional embedding: 
        self.cls_token = nn.Parameter(torch.zeros(1,1, emb_size)) #starts with zeros to avoid any transformer bias 
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, emb_size), requires_grad=False)

        # Encoder Blocks: 
        self.blocks = nn.ModuleList([
            Block(emb_size, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(encoder_depth)])
        self.norm = norm_layer(emb_size)


        # Decoder Blocks: 
        self.decoder_embed = nn.Linear(emb_size, decoder_emb_size, bias=True)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_emb_size, requires_grad=False))
        self.mask_token = nn.Parameter(torch.zeros(1,1, decoder_emb_size))

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_emb_size, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_pred = nn.Linear(decoder_emb_size, patch_size**2 * in_chans, bias=True)
        self.norm_pix_loss = norm_pix_loss
        # Projection: 

        self.projection = nn.Sequential(
            nn.Conv2d(in_channels=in_chans, out_channels=patch_size**2 * in_chans,kernel_size=patch_size, stride=patch_size),
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
        )

        def forward(self, x, mask_ratio):
            x = self.pos_embed


# # Test the MaskedAutoEncoder class for patch embedding
# if __name__ == "__main__":
#     # Create a random tensor to represent a batch of images (batch_size, channels, height, width)
#     batch_size = 1
#     img_size = 224
#     patch_size = 16
#     in_chans = 3
#     emb_size = 1024
#     dummy_images = torch.randn(batch_size, in_chans, img_size, img_size)

#     # Initialize and apply the encoder
#     model = MaskedAutoEncoder(emb_size=emb_size, patch_size=patch_size, in_chans=in_chans, img_size=img_size)
#     patch_embeddings = model(dummy_images)

#     print("Patch Embeddings Shape:", patch_embeddings.shape)