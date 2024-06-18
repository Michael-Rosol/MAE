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
            Block(emb_size, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(encoder_depth)])
        self.norm = norm_layer(emb_size)


        # Decoder Blocks: 
        self.decoder_embed = nn.Linear(emb_size, decoder_emb_size, bias=True)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_emb_size, requires_grad=False))
        self.mask_token = nn.Parameter(torch.zeros(1,1, decoder_emb_size))

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_emb_size, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_pred = nn.Linear(decoder_emb_size, patch_size**2 * in_chans, bias=True)
        self.norm_pix_loss = norm_pix_loss
        # Projection: 

        self.projection = nn.Sequential(
            nn.Conv2d(in_channels=in_chans, out_channels=patch_size**2 * in_chans,kernel_size=patch_size, stride=patch_size),
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
        )

    def random_masking(self, x, mask_ratio):
            """
            Perform per-sample random masking by per-sample shuffling.
            Per-sample shuffling is done by argsort random noise.
            x: [N, L, D], sequence
            """
            B, L, D = x.shape  # batch, length, dim
            len_keep = int(L * (1 - mask_ratio))
        
            noise = torch.rand(B, L, device=x.device)  # noise in [0, 1]

            # sort noise for each sample, then ids_shuffle to keep original index
            ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
            ids_restore = torch.argsort(ids_shuffle, dim=1)

            # keep the first subset
            ids_keep = ids_shuffle[:, :len_keep]
            x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

            # generate the binary mask: 0 is keep, 1 is remove
            mask = torch.ones([B, L], device=x.device)
            mask[:, :len_keep] = 0
            # unshuffle to get the binary mask
            mask = torch.gather(mask, dim=1, index=ids_restore)

            return x_masked, mask, ids_restore


    def encoder(self, x, mask_ratio):
        x = self.pos_embed(x)
            
        B, _, _ = x.shape 

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
            
           
        x = x + self.pos_embed

        x, mask, restore_id = self.random_masking(x, mask_ratio)
            
        #cls tokenization: 
        cls_tokens = self.cls_token.expand(B, -1, -1) 
        x = torch.cat((cls_tokens,x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
            x = self.norm(x)

        return x, mask, ids_restore

    def decoder(self, x, ids_restore): 
            
        x = self.decoder_embed(x) 

        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def loss(self, imgs, pred, mask):

        """
        imgs: [N, 3, H, W]
        pred: [N, L, patch*patch*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """

        target = self.projection(imgs) 

        loss = (pred - target) ** 2 
        loss = loss.mean(dim=-1)

        loss = (loss * mask).sum() / mask.sum() 
        return loss 
        
    def forward(self, imgs, mask_ratio=0.75):
        x, mask, restore_ids = self.encoder(imgs, mask_ratio)
        pred = self.decoder(x, restore_ids) 
        loss  = self.loss(imgs, pred, mask) 
        return loss, pred, mask
        
# Test the MaskedAutoEncoder class for patch embedding
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.rand(1,3, 224, 224).to(device)
    model = MaskedAutoEncoder().to(device)
    print(model(x)[1].shape)