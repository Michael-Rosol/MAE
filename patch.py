import torch
import os 
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch import Tensor
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
# class PatchEmbedding(nn.Module):
#     def __init__(self, in_chans=3, patch_size=16, emb_size=1024):
#         super().__init__()
#         self.patch_size = patch_size
#         self.projection = nn.Sequential(
#             Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
#             nn.Linear(patch_size * patch_size * in_chans, emb_size)
#         )

#     def forward(self, x: Tensor) -> Tensor:
#         x = self.projection(x)
#         return x

# # Define transform to normalize the data and convert to tensor
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

# # Download and load the CIFAR-10 training dataset
# dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# # Get a sample datapoint
# sample_datapoint, _ = next(iter(dataloader))
# print("Initial Shape:", sample_datapoint.shape)

# # Initialize PatchEmbedding and apply it to the sample datapoint
# embedding = PatchEmbedding()(sample_datapoint)
# print("Patches shape:", embedding.shape)

# # Optionally, print out the patches
# print("Patches:")
# print(embedding)

class Patchify(nn.Module):
    def __init__(self, patch_size=56):
        super().__init__()
        self.p = patch_size
        self.unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x -> B c h w
        bs, c, h, w = x.shape
        
        x = self.unfold(x)
        # x -> B (c*p*p) L
        
        # Reshaping into the shape we want
        a = x.view(bs, c, self.p, self.p, -1).permute(0, 4, 1, 2, 3)
        # a -> ( B no.of patches c p p )
        return a
    
patch = Patchify()
img_src = './timg1.jpeg'
image = cv2.imread(img_src)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image.astype('float32') / 255.0  # Normalize to [0, 1]
image = torch.from_numpy(image)
image = image.permute(2,0,1)
image = image.unsqueeze(0) #to add the batch dimension
p = patch(image)
p = p.squeeze() #to remove the batch dimension for plotting

def plot_patches(tensor):
    fig = plt.figure(figsize=(8, 8))
    grid = ImageGrid(fig, 111, nrows_ncols=(4, 4), axes_pad=0.1)

    for i, ax in enumerate(grid):
        patch = tensor[i].permute(1, 2, 0).numpy() 
        ax.imshow(patch)
        ax.axis('off')

    plt.show()

plot_patches(p)