# Personal Research on META AI's Masked Autoencoder Model 

### Personal Note
Inspired from META's Segment Anything (SAM) Paper, I decided to learn more about the structure of the masked autoencoder (MAE) that is used in SAM.

### Discovery 
- I primarily modified the way masking is performed by the masked autoencoder model by lessening the randomness of the model and implementing more weighting to target the centralized region of the image patches.
-  I further saw that adding proportional masking where there is no randomess lessens the effectiveness of the model. This even masking was done through a skip method of masking every other patch in a linear fashion. 

### Takeaway 
Ultimately, the model.py containing the modified masked autoencoder was capable to efficiently become a self-supervised autoencoder while being more efficient in loss than the original model due to its slight centralized masking. 

### References: 

 @Article{MaskedAutoencoders2021,
  author  = {Kaiming He and Xinlei Chen and Saining Xie and Yanghao Li and Piotr Doll{\'a}r and Ross Girshick},
  journal = {arXiv:2111.06377},
  title   = {Masked Autoencoders Are Scalable Vision Learners},
  year    = {2021},
}
