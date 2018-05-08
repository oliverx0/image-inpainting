from data_transforms.generate_mask import GenerateMask
from torchvision import transforms, utils
from torchvision.transforms import Resize, RandomCrop, ToTensor
from datasets.mask_dataset import MaskDataset

from torch.utils.data import DataLoader

import numpy as np 
import matplotlib.pyplot as plt
  
from partial_conv import PartialConv

transform = transforms.Compose([
  Resize(400),
  RandomCrop(400),
  ToTensor(),
  GenerateMask()
])

transformed_dataset = MaskDataset('data/mscoco', transform=transform)
dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True, num_workers=4)

pp = PartialConv(1, 6, 4)

# for i_batch, sample_batched in enumerate(dataloader):
#   print(i_batch, sample_batched['image'].size(),
#         sample_batched['mask'].size())

#     # observe 4th batch and stop.
#   if i_batch == 3:
#     plt.imshow(utils.make_grid(sample_batched['mask']).numpy().transpose(1,2,0) * 255)
#     plt.show()
#     break




