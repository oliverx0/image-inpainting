import numpy as np 
import random
import torch

class GenerateMask(object):
  """Generates a mask for the given sample."""
  
  MASK_RATIOS =[
    (0.01, 0.1),
    (0.1, 0.2),
    (0.2, 0.3),
    (0.3, 0.4),
    (0.4, 0.5),
    (0.5, 0.6)
  ]

  def random_mask(self, image):
    width, height = image.shape[1], image.shape[2]
    mask_grid = np.ones((width, height), dtype=np.bool)

    mask_ratio = random.choice(self.MASK_RATIOS)
    min_ratio, max_ratio = mask_ratio
    random_ratio = random.uniform(min_ratio, max_ratio)

    return (np.random.rand(width * height) > random_ratio).astype(int).reshape((1, width,height))


  def __call__(self, image):
    return {
      'image': image, 
      'mask': torch.from_numpy(self.random_mask(image))
    }