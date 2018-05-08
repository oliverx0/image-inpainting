import os
from PIL import Image
from torch.utils.data import Dataset

class MaskDataset(Dataset):
  """Face Landmarks dataset."""

  def __init__(self, root_dir, transform=None):
    """
    Args:
        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be appliedon a sample.
    """
    self.root_dir = root_dir
    self.images = os.listdir(root_dir)
    self.images.sort()
    self.transform = transform

  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
    img_name = os.path.join(self.root_dir, self.images[idx])
    image = Image.open(img_name).convert("RGB")

    if self.transform:
        image = self.transform(image)

    return image
    


