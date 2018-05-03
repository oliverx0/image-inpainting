import os
import matplotlib.image as mpimg
from torch.utils.data import Dataset

class MaskDataset(Dataset):
  """Face Landmarks dataset."""

  def __init__(self, root_dir, transform=None):
    """
    Args:
        csv_file (string): Path to the csv file with annotations.
        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied
            on a sample.
    """
    self.root_dir = root_dir
    self.images = os.listdir(root_dir)
    self.images.sort()
    self.transform = transform

  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
    img_name = os.path.join(self.root_dir, self.images[idx])
    image = mpimg.imread(img_name)

    if self.transform:
        image = self.transform(image)

    return image
    


