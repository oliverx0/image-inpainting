import torch
import torch.nn as nn
import torch.nn.functional as F

class PartialConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
    	self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
    	
    	# We use broadcasting in mask operations to avoid using more channels
    	self.mask_update = nn.Conv2d(1, 1, kernel_size)
    	self.mask_update.weight = torch.nn.Parameter(torch.ones(self.mask_update.weight.shape))
        self.mask_update.bias = torch.nn.Parameter(torch.zeros(self.mask_update.bias.shape))

    def forward(self, image, mask):

    	# Partial convolution
    	input_tensor = image * mask * (1.0/mask.sum().item())
    	result = self.conv(input_tensor)
    	result[result == self.conv.bias] = 0

    	# Mask update
    	mask = self.mask_update(mask)
    	mask[mask != 0] = 1

        return result, mask