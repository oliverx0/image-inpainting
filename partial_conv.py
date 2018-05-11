import torch
import torch.nn as nn
import torch.nn.functional as F

class PartialConv(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride):

    super(PartialConv, self).__init__()
    
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
    
    # The mask layer will aways have 1 channel.
    # We use broadcasting in mask update operations to avoid using more channels.
    self.mask_update = nn.Conv2d(1, 1, kernel_size, stride)

    # Initialize weights of mask update to 1 and bias to 0
    self.mask_update.weight = torch.nn.Parameter(torch.ones(self.mask_update.weight.shape))
    self.mask_update.bias = torch.nn.Parameter(torch.zeros(self.mask_update.bias.shape))

  def forward(self, image, mask):

    # Concatenate image with mask
    input_tensor = image * mask

    # Mask update -> each value in mask_update result represents sum(M)
    mask = self.mask_update(mask)

    # Run the partial convolution
    # Step 1: Save current bias, reshaped to be able to add it later via broadcasting
    conv_bias = self.conv.bias
    conv_bias = conv_bias.reshape(1, conv_bias.shape[0], 1, 1)

    # Step 2: Set current bias to 0
    self.conv.bias = torch.nn.Parameter(torch.zeros(self.conv.bias.shape))

    # Step 3: Run convolution
    result = self.conv(input_tensor)

    # Step 4: Update mask to be 1/sum(M), where M = "partial mask"
    mask[mask != 0] = 1/mask[mask != 0]

    # Step 5: Multiply result with 1/sum(M)
    result *= mask 

    # Step 6: Add bias to result
    # One might think of doing indexing based on result != 0 by repeating conv_bias. 
    # Dont. You wont know if result is 0 because of mask or because of value. 
    result += conv_bias

    # Step 7: Update mask to 1 where sum(M) > 0
    mask[mask != 0] = 1

    # Step 8: Result = 0 if sum(M) > 0
    # TODO: See how it works removing this step (except in last layer)
    result *= mask 

    return result, mask