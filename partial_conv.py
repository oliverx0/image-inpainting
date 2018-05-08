import torch
import torch.nn as nn
import torch.nn.functional as F

class PartialConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):

    	super(PartialConv, self).__init__(
    		in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return LinearFunction.apply(input, self.weight, self.bias)

    # def extra_repr(self):
    #     # (Optional)Set the extra information about this module. You can test
    #     # it by printing an object of this class.
    #     return 'in_features={}, out_features={}, bias={}'.format(
    #         self.in_features, self.out_features, self.bias is not None
    #     )