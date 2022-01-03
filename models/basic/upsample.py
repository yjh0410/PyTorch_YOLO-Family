import torch
import torch.nn as nn


class UpSample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corner=None):
        super(UpSample, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corner = align_corner

    def forward(self, x):
        return torch.nn.functional.interpolate(input=x, 
                                               size=self.size, 
                                               scale_factor=self.scale_factor, 
                                               mode=self.mode, 
                                               align_corners=self.align_corner
                                               )
