import torch
import torch.nn as nn

from ..basic.conv import Conv


class ConvBlocks(nn.Module):
    def __init__(self, c1, c2):  # in_channels, inner_channels
        super().__init__()
        c_ = c2 *2
        self.convs = nn.Sequential(
            Conv(c1, c2, k=1),
            Conv(c2, c_, k=3, p=1),
            Conv(c_, c2, k=1),
            Conv(c2, c_, k=3, p=1),
            Conv(c_, c2, k=1)
        )

    def forward(self, x):
        return self.convs(x)
