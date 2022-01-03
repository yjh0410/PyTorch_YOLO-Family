import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, p=0, s=1, d=1, g=1, act=True, bias=False):
        super(Conv, self).__init__()
        if act:
            self.convs = nn.Sequential(
                nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=bias),
                nn.BatchNorm2d(c2),
                nn.LeakyReLU(0.1, inplace=True)
            )
        else:
            self.convs = nn.Sequential(
                nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=bias),
                nn.BatchNorm2d(c2)
            )

    def forward(self, x):
        return self.convs(x)


# ConvBlocks
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
