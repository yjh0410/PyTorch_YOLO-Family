import torch
import torch.nn as nn
from ..basic.conv import Conv


# Dilated Encoder
class DilatedBottleneck(nn.Module):
    def __init__(self, c, d=1, e=0.5, act='lrelu'):
        super(DilatedBottleneck, self).__init__()
        c_ = int(c * e)
        self.branch = nn.Sequential(
            Conv(c, c_, k=1, act=act),
            Conv(c_, c_, k=3, p=d, d=d, act=act),
            Conv(c_, c, k=1, act=act)
        )

    def forward(self, x):
        return x + self.branch(x)


class DilatedEncoder(nn.Module):
    """ DilateEncoder """
    def __init__(self, c1, c2, act='lrelu', dilation_list=[2, 4, 6, 8]):
        super(DilatedEncoder, self).__init__()
        self.projector = nn.Sequential(
            Conv(c1, c2, k=1, act=None),
            Conv(c2, c2, k=3, p=1, act=None)
        )
        encoders = []
        for d in dilation_list:
            encoders.append(DilatedBottleneck(c=c2, d=d, act=act))
        self.encoders = nn.Sequential(*encoders)

    def forward(self, x):
        x = self.projector(x)
        x = self.encoders(x)

        return x
