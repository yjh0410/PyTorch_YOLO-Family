import torch
import torch.nn as nn


def get_activation(name="lrelu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name is None:
        module = nn.Identity()
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


# Basic conv layer
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, p=0, s=1, d=1, g=1, act='lrelu', depthwise=False, bias=False):
        super(Conv, self).__init__()
        if depthwise:
            assert c1 == c2
            self.convs = nn.Sequential(
                nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=c1, bias=bias),
                nn.BatchNorm2d(c2),
                get_activation(name=act),
                nn.Conv2d(c2, c2, kernel_size=1, bias=bias),
                nn.BatchNorm2d(c2),
                get_activation(name=act)
            )
        else:
            self.convs = nn.Sequential(
                nn.Conv2d(c1, c2, k, stride=s, padding=p, dilation=d, groups=g, bias=bias),
                nn.BatchNorm2d(c2),
                get_activation(name=act)
            )

    def forward(self, x):
        return self.convs(x)


# ConvBlocks
class ConvBlocks(nn.Module):
    def __init__(self, c1, c2, act='lrelu'):  # in_channels, inner_channels
        super().__init__()
        c_ = c2 *2
        self.convs = nn.Sequential(
            Conv(c1, c2, k=1, act=act),
            Conv(c2, c_, k=3, p=1, act=act),
            Conv(c_, c2, k=1, act=act),
            Conv(c2, c_, k=3, p=1, act=act),
            Conv(c_, c2, k=1, act=act)
        )

    def forward(self, x):
        return self.convs(x)
