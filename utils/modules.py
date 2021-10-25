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


class ConvBlocks(nn.Module):
    def __init__(self, c1, c2):
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


# Spatial Pyramid Pooling
class SPP(nn.Module):
    """
        Spatial Pyramid Pooling
    """
    def __init__(self, c1, c2, e=0.5):
        super(SPP, self).__init__()
        c_ = int(c1 * e)
        self.cv1 = Conv(c1, c_, k=1)
        self.cv2 = Conv(c_*4, c2, k=1)

    def forward(self, x):
        x = self.cv1(x)
        x_1 = torch.nn.functional.max_pool2d(x, 5, stride=1, padding=2)
        x_2 = torch.nn.functional.max_pool2d(x, 9, stride=1, padding=4)
        x_3 = torch.nn.functional.max_pool2d(x, 13, stride=1, padding=6)
        x = torch.cat([x, x_1, x_2, x_3], dim=1)
        x = self.cv2(x)

        return x


# Copy from yolov5
class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, d=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k=1)
        self.cv2 = Conv(c_, c2, k=3, p=d, g=g, d=d)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


# Copy from yolov5
class BottleneckCSP(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k=1)
        self.cv2 = Conv(c1, c_, k=1)
        self.cv3 = Conv(2 * c_, c2, k=1)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


# Dilated Encoder
class DilatedBottleneck(nn.Module):
    def __init__(self, c, d=1, e=0.5, act=True):
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
    def __init__(self, c1, c2, act=True, dilation_list=[2, 4, 6, 8]):
        super(DilatedEncoder, self).__init__()
        self.projector = nn.Sequential(
            Conv(c1, c2, k=1, act=False),
            Conv(c2, c2, k=3, p=1, act=False)
        )
        encoders = []
        for d in dilation_list:
            encoders.append(DilatedBottleneck(c=c2, d=d, act=act))
        self.encoders = nn.Sequential(*encoders)

    def forward(self, x):
        x = self.projector(x)
        x = self.encoders(x)

        return x
