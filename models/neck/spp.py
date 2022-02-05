import torch
import torch.nn as nn

from ..basic.conv import Conv


# Spatial Pyramid Pooling
class SPP(nn.Module):
    """
        Spatial Pyramid Pooling
    """
    def __init__(self, c1, c2, e=0.5, kernel_sizes=[5, 9, 13], act='lrelu'):
        super(SPP, self).__init__()
        c_ = int(c1 * e)
        self.cv1 = Conv(c1, c_, k=1, act=act)
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
                for k in kernel_sizes
            ]
        )
        
        self.cv2 = Conv(c_*(len(kernel_sizes) + 1), c2, k=1, act=act)

    def forward(self, x):
        x = self.cv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.cv2(x)

        return x


class SPPBlock(nn.Module):
    """
        Spatial Pyramid Pooling Block
    """
    def __init__(self, c1, c2, e=0.5, kernel_sizes=[5, 9, 13], act='lrelu'):
        super(SPPBlock, self).__init__()
        self.m = nn.Sequential(
            Conv(c1, c1//2, k=1, act=act),
            Conv(c1//2, c1, k=3, p=1, act=act),
            SPP(c1, c1//2, e=e, kernel_sizes=kernel_sizes, act=act),
            Conv(c1//2, c1, k=3, p=1, act=act),
            Conv(c1, c2, k=1, act=act)
        )

        
    def forward(self, x):
        x = self.m(x)

        return x


class SPPBlockCSP(nn.Module):
    """
        CSP Spatial Pyramid Pooling Block
    """
    def __init__(self, c1, c2, e=0.5, kernel_sizes=[5, 9, 13], act='lrelu'):
        super(SPPBlockCSP, self).__init__()
        self.cv1 = Conv(c1, c1//2, k=1, act=act)
        self.cv2 = Conv(c1, c1//2, k=1, act=act)
        self.m = nn.Sequential(
            Conv(c1//2, c1//2, k=3, p=1, act=act),
            SPP(c1//2, c1//2, e=e, kernel_sizes=kernel_sizes, act=act),
            Conv(c1//2, c1//2, k=3, p=1, act=act)
        )
        self.cv3 = Conv(c1, c2, k=1, act=act)

        
    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.m(x2)
        y = self.cv3(torch.cat([x1, x3], dim=1))

        return y


class SPPBlockDW(nn.Module):
    """
        Depth-wise Spatial Pyramid Pooling Block
    """
    def __init__(self, c1, c2, e=0.5, kernel_sizes=[5, 9, 13], act='lrelu'):
        super(SPPBlockDW, self).__init__()
        self.m = nn.Sequential(
            Conv(c1, c1//2, k=1, act=act),
            Conv(c1//2, c1//2, k=3, p=1, g=c1//2, act=act),
            SPP(c1//2, c1//2, e=e, kernel_sizes=kernel_sizes, act=act),
            Conv(c1//2, c1//2, k=3, p=1, g=c1//2, act=act),
            Conv(c1//2, c2, k=1, act=act)
        )
        
    def forward(self, x):
        return self.m(x)
