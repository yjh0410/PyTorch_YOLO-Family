import torch
import torch.nn as nn
from ..basic.conv import Conv, ConvBlocks
from ..basic.upsample import UpSample
from ..basic.bottleneck_csp import BottleneckCSP


# YoloFPN
class YoloFPN(nn.Module):
    def __init__(self, in_dim=[512, 1024, 2048]):
        super(YoloFPN, self).__init__()
        c3, c4, c5 = in_dim
        # head
        # P3/8-small
        self.head_convblock_0 = ConvBlocks(c5, c5//2)
        self.head_conv_0 = Conv(c5//2, c4//2, k=1)
        self.head_upsample_0 = UpSample(scale_factor=2)
        self.head_conv_1 = Conv(c5//2, c5, k=3, p=1)

        # P4/16-medium
        self.head_convblock_1 = ConvBlocks(c4 + c4//2, c4//2)
        self.head_conv_2 = Conv(c4//2, c3//2, k=1)
        self.head_upsample_1 = UpSample(scale_factor=2)
        self.head_conv_3 = Conv(c4//2, c4, k=3, p=1)

        # P8/32-large
        self.head_convblock_2 = ConvBlocks(c3 + c3//2, c3//2)
        self.head_conv_4 = Conv(c3//2, c3, k=3, p=1)


    def forward(self, features):
        c3, c4, c5 = features
        
        # p5/32
        p5 = self.head_convblock_0(c5)
        p5_up = self.head_upsample_0(self.head_conv_0(p5))
        p5 = self.head_conv_1(p5)

        # p4/16
        p4 = self.head_convblock_1(torch.cat([c4, p5_up], dim=1))
        p4_up = self.head_upsample_1(self.head_conv_2(p4))
        p4 = self.head_conv_3(p4)

        # P3/8
        p3 = self.head_convblock_2(torch.cat([c3, p4_up], dim=1))
        p3 = self.head_conv_4(p3)

        return [p3, p4, p5]


# YoloPaFPN
class YoloPaFPN(nn.Module):
    def __init__(self, 
                 in_dim=[256, 512, 1024], 
                 depth=1.0, 
                 depthwise=False,
                 act='silu'):
        super(YoloPaFPN, self).__init__()
        c3, c4, c5 = in_dim
        nblocks = int(3 * depth)
        self.head_conv_0 = Conv(c5, c5//2, k=1, act=act)  # 10
        self.head_upsample_0 = UpSample(scale_factor=2)
        self.head_csp_0 = BottleneckCSP(c4 + c5//2, c4, n=nblocks, shortcut=False, depthwise=depthwise, act=act)

        # P3/8-small
        self.head_conv_1 = Conv(c4, c4//2, k=1, act=act)  # 14
        self.head_upsample_1 = UpSample(scale_factor=2)
        self.head_csp_1 = BottleneckCSP(c3 + c4//2, c3, n=nblocks, shortcut=False, depthwise=depthwise, act=act)

        # P4/16-medium
        self.head_conv_2 = Conv(c3, c3, k=3, p=1, s=2, depthwise=depthwise, act=act)
        self.head_csp_2 = BottleneckCSP(c3 + c4//2, c4, n=nblocks, shortcut=False, depthwise=depthwise, act=act)

        # P8/32-large
        self.head_conv_3 = Conv(c4, c4, k=3, p=1, s=2, depthwise=depthwise, act=act)
        self.head_csp_3 = BottleneckCSP(c4 + c5//2, c5, n=nblocks, shortcut=False, depthwise=depthwise)


    def forward(self, features):
        c3, c4, c5 = features

        c6 = self.head_conv_0(c5)
        c7 = self.head_upsample_0(c6)   # s32->s16
        c8 = torch.cat([c7, c4], dim=1)
        c9 = self.head_csp_0(c8)
        # P3/8
        c10 = self.head_conv_1(c9)
        c11 = self.head_upsample_1(c10)   # s16->s8
        c12 = torch.cat([c11, c3], dim=1)
        c13 = self.head_csp_1(c12)  # to det
        # p4/16
        c14 = self.head_conv_2(c13)
        c15 = torch.cat([c14, c10], dim=1)
        c16 = self.head_csp_2(c15)  # to det
        # p5/32
        c17 = self.head_conv_3(c16)
        c18 = torch.cat([c17, c6], dim=1)
        c19 = self.head_csp_3(c18)  # to det

        return [c13, c16, c19] # [P3, P4, P5]


# build Head
def build_fpn(model_name='yolofpn', 
              in_dim=[256, 512, 1024], 
              depth=1.0, 
              depthwise=False, 
              act='silu'):
    if model_name == 'yolofpn':
        print("Head: YoloFPN ...")
        return YoloFPN(in_dim)
        
    elif model_name == 'yolopafpn':
        print('Head: YoloPaFPN ...')
        return YoloPaFPN(in_dim, depth, depthwise, act)
    
    else:
        print("Unknown FPN version ...")
        exit()
