#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import os
import torch
import torch.nn as nn


class SiLU(nn.Module):
    """export-friendly version of nn.SiLU()"""

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(
        self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"
    ):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList(
            [
                nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
                for ks in kernel_sizes
            ]
        )
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class CSPLayer(nn.Module):
    """C3 in yolov5, CSP Bottleneck with 3 convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(
                hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act
            )
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return self.conv(x)


# CSPDarkNet
class CSPDarknet(nn.Module):
    def __init__(
        self,
        dep_mul,
        wid_mul,
        out_features=("dark3", "dark4", "dark5"),
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # stem
        self.stem = Focus(3, base_channels, ksize=3, act=act)

        # dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark5
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
                depthwise=depthwise,
                act=act,
            ),
        )


    def freeze_stage(self):
        # Because the YOLOX-Backbone has been trained on COCO, we freeze all stages to save computation.
        print('freeze all stage of YOLOX-Backbone ...')
        for m in self.parameters():
            m.requires_grad = False
        

    def forward(self, x):
        outputs = {}
        c1 = self.stem(x)
        c2 = self.dark2(c1)
        c3 = self.dark3(c2)
        c4 = self.dark4(c3)
        c5 = self.dark5(c4)

        return c3, c4, c5


def yolox_cspdarknet_s(pretrained=False, freeze=False):
    # build backbone
    backbone = CSPDarknet(dep_mul=0.33, wid_mul=0.5, depthwise=False, act='silu')

    # load weight
    if pretrained:
        print('Loading pretrained cspdarknet_s ...')
        path_to_dir = os.path.dirname(os.path.abspath(__file__))
        path_to_weight = path_to_dir + '/weights/yolox_backbone/yolox_cspdarknet_s.pth'
        checkpoint = torch.load(path_to_weight, map_location='cpu')
        backbone.load_state_dict(checkpoint)

    # freeze stage
    if freeze:
        backbone.freeze_stage()

    return backbone


def yolox_cspdarknet_m(pretrained=False, freeze=False):
    # build backbone
    backbone = CSPDarknet(dep_mul=0.67, wid_mul=0.75, depthwise=False, act='silu')

    # load weight
    if pretrained:
        print('Loading pretrained cspdarknet_m ...')
        path_to_dir = os.path.dirname(os.path.abspath(__file__))
        path_to_weight = path_to_dir + '/weights/yolox_backbone/yolox_cspdarknet_m.pth'
        checkpoint = torch.load(path_to_weight, map_location='cpu')
        backbone.load_state_dict(checkpoint)

    # freeze stage
    if freeze:
        backbone.freeze_stage()

    return backbone


def yolox_cspdarknet_l(pretrained=False, freeze=False):
    # build backbone
    backbone = CSPDarknet(dep_mul=1.0, wid_mul=1.0, depthwise=False, act='silu')

    # load weight
    if pretrained:
        print('Loading pretrained cspdarknet_l ...')
        path_to_dir = os.path.dirname(os.path.abspath(__file__))
        path_to_weight = path_to_dir + '/weights/yolox_backbone/yolox_cspdarknet_l.pth'
        checkpoint = torch.load(path_to_weight, map_location='cpu')
        backbone.load_state_dict(checkpoint)

    # freeze stage
    if freeze:
        backbone.freeze_stage()

    return backbone


def yolox_cspdarknet_x(pretrained=False, freeze=False):
    # build backbone
    backbone = CSPDarknet(dep_mul=1.33, wid_mul=1.25, depthwise=False, act='silu')

    # load weight
    if pretrained:
        print('Loading pretrained cspdarknet_x ...')
        path_to_dir = os.path.dirname(os.path.abspath(__file__))
        path_to_weight = path_to_dir + '/weights/yolox_backbone/yolox_cspdarknet_x.pth'
        checkpoint = torch.load(path_to_weight, map_location='cpu')
        backbone.load_state_dict(checkpoint)

    # freeze stage
    if freeze:
        backbone.freeze_stage()

    return backbone


def yolox_cspdarknet_tiny(pretrained=False, freeze=False):
    # build backbone
    backbone = CSPDarknet(dep_mul=0.33, wid_mul=0.375, depthwise=False, act='silu')

    # load weight
    if pretrained:
        print('Loading pretrained cspdarknet_tiny ...')
        path_to_dir = os.path.dirname(os.path.abspath(__file__))
        path_to_weight = path_to_dir + '/weights/yolox_backbone/yolox_cspdarknet_tiny.pth'
        checkpoint = torch.load(path_to_weight, map_location='cpu')
        backbone.load_state_dict(checkpoint)

    # freeze stage
    if freeze:
        backbone.freeze_stage()

    return backbone


def yolox_cspdarknet_nano(pretrained=False, freeze=False):
    # build backbone
    backbone = CSPDarknet(dep_mul=0.33, wid_mul=0.25, depthwise=True, act='silu')

    # load weight
    if pretrained:
        print('Loading pretrained cspdarknet_nano ...')
        path_to_dir = os.path.dirname(os.path.abspath(__file__))
        path_to_weight = path_to_dir + '/weights/yolox_backbone/yolox_cspdarknet_nano.pth'
        checkpoint = torch.load(path_to_weight, map_location='cpu')
        backbone.load_state_dict(checkpoint)

    # freeze stage
    if freeze:
        backbone.freeze_stage()

    return backbone


if __name__ == '__main__':
    import time
    net = yolox_cspdarknet_nano(pretrained=True)
    x = torch.randn(1, 3, 224, 224)
    t0 = time.time()
    outputs = net(x)
    t1 = time.time()
    print('Time: ', t1 - t0)
    for y in outputs:
        print(y.shape)
