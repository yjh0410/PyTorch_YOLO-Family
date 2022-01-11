"""
    This is a CSPDarkNet-53 with Mish.
"""
import os
import torch
import torch.nn as nn


def ConvNormActivation(inplanes,
                       planes,
                       kernel_size=3,
                       stride=1,
                       padding=0,
                       dilation=1,
                       groups=1):
    """
    A help function to build a 'conv-bn-activation' module
    """
    layers = []
    layers.append(nn.Conv2d(inplanes,
                            planes,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            dilation=dilation,
                            groups=groups,
                            bias=False))
    layers.append(nn.BatchNorm2d(planes, eps=1e-4, momentum=0.03))
    layers.append(nn.Mish(inplace=True))
    return nn.Sequential(*layers)


class DarkBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 dilation=1,
                 downsample=None):
        """Residual Block for DarkNet.
        This module has the dowsample layer (optional),
        1x1 conv layer and 3x3 conv layer.
        """
        super(DarkBlock, self).__init__()

        self.downsample = downsample

        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-4, momentum=0.03)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-4, momentum=0.03)

        self.conv1 = nn.Conv2d(
            planes,
            inplanes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        self.conv2 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            bias=False
        )

        self.activation = nn.Mish(inplace=True)

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out += identity

        return out


class CrossStagePartialBlock(nn.Module):
    """CSPNet: A New Backbone that can Enhance Learning Capability of CNN.
    Refer to the paper for more details: https://arxiv.org/abs/1911.11929.
    In this module, the inputs go throuth the base conv layer at the first,
    and then pass the two partial transition layers.
    1. go throuth basic block (like DarkBlock)
        and one partial transition layer.
    2. go throuth the other partial transition layer.
    At last, They are concat into fuse transition layer.
    Args:
        inplanes (int): number of input channels.
        planes (int): number of output channels
        stage_layers (nn.Module): the basic block which applying CSPNet.
        is_csp_first_stage (bool): Is the first stage or not.
            The number of input and output channels in the first stage of
            CSPNet is different from other stages.
        dilation (int): conv dilation
        stride (int): stride for the base layer
    """

    def __init__(self,
                 inplanes,
                 planes,
                 stage_layers,
                 is_csp_first_stage,
                 dilation=1,
                 stride=2):
        super(CrossStagePartialBlock, self).__init__()

        self.base_layer = ConvNormActivation(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation
        )
        self.partial_transition1 = ConvNormActivation(
            inplanes=planes,
            planes=inplanes if not is_csp_first_stage else planes,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.stage_layers = stage_layers

        self.partial_transition2 = ConvNormActivation(
            inplanes=inplanes if not is_csp_first_stage else planes,
            planes=inplanes if not is_csp_first_stage else planes,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.fuse_transition = ConvNormActivation(
            inplanes=planes if not is_csp_first_stage else planes * 2,
            planes=planes,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x):
        x = self.base_layer(x)

        out1 = self.partial_transition1(x)

        out2 = self.stage_layers(x)
        out2 = self.partial_transition2(out2)

        out = torch.cat([out2, out1], dim=1)
        out = self.fuse_transition(out)

        return out


def make_cspdark_layer(block,
                       inplanes,
                       planes,
                       num_blocks,
                       is_csp_first_stage,
                       dilation=1):
    downsample = ConvNormActivation(
        inplanes=planes,
        planes=planes if is_csp_first_stage else inplanes,
        kernel_size=1,
        stride=1,
        padding=0
    )

    layers = []
    for i in range(0, num_blocks):
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes if is_csp_first_stage else inplanes,
                downsample=downsample if i == 0 else None,
                dilation=dilation
            )
        )
    return nn.Sequential(*layers)


class CSPDarkNet53(nn.Module):
    """CSPDarkNet backbone.
    Refer to the paper for more details: https://arxiv.org/pdf/1804.02767
    Args:
        depth (int): Depth of Darknet, from {53}.
        num_stages (int): Darknet stages, normally 5.
        with_csp (bool): Use cross stage partial connection or not.
        out_features (List[str]): Output features.
        norm_type (str): type of normalization layer.
        res5_dilation (int): dilation for the last stage
    """

    def __init__(self):
        super(CSPDarkNet53, self).__init__()
     
        self.block =  DarkBlock
        self.stage_blocks = (1, 2, 8, 8, 4)
        self.with_csp = True
        self.inplanes = 32

        self._make_stem_layer()

        dark_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            planes = 64 * 2 ** i
            dilation = 1
            stride = 2
            layer = make_cspdark_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                is_csp_first_stage=True if i == 0 else False,
                dilation=dilation
            )
            layer = CrossStagePartialBlock(
                self.inplanes,
                planes,
                stage_layers=layer,
                is_csp_first_stage=True if i == 0 else False,
                dilation=dilation,
                stride=stride
            )
            self.inplanes = planes
            dark_layers.append(layer)

        self.dark_layers = nn.ModuleList(dark_layers)


    def _make_stem_layer(self):
        self.conv1 = nn.Conv2d(
            3,
            self.inplanes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-4, momentum=0.03)
        self.act1 = nn.Mish(inplace=True)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        c1 = self.dark_layers[0](x)
        c2 = self.dark_layers[1](c1)
        c3 = self.dark_layers[2](c2)
        c4 = self.dark_layers[3](c3)
        c5 = self.dark_layers[4](c4)

        return c3, c4, c5


def cspdarknet53(pretrained=False):
    """
    Create a CSPDarkNet.
    """
    model = CSPDarkNet53()
    if pretrained:
        print('Loading the pretrained model ...')
        path_to_weight = os.path.dirname(os.path.abspath(__file__)) + '/weights/cspdarknet53/cspdarknet53.pth'
        model.load_state_dict(torch.load(path_to_weight, map_location='cpu'), strict=False)

    return model


if __name__=='__main__':
    img_size = 512
    input = torch.randn(1, 3, img_size, img_size)
    
    model = cspdarknet53(pretrained=True)
    output = model(input)
    for y in output:
        print(y.size())

