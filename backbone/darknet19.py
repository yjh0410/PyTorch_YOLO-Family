import torch
import torch.nn as nn
import os


__all__ = ['darknet19']


class Conv_BN_LeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, padding=0, stride=1, dilation=1):
        super(Conv_BN_LeakyReLU, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, padding=padding, stride=stride, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.convs(x)


class DarkNet_19(nn.Module):
    def __init__(self):
        print("Initializing the darknet19 network ......")
        
        super(DarkNet_19, self).__init__()
        # backbone network : DarkNet-19
        # output : stride = 2, c = 32
        self.conv_1 = nn.Sequential(
            Conv_BN_LeakyReLU(3, 32, 3, 1),
            nn.MaxPool2d((2,2), 2),
        )

        # output : stride = 2, c = 64
        self.conv_2 = nn.Sequential(
            Conv_BN_LeakyReLU(32, 64, 3, 1)
        )

        # output : stride = 4, c = 128
        self.maxpool_2 = nn.MaxPool2d((2,2), 2)
        self.conv_3 = nn.Sequential(
            Conv_BN_LeakyReLU(64, 128, 3, 1),
            Conv_BN_LeakyReLU(128, 64, 1),
            Conv_BN_LeakyReLU(64, 128, 3, 1)
        )

        # output : stride = 8, c = 256
        self.maxpool_3 = nn.MaxPool2d((2,2), 2)
        self.conv_4 = nn.Sequential(
            Conv_BN_LeakyReLU(128, 256, 3, 1),
            Conv_BN_LeakyReLU(256, 128, 1),
            Conv_BN_LeakyReLU(128, 256, 3, 1),
        )

        # output : stride = 16, c = 512
        self.maxpool_4 = nn.MaxPool2d((2, 2), 2)
        self.conv_5 = nn.Sequential(
            Conv_BN_LeakyReLU(256, 512, 3, 1),
            Conv_BN_LeakyReLU(512, 256, 1),
            Conv_BN_LeakyReLU(256, 512, 3, 1),
            Conv_BN_LeakyReLU(512, 256, 1),
            Conv_BN_LeakyReLU(256, 512, 3, 1),
        )
        
        # output : stride = 32, c = 1024
        self.maxpool_5 = nn.MaxPool2d((2, 2), 2)
        self.conv_6 = nn.Sequential(
            Conv_BN_LeakyReLU(512, 1024, 3, 1),
            Conv_BN_LeakyReLU(1024, 512, 1),
            Conv_BN_LeakyReLU(512, 1024, 3, 1),
            Conv_BN_LeakyReLU(1024, 512, 1),
            Conv_BN_LeakyReLU(512, 1024, 3, 1)
        )


    def forward(self, x):
        c1 = self.conv_1(x)
        c1 = self.conv_2(c1)
        c2 = self.conv_3(self.maxpool_2(c1))
        c3 = self.conv_4(self.maxpool_2(c2))
        c4 = self.conv_5(self.maxpool_4(c3))
        c5 = self.conv_6(self.maxpool_5(c4))

        return c5


def darknet19(pretrained=False, **kwargs):
    """Constructs a darknet-19 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DarkNet_19()
    if pretrained:
        print('Loading the pretrained model ...')
        path_to_dir = os.path.dirname(os.path.abspath(__file__))
        print('Loading the darknet19 ...')
        model.load_state_dict(torch.load(path_to_dir + '/weights/darknet19/darknet19.pth'), strict=False)
    return model
