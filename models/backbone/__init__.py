from .resnet import resnet18, resnet50, resnet101
from .darknet import darknet53
from .cspdarknet_tiny import cspdarknet_tiny
from .cspdarknet53 import cspdarknet53
from .shufflenetv2 import shufflenetv2
from .vit import vit_base_patch16_224


def build_backbone(model_name='r18', pretrained=False, img_size=224):
    if model_name == 'r18':
        print('Backbone: ResNet-18 ...')
        model = resnet18(pretrained=pretrained)
        feature_channels = [128, 256, 512]
        strides = [8, 16, 32]
    elif model_name == 'r50':
        print('Backbone: ResNet-50 ...')
        model = resnet50(pretrained=pretrained)
        feature_channels = [512, 1024, 2048]
        strides = [8, 16, 32]
    elif model_name == 'r101':
        print('Backbone: ResNet-101 ...')
        model = resnet101(pretrained=pretrained)
        feature_channels = [512, 1024, 2048]
        strides = [8, 16, 32]
    elif model_name == 'd53':
        print('Backbone: DarkNet-53 ...')
        model = darknet53(pretrained=pretrained)
        feature_channels = [256, 512, 1024]
        strides = [8, 16, 32]
    elif model_name == 'cspd53':
        print('Backbone: CSPDarkNet-53 ...')
        model = cspdarknet53(pretrained=pretrained)
        feature_channels = [256, 512, 1024]
        strides = [8, 16, 32]
    elif model_name == 'cspd_tiny':
        print('Backbone: CSPDarkNet-Tiny ...')
        model = cspdarknet_tiny(pretrained=pretrained)
        feature_channels = [128, 256, 512]
        strides = [8, 16, 32]
    elif model_name == 'sfnet_v2':
        print('Backbone: ShuffleNet-V2 ...')
        model = shufflenetv2(pretrained=pretrained)
        feature_channels = [116, 232, 464]
        strides = [8, 16, 32]
    elif model_name == 'vit_base_16':
        print('Backbone: ViT_Base_16 ...')
        model = vit_base_patch16_224(img_size=img_size, pretrained=pretrained)
        feature_channels = [None, None, 768]
        strides = [None, None, 16]

    return model, feature_channels, strides
