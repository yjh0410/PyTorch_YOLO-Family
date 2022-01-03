from .spp import SPP
from .dilated_encoder import DilatedEncoder
from ..basic.conv import ConvBlocks


def build_neck(model, in_ch, out_ch):
    if model == 'conv_blocks':
        print("Neck: ConvBlocks")
        neck = ConvBlocks(c1=in_ch, c2=out_ch)
    elif model == 'spp':
        print("Neck: SPP")
        neck = SPP(c1=in_ch, c2=out_ch)
    elif model == 'dilated_encoder':
        print("Neck: Dilated Encoder")
        neck = DilatedEncoder(c1=in_ch, c2=out_ch)

    return neck
