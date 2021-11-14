import torch
from thop import profile


def FLOPs_and_Params(model, size):
    device = model.device
    x = torch.randn(1, 3, size, size).to(device)

    flops, params = profile(model, inputs=(x, ))
    print('FLOPs : ', flops / 1e9, ' B')
    print('Params : ', params / 1e6, ' M')


if __name__ == "__main__":
    pass
