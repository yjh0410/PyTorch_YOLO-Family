import math
import torch


def iou_score(bboxes_a, bboxes_b, batch_size):
    """
        Input:\n
        bboxes_a : [B*N, 4] = [x1, y1, x2, y2] \n
        bboxes_b : [B*N, 4] = [x1, y1, x2, y2] \n

        Output:\n
        iou : [B, N] = [iou, ...] \n
    """
    tl = torch.max(bboxes_a[:, :2], bboxes_b[:, :2])
    br = torch.min(bboxes_a[:, 2:], bboxes_b[:, 2:])
    area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
    area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)

    en = (tl < br).type(tl.type()).prod(dim=1)
    area_i = torch.prod(br - tl, 1) * en  # * ((tl < br).all())
    iou = area_i / (area_a + area_b - area_i + 1e-14)

    return iou.view(batch_size, -1)


def giou_score(bboxes_a, bboxes_b, batch_size):
    """
        bbox_1 : [B*N, 4] = [x1, y1, x2, y2]
        bbox_2 : [B*N, 4] = [x1, y1, x2, y2]
    """
    # iou
    tl = torch.max(bboxes_a[:, :2], bboxes_b[:, :2])
    br = torch.min(bboxes_a[:, 2:], bboxes_b[:, 2:])
    area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
    area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)

    en = (tl < br).type(tl.type()).prod(dim=1)
    area_i = torch.prod(br - tl, 1) * en  # * ((tl < br).all())
    area_u = area_a + area_b - area_i
    iou = (area_i / (area_u + 1e-14)).clamp(0)
    
    # giou
    tl = torch.min(bboxes_a[:, :2], bboxes_b[:, :2])
    br = torch.max(bboxes_a[:, 2:], bboxes_b[:, 2:])
    en = (tl < br).type(tl.type()).prod(dim=1)
    area_c = torch.prod(br - tl, 1) * en  # * ((tl < br).all())

    giou = (iou - (area_c - area_u) / (area_c + 1e-14))

    return giou.view(batch_size, -1)


def ciou_score(bboxes_a, bboxes_b, batch_size):
    """
        Input:\n
        bboxes_a : [B*N, 4] = [x1, y1, x2, y2] \n
        bboxes_b : [B*N, 4] = [x1, y1, x2, y2] \n

        Output:\n
        iou : [B, N] = [ciou, ...] \n
    """
    tl = torch.max(bboxes_a[:, :2], bboxes_b[:, :2])
    br = torch.min(bboxes_a[:, 2:], bboxes_b[:, 2:])
    area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
    area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)

    en = (tl < br).type(tl.type()).prod(dim=1)
    area_i = torch.prod(br - tl, 1) * en  # * ((tl < br).all())
    iou = area_i / (area_a + area_b - area_i + 1e-7)

    cw = torch.max(bboxes_a[..., 2], bboxes_b[..., 2]) - torch.min(bboxes_a[..., 0], bboxes_b[..., 0])
    ch = torch.max(bboxes_a[..., 3], bboxes_b[..., 3]) - torch.min(bboxes_a[..., 1], bboxes_b[..., 1])

    c2 = cw ** 2 + ch ** 2 + 1e-7
    rho2 = ((bboxes_b[..., 0] + bboxes_b[..., 2] - bboxes_a[..., 0] - bboxes_a[..., 2]) ** 2 +
            (bboxes_b[..., 1] + bboxes_b[..., 3] - bboxes_a[..., 1] - bboxes_a[..., 3]) ** 2) / 4
    w1 = bboxes_a[..., 2] - bboxes_a[..., 0]
    h1 = bboxes_a[..., 3] - bboxes_a[..., 1]
    w2 = bboxes_b[..., 2] - bboxes_b[..., 0]
    h2 = bboxes_b[..., 3] - bboxes_b[..., 1]
    v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
    with torch.no_grad():
        alpha = v / (v - iou + (1. + 1e-7))

    ciou = iou - (rho2 / c2 + v * alpha)

    return ciou.view(batch_size, -1)


if __name__ == '__main__':
    bboxes_a = torch.tensor([[10, 10, 20, 20]])
    bboxes_b = torch.tensor([[13, 15, 27, 25]])
    iou = iou_score(bboxes_a, bboxes_b, 1)
    print(iou)
    giou = giou_score(bboxes_a, bboxes_b, 1)
    print(giou)
    ciou = ciou_score(bboxes_a, bboxes_b, 1)
    print(ciou)
