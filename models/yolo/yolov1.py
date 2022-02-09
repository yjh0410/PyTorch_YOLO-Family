import numpy as np
import torch
import torch.nn as nn

from utils import box_ops

from ..basic.conv import Conv 
from ..neck import build_neck
from ..backbone import build_backbone


class YOLOv1(nn.Module):
    def __init__(self, 
                 cfg=None,
                 device=None, 
                 img_size=None, 
                 num_classes=20, 
                 trainable=False, 
                 conf_thresh=0.001, 
                 nms_thresh=0.6,
                 center_sample=False):
        super(YOLOv1, self).__init__()
        self.cfg = cfg
        self.device = device
        self.img_size = img_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.center_sample = center_sample

        # backbone
        self.backbone, feature_channels, strides = build_backbone(model_name=cfg['backbone'], 
                                                                  pretrained=trainable)
        self.stride = [strides[-1]]
        feature_dim = feature_channels[-1]
        head_dim = 512

        # build grid cell
        self.grid_xy = self.create_grid(img_size)
        
        # neck
        self.neck = build_neck(model=cfg['neck'], in_ch=feature_dim, out_ch=head_dim)

        # head
        self.cls_feat = nn.Sequential(
            Conv(head_dim, head_dim, k=3, p=1, s=1),
            Conv(head_dim, head_dim, k=3, p=1, s=1)
        )
        self.reg_feat = nn.Sequential(
            Conv(head_dim, head_dim, k=3, p=1, s=1),
            Conv(head_dim, head_dim, k=3, p=1, s=1),
            Conv(head_dim, head_dim, k=3, p=1, s=1),
            Conv(head_dim, head_dim, k=3, p=1, s=1)
        )

        # head
        self.obj_pred = nn.Conv2d(head_dim, 1, kernel_size=1)
        self.cls_pred = nn.Conv2d(head_dim, self.num_classes, kernel_size=1)
        self.reg_pred = nn.Conv2d(head_dim, 4, kernel_size=1)

        if self.trainable:
            self.init_bias()


    def init_bias(self):               
        # init bias
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        nn.init.constant_(self.obj_pred.bias, bias_value)


    def create_grid(self, img_size):
        """img_size: [H, W]"""
        img_h = img_w = img_size
        # generate grid cells
        fmp_h, fmp_w = img_h // self.stride[0], img_w // self.stride[0]
        grid_y, grid_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
        # [H, W, 2] -> [HW, 2]
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float().view(-1, 2)
        # [HW, 2] -> [1, HW, 2]
        grid_xy = grid_xy.unsqueeze(0).to(self.device)

        return grid_xy


    def set_grid(self, img_size):
        self.grid_xy = self.create_grid(img_size)
        self.img_size = img_size


    def decode_bbox(self, reg_pred):
        """reg_pred: [B, N, 4]"""
        # txty -> xy
        if self.center_sample:
            xy_pred = reg_pred[..., :2].sigmoid() * 2.0 - 1.0 + self.grid_xy
        else:
            xy_pred = reg_pred[..., :2].sigmoid() + self.grid_xy
        # twth -> wh
        wh_pred = reg_pred[..., 2:].exp()
        xywh_pred = torch.cat([xy_pred, wh_pred], dim=-1)
        # xywh -> x1y1x2y2
        x1y1_pred = xywh_pred[..., :2] - xywh_pred[..., 2:] / 2
        x2y2_pred = xywh_pred[..., :2] + xywh_pred[..., 2:] / 2
        box_pred = torch.cat([x1y1_pred, x2y2_pred], dim=-1)
        # rescale bbox
        box_pred = box_pred * self.stride[0]

        return box_pred


    def nms(self, dets, scores):
        """"Pure Python NMS YOLOv4."""
        x1 = dets[:, 0]  #xmin
        y1 = dets[:, 1]  #ymin
        x2 = dets[:, 2]  #xmax
        y2 = dets[:, 3]  #ymax

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            # compute iou
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-14)
            #reserve all the boundingbox whose ovr less than thresh
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep


    def postprocess(self, bboxes, scores):
        """
        bboxes: (N, 4), bsize = 1
        scores: (N, C), bsize = 1
        """

        cls_inds = np.argmax(scores, axis=1)
        scores = scores[(np.arange(scores.shape[0]), cls_inds)]
        
        # threshold
        keep = np.where(scores >= self.conf_thresh)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        # NMS
        keep = np.zeros(len(bboxes), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(cls_inds == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        cls_inds = cls_inds[keep]

        return bboxes, scores, cls_inds


    @torch.no_grad()
    def inference_single_image(self, x):
        # backbone
        x = self.backbone(x)[-1]

        # neck
        x = self.neck(x)

        # head
        cls_feat = self.cls_feat(x)
        reg_feat = self.reg_feat(x)

        # pred
        obj_pred = self.obj_pred(reg_feat)[0]
        cls_pred = self.cls_pred(cls_feat)[0]
        reg_pred = self.reg_pred(reg_feat)[0]

        # [1, H, W] -> [1, HW] -> [HW, 1] -> [HW, 1]
        obj_pred =obj_pred.flatten(1).permute(1, 0).contiguous()
        # [C, H, W] -> [C, HW] -> [HW, C] -> [HW, C]
        cls_pred =cls_pred.flatten(1).permute(1, 0).contiguous()
        # [4, H, W] -> [4, HW] -> [HW, 4]
        reg_pred = reg_pred.flatten(1).permute(1, 0).contiguous()
        box_pred = self.decode_bbox(reg_pred[None])[0] # [B, HW, 4] -> [HW, 4]
        # normalize bbox
        bboxes = torch.clamp(box_pred / self.img_size, 0., 1.)

        # scores
        scores = torch.sigmoid(obj_pred) * torch.softmax(cls_pred, dim=-1)

        # to cpu
        scores = scores.to('cpu').numpy()
        bboxes = bboxes.to('cpu').numpy()

        # post-process
        bboxes, scores, cls_inds = self.postprocess(bboxes, scores)

        return bboxes, scores, cls_inds


    def forward(self, x, targets=None):
        if not self.trainable:
            return self.inference_single_image(x)
        else:
            B = x.size(0)
            C = self.num_classes
            # backbone
            x = self.backbone(x)[-1]

            # neck
            x = self.neck(x)

            # head
            cls_feat = self.cls_feat(x)
            reg_feat = self.reg_feat(x)

            # pred
            obj_pred = self.obj_pred(reg_feat)
            cls_pred = self.cls_pred(cls_feat)
            reg_pred = self.reg_pred(reg_feat)

            # [B, 1, H, W] -> [B, 1, HW] -> [B, HW, 1]
            obj_pred =obj_pred.flatten(2).permute(0, 2, 1).contiguous()
            # [B, C, H, W] -> [B, C, HW] -> [B, HW, C]
            cls_pred =cls_pred.flatten(2).permute(0, 2, 1).contiguous()
            # [B, 4, H, W] -> [B, 4, HW] -> [B, HW, 4]
            reg_pred = reg_pred.flatten(2).permute(0, 2, 1).contiguous()
            box_pred = self.decode_bbox(reg_pred)
            # normalize bbox
            box_pred = box_pred / self.img_size

            # compute giou between prediction bbox and target bbox
            x1y1x2y2_pred = box_pred.view(-1, 4)
            x1y1x2y2_gt = targets[..., 2:6].view(-1, 4)

            # giou: [B, HW,]
            giou_pred = box_ops.giou_score(x1y1x2y2_pred, x1y1x2y2_gt, batch_size=B)

            # we set giou as the target of the objectness
            targets = torch.cat([0.5 * (giou_pred[..., None].clone().detach() + 1.0), targets], dim=-1)

            return obj_pred, cls_pred, giou_pred, targets
