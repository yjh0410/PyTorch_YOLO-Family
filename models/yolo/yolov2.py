import numpy as np
import torch
import torch.nn as nn

from utils import box_ops
from utils import criterion

from ..basic.conv import Conv 
from ..neck import build_neck
from ..backbone import build_backbone



class YOLOv2(nn.Module):
    def __init__(self, 
                 cfg=None,
                 device=None, 
                 img_size=None, 
                 num_classes=20, 
                 trainable=False, 
                 conf_thresh=0.001, 
                 nms_thresh=0.6,
                 center_sample=False):
        super(YOLOv2, self).__init__()
        self.cfg = cfg
        self.device = device
        self.img_size = img_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.center_sample = center_sample
        self.anchor_size = torch.tensor(cfg["anchor_size"])  # [KA, 2]
        self.num_anchors = len(cfg["anchor_size"])

        # backbone
        self.backbone, feature_channels, strides = build_backbone(model_name=cfg['backbone'], 
                                                                  pretrained=trainable)
        self.stride = [strides[-1]]
        feature_dim = feature_channels[-1]
        head_dim = 512

        # build grid cell
        self.grid_xy, self.anchor_wh = self.create_grid(img_size)

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
        self.obj_pred = nn.Conv2d(head_dim, self.num_anchors * 1, kernel_size=1)
        self.cls_pred = nn.Conv2d(head_dim, self.num_anchors * self.num_classes, kernel_size=1)
        self.reg_pred = nn.Conv2d(head_dim, self.num_anchors * 4, kernel_size=1)

        if self.trainable:
            # init bias
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
        # [HW, 2] -> [1, HW, 1, 2]   
        grid_xy = grid_xy[None, :, None, :].to(self.device)
        # [1, HW, 1, 2]
        anchor_wh = self.anchor_size.repeat(fmp_h*fmp_w, 1, 1).unsqueeze(0).to(self.device)

        return grid_xy, anchor_wh


    def set_grid(self, img_size):
        self.grid_xy, self.anchor_wh = self.create_grid(img_size)
        self.img_size = img_size


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


    def decode_bbox(self, reg_pred):
        """reg_pred: [B, N, KA, 4]"""
        B = reg_pred.size(0)
        # txty -> cxcy
        if self.center_sample:
            xy_pred = (reg_pred[..., :2].sigmoid() * 2.0 - 1.0 + self.grid_xy) * self.stride[0]
        else:
            xy_pred = (reg_pred[..., :2].sigmoid() + self.grid_xy) * self.stride[0]
        # twth -> wh
        wh_pred = reg_pred[..., 2:].exp() * self.anchor_wh
        xywh_pred = torch.cat([xy_pred, wh_pred], dim=-1).view(B, -1, 4)
        # xywh -> x1y1x2y2
        x1y1_pred = xywh_pred[..., :2] - xywh_pred[..., 2:] / 2
        x2y2_pred = xywh_pred[..., :2] + xywh_pred[..., 2:] / 2
        box_pred = torch.cat([x1y1_pred, x2y2_pred], dim=-1)

        return box_pred


    @torch.no_grad()
    def inference_single_image(self, x):
        KA = self.num_anchors
        C = self.num_classes
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

        # [KA*1, H, W] -> [H, W, KA*1] -> [HW*KA, 1]
        obj_pred = obj_pred.permute(1, 2, 0).contiguous().view(-1, 1)
        # [KA*C, H, W] -> [H, W, KA*C] -> [HW*KA, C]
        cls_pred = cls_pred.permute(1, 2, 0).contiguous().view(-1, C)
        # [KA*4, H, W] -> [H, W, KA*4] -> [HW, KA, 4]
        reg_pred = reg_pred.permute(1, 2, 0).contiguous().view(-1, KA, 4)
        # [HW, KA, 4] -> [HW*KA, 4]
        box_pred = self.decode_bbox(reg_pred[None])[0]
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
            KA = self.num_anchors
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

            # [B, KA*1, H, W] -> [B, H, W, KA*1] -> [B, H*W*KA, 1]
            obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 1)
            # [B, KA*C, H, W] -> [B, H, W, KA*C] -> [B, H*W*KA, C]
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, C)
            # [B, KA*4, H, W] -> [B, H, W, KA*4] -> [B, HW, KA, 4]
            reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, KA, 4)
            # [B, HW, KA, 4] -> [B, HW*KA, 4]
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
