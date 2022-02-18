import torch
import torch.nn as nn
import numpy as np

from ..backbone import build_backbone
from ..neck.fpn import build_fpn
from ..head.coupled_head import CoupledHead
from utils import box_ops


class YOLONanoPlus(nn.Module):
    def __init__(self, 
                 cfg=None,
                 device=None, 
                 img_size=640, 
                 num_classes=80, 
                 trainable=False, 
                 conf_thresh=0.001, 
                 nms_thresh=0.60, 
                 center_sample=False):
        super(YOLONanoPlus, self).__init__()
        self.cfg = cfg
        self.device = device
        self.img_size = img_size
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.center_sample = center_sample

        # backbone
        self.backbone, feature_channels, strides = build_backbone(model_name=cfg["backbone"], 
                                                                  pretrained=trainable)
        self.stride = strides
        anchor_size = cfg["anchor_size"]
        self.anchor_size = torch.tensor(anchor_size).reshape(len(self.stride), len(anchor_size) // 3, 2).float()
        self.num_anchors = self.anchor_size.size(1)
        c3, c4, c5 = feature_channels

        # build grid cell
        self.grid_cell, self.anchors_wh = self.create_grid(img_size)

        # neck
        self.neck = build_fpn(model_name=cfg['neck'], 
                              in_dim=[c3, c4, c5],
                              depth=0.33,
                              depthwise=cfg['depthwise'],
                              act='lrelu')

        # head
        self.head = CoupledHead(in_dim=[c3, c4, c5],
                                stride=self.stride,
                                kernel_size=3,
                                padding=1,
                                width=1.0,
                                num_classes=self.num_classes,
                                num_anchors=self.num_anchors,
                                depthwise=cfg['depthwise'],
                                act='lrelu',
                                init_bias=trainable,
                                center_sample=self.center_sample)


    def create_grid(self, img_size):
        total_grid_xy = []
        total_anchor_wh = []
        w, h = img_size, img_size
        for ind, s in enumerate(self.stride):
            # generate grid cells
            fmp_w, fmp_h = w // s, h // s
            grid_y, grid_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
            # [H, W, 2] -> [HW, 2]
            grid_xy = torch.stack([grid_x, grid_y], dim=-1).float().view(-1, 2)
            # [HW, 2] -> [1, HW, 1, 2]   
            grid_xy = grid_xy[None, :, None, :].to(self.device)
            # [1, HW, 1, 2]
            anchor_wh = self.anchor_size[ind].repeat(fmp_h*fmp_w, 1, 1).unsqueeze(0).to(self.device)

            total_grid_xy.append(grid_xy)
            total_anchor_wh.append(anchor_wh)

        return total_grid_xy, total_anchor_wh


    def set_grid(self, img_size):
        self.img_size = img_size
        self.grid_cell, self.anchors_wh = self.create_grid(img_size)


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
        c3, c4, c5 = self.backbone(x)
   
        # neck
        p3, p4, p5 = self.neck([c3, c4, c5])
            
        # head
        obj_pred, cls_pred, box_pred = self.head([p3, p4, p5], self.grid_cell, self.anchors_wh)
        
        # normalize bbox
        bboxes = torch.clamp(box_pred / self.img_size, 0., 1.)

        # scores
        scores = torch.sigmoid(obj_pred) * torch.softmax(cls_pred, dim=-1)

        # to cpu
        scores = scores[0].to('cpu').numpy() # [N, C]
        bboxes = bboxes[0].to('cpu').numpy() # [N, 4]

        # post-process
        bboxes, scores, cls_inds = self.postprocess(bboxes, scores)

        return bboxes, scores, cls_inds


    def forward(self, x, targets=None):
        if not self.trainable:
            return self.inference_single_image(x)
        else:
            B = x.size(0)
            # backbone
            c3, c4, c5 = self.backbone(x)

            # neck
            p3, p4, p5 = self.neck([c3, c4, c5])

            # head
            obj_pred, cls_pred, box_pred = self.head([p3, p4, p5], self.grid_cell, self.anchors_wh)
                        
            # normalize bbox
            box_pred = box_pred / self.img_size

            # compute giou between prediction bbox and target bbox
            x1y1x2y2_pred = box_pred.view(-1, 4)
            x1y1x2y2_gt = targets[..., 2:6].view(-1, 4)

            # iou: [B, HW,]
            iou_pred = box_ops.giou_score(x1y1x2y2_pred, x1y1x2y2_gt, batch_size=B)
            obj_tgt = 0.5 * (iou_pred[..., None].clone().detach() + 1.0) # [-1, 1] -> [0, 1]

            # we set iou as the target of the objectness
            targets = torch.cat([obj_tgt, targets], dim=-1)

            return obj_pred, cls_pred, iou_pred, targets
