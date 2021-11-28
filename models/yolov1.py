import numpy as np
import torch
import torch.nn as nn
from utils.modules import Conv, DilatedEncoder
from backbone.resnet import resnet50
from utils import box_ops
from utils import loss


class YOLOv1(nn.Module):
    def __init__(self, 
                 device, 
                 img_size=None, 
                 num_classes=20, 
                 trainable=False, 
                 conf_thresh=0.001, 
                 nms_thresh=0.6,
                 anchor_size=None):
        super(YOLOv1, self).__init__()
        self.device = device
        self.img_size = img_size
        self.num_classes = num_classes
        self.stride = [32]
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.grid_xy = self.create_grid(img_size)

        # backbone
        self.backbone = resnet50(pretrained=trainable)
        c5 = 2048
        p5 = 512
        # neck
        self.neck = DilatedEncoder(c1=c5, c2=p5)

        # head
        self.cls_feat = nn.Sequential(
            Conv(p5, p5, k=3, p=1, s=1),
            Conv(p5, p5, k=3, p=1, s=1)
        )
        self.reg_feat = nn.Sequential(
            Conv(p5, p5, k=3, p=1, s=1),
            Conv(p5, p5, k=3, p=1, s=1),
            Conv(p5, p5, k=3, p=1, s=1),
            Conv(p5, p5, k=3, p=1, s=1)
        )

        # head
        self.obj_pred = nn.Conv2d(p5, 1, kernel_size=1)
        self.cls_pred = nn.Conv2d(p5, self.num_classes, kernel_size=1)
        self.reg_pred = nn.Conv2d(p5, 4, kernel_size=1)

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
        xy_pred = reg_pred[..., :2].sigmoid() + self.grid_xy
        wh_pred = reg_pred[..., 2:].exp()
        xywh_pred = torch.cat([xy_pred, wh_pred], dim=-1)
        # xywh -> x1y1x2y2
        x1y1_pred = xywh_pred[..., :2] - xywh_pred[..., 2:] / 2
        x2y2_pred = xywh_pred[..., :2] + xywh_pred[..., 2:] / 2
        box_pred = torch.cat([x1y1_pred, x2y2_pred], dim=-1) * self.stride[0]

        return box_pred


    def nms(self, dets, scores):
        """"Pure Python NMS YOLOv4."""
        x1 = dets[:, 0]  #xmin
        y1 = dets[:, 1]  #ymin
        x2 = dets[:, 2]  #xmax
        y2 = dets[:, 3]  #ymax

        areas = (x2 - x1) * (y2 - y1)                 # the size of bbox
        order = scores.argsort()[::-1]                        # sort bounding boxes by decreasing order

        keep = []                                             # store the final bounding boxes
        while order.size > 0:
            i = order[0]                                      #the index of the bbox with highest confidence
            keep.append(i)                                    #save it to keep
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


    def forward(self, x, targets=None):
        B = x.size(0)
        C = self.num_classes
        # backbone
        x = self.backbone(x)

        # neck
        x = self.neck(x)

        # head
        cls_feat = self.cls_feat(x)
        reg_feat = self.reg_feat(x)

        # pred
        obj_pred = self.obj_pred(reg_feat)
        cls_pred = self.cls_pred(cls_feat)
        reg_pred = self.reg_pred(reg_feat)

        # [B, 1, H, W] -> [B, H, W, 1] -> [B, H*W, 1]
        obj_pred =obj_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 1)
        # [B, C, H, W] -> [B, H, W, C] -> [B, H*W, C]
        cls_pred =cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, C)
        # [B, 4, H, W] -> [B, H, W, 4] -> [B, HW, 4]
        reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 4)
        box_pred = self.decode_bbox(reg_pred)

        # train
        if self.trainable:
            # decode bbox: [B, HW*KA, 4]
            x1y1x2y2_pred = (box_pred / self.img_size).view(-1, 4)
            x1y1x2y2_gt = targets[..., -4:].view(-1, 4)

            # giou: [B, HW*KA,]
            giou_pred = box_ops.giou_score(x1y1x2y2_pred, x1y1x2y2_gt, batch_size=B)

            # we set giou as the target of the objectness prediction
            targets = torch.cat([0.5 * (giou_pred.view(B, -1, 1).clone().detach() + 1.0), targets], dim=-1)

            # loss
            obj_loss, cls_loss, reg_loss, total_loss = loss.loss(pred_obj=obj_pred,
                                                                  pred_cls=cls_pred,
                                                                  pred_giou=giou_pred,
                                                                  targets=targets)

            return obj_loss, cls_loss, reg_loss, total_loss

        # test
        else:
            with torch.no_grad():
                # batch size = 1
                # [B, H*W*KA, C] -> [H*W*KA, C]
                scores = torch.sigmoid(obj_pred)[0] * torch.softmax(cls_pred, dim=-1)[0]
                # [B, H*W*KA, 4] -> [H*W*KA, 4]
                bboxes = torch.clamp((box_pred / self.img_size)[0], 0., 1.)

                # to cpu
                scores = scores.to('cpu').numpy()
                bboxes = bboxes.to('cpu').numpy()

                # post-process
                bboxes, scores, cls_inds = self.postprocess(bboxes, scores)

                return bboxes, scores, cls_inds
