import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from backbone.shufflenetv2 import shufflenetv2
from utils.modules import SPP, Conv
from utils import box_ops
from utils import loss


class YOLONano(nn.Module):
    def __init__(self, 
                 device, 
                 img_size=640, 
                 num_classes=80, 
                 trainable=False, 
                 conf_thresh=0.001, 
                 nms_thresh=0.60, 
                 anchor_size=None):

        super(YOLONano, self).__init__()
        self.device = device
        self.img_size = img_size
        self.num_classes = num_classes
        self.stride = [8, 16, 32]
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.anchor_size = torch.tensor(anchor_size).reshape(len(self.stride), len(anchor_size) // 3, 2).float()
        self.num_anchors = self.anchor_size.size(1)
        self.grid_cell, self.anchors_wh = self.create_grid(img_size)

        # backbone
        print('Use backbone: shufflenetv2_1.0x')
        self.backbone = shufflenetv2(pretrained=trainable)
        c3, c4, c5 = 116, 232, 464

        # neck
        self.neck = SPP(c1=c5, c2=c5)

        # FPN+PAN
        self.conv1x1_0 = Conv(c3, 96, k=1)
        self.conv1x1_1 = Conv(c4, 96, k=1)
        self.conv1x1_2 = Conv(c5, 96, k=1)

        self.smooth_0 = Conv(96, 96, k=3, p=1)
        self.smooth_1 = Conv(96, 96, k=3, p=1)
        self.smooth_2 = Conv(96, 96, k=3, p=1)
        self.smooth_3 = Conv(96, 96, k=3, p=1)

        # det head
        self.head_conv_1 = nn.Sequential(
            Conv(96, 96, k=3, p=1, g=96),
            Conv(96, 96, k=1),
            Conv(96, 96, k=3, p=1, g=96),
            Conv(96, 96, k=1)
        )
        self.head_conv_2 = nn.Sequential(
            Conv(96, 96, k=3, p=1, g=96),
            Conv(96, 96, k=1),
            Conv(96, 96, k=3, p=1, g=96),
            Conv(96, 96, k=1)
        )
        self.head_conv_3 = nn.Sequential(
            Conv(96, 96, k=3, p=1, g=96),
            Conv(96, 96, k=1),
            Conv(96, 96, k=3, p=1, g=96),
            Conv(96, 96, k=1)
        )

        # det conv
        self.head_det_1 = nn.Conv2d(96, self.num_anchors * (1 + self.num_classes + 4), 1)
        self.head_det_2 = nn.Conv2d(96, self.num_anchors * (1 + self.num_classes + 4), 1)
        self.head_det_3 = nn.Conv2d(96, self.num_anchors * (1 + self.num_classes + 4), 1)

        if self.trainable:
            # init bias
            self.init_bias()


    def init_bias(self):               
        # init bias
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        nn.init.constant_(self.head_det_1.bias[..., :self.num_anchors], bias_value)
        nn.init.constant_(self.head_det_2.bias[..., :self.num_anchors], bias_value)
        nn.init.constant_(self.head_det_3.bias[..., :self.num_anchors], bias_value)


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
        """"Pure Python NMS Baseline."""
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
        KA = self.num_anchors
        C = self.num_classes
        # backbone
        c3, c4, c5 = self.backbone(x)

        # neck
        c5 = self.neck(c5)

        p3 = self.conv1x1_0(c3)
        p4 = self.conv1x1_1(c4)
        p5 = self.conv1x1_2(c5)

        # top-down
        p4 = self.smooth_0(p4 + F.interpolate(p5, scale_factor=2.0))
        p3 = self.smooth_1(p3 + F.interpolate(p4, scale_factor=2.0))

        # bottom-up
        p4 = self.smooth_2(p4 + F.interpolate(p3, scale_factor=0.5))
        p5 = self.smooth_3(p5 + F.interpolate(p4, scale_factor=0.5))

        # det head
        pred_s = self.head_det_1(self.head_conv_1(p3))
        pred_m = self.head_det_2(self.head_conv_2(p4))
        pred_l = self.head_det_3(self.head_conv_3(p5))

        preds = [pred_s, pred_m, pred_l]
        obj_pred_list = []
        cls_pred_list = []
        box_pred_list = []

        for i, pred in enumerate(preds):
            # [B, KA*(1 + C + 4), H, W] -> [B, KA, H, W] -> [B, H, W, KA] ->  [B, HW*KA, 1]
            obj_pred_i = pred[:, :KA, :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, 1)
            # [B, KA*(1 + C + 4), H, W] -> [B, KA*C, H, W] -> [B, H, W, KA*C] -> [B, H*W*KA, C]
            cls_pred_i = pred[:, KA:KA*(1+C), :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, C)
            # [B, KA*(1 + C + 4), H, W] -> [B, KA*4, H, W] -> [B, H, W, KA*4] -> [B, HW, KA, 4]
            reg_pred_i = pred[:, KA*(1+C):, :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, KA, 4)
            # txtytwth -> xywh
            xy_pred_i = (reg_pred_i[..., :2].sigmoid() + self.grid_cell[i]) * self.stride[i]
            wh_pred_i = reg_pred_i[..., 2:].exp() * self.anchors_wh[i]
            xywh_pred_i = torch.cat([xy_pred_i, wh_pred_i], dim=-1).view(B, -1, 4)
            # xywh -> x1y1x2y2
            x1y1_pred_i = xywh_pred_i[..., :2] - xywh_pred_i[..., 2:] / 2
            x2y2_pred_i = xywh_pred_i[..., :2] + xywh_pred_i[..., 2:] / 2
            box_pred_i = torch.cat([x1y1_pred_i, x2y2_pred_i], dim=-1)

            obj_pred_list.append(obj_pred_i)
            cls_pred_list.append(cls_pred_i)
            box_pred_list.append(box_pred_i)
        
        obj_pred = torch.cat(obj_pred_list, dim=1)
        cls_pred = torch.cat(cls_pred_list, dim=1)
        box_pred = torch.cat(box_pred_list, dim=1)
        
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
