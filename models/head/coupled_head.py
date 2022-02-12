import torch
import torch.nn as nn

from ..basic.conv import Conv


class CoupledHead(nn.Module):
    def __init__(self, 
                 in_dim=[256, 512, 1024], 
                 stride=[8, 16, 32],
                 kernel_size=3,
                 padding=1,
                 width=1.0, 
                 num_classes=80, 
                 num_anchors=3,
                 depthwise=False,
                 act='silu', 
                 init_bias=True,
                 center_sample=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.width = width
        self.stride = stride
        self.center_sample = center_sample

        self.head_feat = nn.ModuleList()
        self.head_pred = nn.ModuleList()

        for c in in_dim:
            head_dim = int(c * width)
            self.head_feat.append(
                nn.Sequential(
                    Conv(head_dim, head_dim, k=kernel_size, p=padding, act=act, depthwise=depthwise),
                    Conv(head_dim, head_dim, k=kernel_size, p=padding, act=act, depthwise=depthwise),
                )
            )
            self.head_pred.append(
                nn.Conv2d(head_dim, num_anchors * (1 + num_classes + 4), kernel_size=1)
            )

        if init_bias:
            # init bias
            self.init_bias()


    def init_bias(self):
        # init bias
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        for head_pred in self.head_pred:
            nn.init.constant_(head_pred.bias[..., :self.num_anchors], bias_value)


    def forward(self, features, grid_cell=None, anchors_wh=None):
        """
            features: (List of Tensor) of multiple feature maps
        """
        B = features[0].size(0)
        obj_preds = []
        cls_preds = []
        box_preds = []
        for i in range(len(features)):
            feat = features[i]
            head_feat = self.head_feat[i](feat)
            head_pred = self.head_pred[i](head_feat)
            # obj_pred / cls_pred / reg_pred
            obj_pred = head_pred[:, :self.num_anchors, :, :]
            cls_pred = head_pred[:, self.num_anchors:self.num_anchors*(1+self.num_classes), :, :]
            reg_pred = head_pred[:, self.num_anchors*(1+self.num_classes):, :, :]

            # [B, KA, H, W] -> [B, H, W, KA] ->  [B, HW*KA, 1]
            obj_preds.append(obj_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 1))
            # [[B, KA*C, H, W] -> [B, H, W, KA*C] -> [B, H*W*KA, C]
            cls_preds.append(cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes))
            # [B, KA*4, H, W] -> [B, H, W, KA*4] -> [B, HW, KA, 4]
            reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_anchors, 4)

            # decode box
            ## txty -> xy
            if self.center_sample:     
                xy_pred = (grid_cell[i] + reg_pred[..., :2].sigmoid() * 2.0 - 1.0) * self.stride[i]
            else:
                xy_pred = (grid_cell[i] + reg_pred[..., :2].sigmoid()) * self.stride[i]
            ## twth -> wh
            if anchors_wh is not None:
                wh_pred = reg_pred[..., 2:].exp() * anchors_wh[i]
            else:
                wh_pred = reg_pred[..., 2:].exp() * self.stride[i]
            ## xywh -> x1y1x2y2
            x1y1_pred = xy_pred - wh_pred * 0.5
            x2y2_pred = xy_pred + wh_pred * 0.5
            box_preds.append(torch.cat([x1y1_pred, x2y2_pred], dim=-1).view(B, -1, 4))

        obj_preds = torch.cat(obj_preds, dim=1)  # [B, N, 1]
        cls_preds = torch.cat(cls_preds, dim=1)  # [B, N, C]
        box_preds = torch.cat(box_preds, dim=1)  # [B, N, 4]

        return obj_preds, cls_preds, box_preds
