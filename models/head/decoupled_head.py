import torch
import torch.nn as nn

from ..basic.conv import Conv


class DecoupledHead(nn.Module):
    def __init__(self, 
                 in_dim=[256, 512, 1024], 
                 stride=[8, 16, 32],
                 head_dim=256, 
                 width=1.0, 
                 num_classes=80, 
                 num_anchors=3,
                 depthwise=False,
                 grid_cell=None,
                 anchors_wh=None,
                 act='silu', 
                 init_bias=True,
                 center_sample=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.head_dim = int(head_dim * width)
        self.width = width
        self.stride = stride
        self.grid_cell = grid_cell
        self.anchors_wh = anchors_wh
        self.center_sample = center_sample

        self.input_proj = nn.ModuleList()
        self.cls_feat = nn.ModuleList()
        self.reg_feat = nn.ModuleList()
        self.obj_pred = nn.ModuleList()
        self.cls_pred = nn.ModuleList()
        self.reg_pred = nn.ModuleList()

        for c in in_dim:
            self.input_proj.append(
                Conv(c, self.head_dim, k=1, act=act)
            )
            self.cls_feat.append(
                nn.Sequential(
                    Conv(self.head_dim, self.head_dim, k=3, p=1, act=act, depthwise=depthwise),
                    Conv(self.head_dim, self.head_dim, k=3, p=1, act=act, depthwise=depthwise)
                )
            )
            self.reg_feat.append(
                nn.Sequential(
                    Conv(self.head_dim, self.head_dim, k=3, p=1, act=act, depthwise=depthwise),
                    Conv(self.head_dim, self.head_dim, k=3, p=1, act=act, depthwise=depthwise)
                )
            )
            self.obj_pred.append(
                nn.Conv2d(self.head_dim, num_anchors * 1, kernel_size=1)
            )
            self.cls_pred.append(
                nn.Conv2d(self.head_dim, num_anchors * num_classes, kernel_size=1)
            )
            self.reg_pred.append(
                nn.Conv2d(self.head_dim, num_anchors * 4, kernel_size=1)
            )

        if init_bias:
            # init bias
            self.init_bias()


    def init_bias(self):               
        # init bias
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        for obj_pred in self.obj_pred:
            nn.init.constant_(obj_pred.bias, bias_value)


    def forward(self, features):
        """
            features: (List of Tensor) of multiple feature maps
        """
        B = features[0].size(0)
        obj_preds = []
        cls_preds = []
        box_preds = []
        for i in range(len(features)):
            feat = features[i]
            feat = self.input_proj[i](feat)
            cls_feat = self.cls_feat[i](feat)
            reg_feat = self.reg_feat[i](feat)
            obj_pred = self.obj_pred[i](reg_feat)
            cls_pred = self.cls_pred[i](cls_feat)
            reg_pred = self.reg_pred[i](reg_feat)

            # [B, KA, H, W] -> [B, H, W, KA] ->  [B, HW*KA, 1]
            obj_preds.append(obj_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 1))
            # [[B, KA*C, H, W] -> [B, H, W, KA*C] -> [B, H*W*KA, C]
            cls_preds.append(cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes))
            # [B, KA*4, H, W] -> [B, H, W, KA*4] -> [B, HW, KA, 4]
            reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_anchors, 4)

            if self.anchors_wh is not None:
                # txty -> xy
                if self.center_sample:     
                    xy_pred = (self.grid_cell[i] + reg_pred[..., :2].sigmoid() * 2.0 - 1.0) * self.stride[i]
                else:
                    xy_pred = (self.grid_cell[i] + reg_pred[..., :2].sigmoid()) * self.stride[i]
                # twth -> wh
                wh_pred = reg_pred[..., 2:].exp() * self.anchors_wh[i]
                # xywh -> x1y1x2y2
                x1y1_pred = xy_pred - wh_pred * 0.5
                x2y2_pred = xy_pred + wh_pred * 0.5
                box_preds.append(torch.cat([x1y1_pred, x2y2_pred], dim=-1).view(B, -1, 4))

        obj_preds = torch.cat(obj_preds, dim=1)  # [B, N, 1]
        cls_preds = torch.cat(cls_preds, dim=1)  # [B, N, C]
        box_preds = torch.cat(box_preds, dim=1)  # [B, N, 4]

        return obj_preds, cls_preds, box_preds
