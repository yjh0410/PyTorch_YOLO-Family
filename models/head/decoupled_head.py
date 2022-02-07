import torch
import torch.nn as nn

from ..basic.conv import Conv


class DecoupledHead(nn.Module):
    def __init__(self, 
                 in_dim=[256, 512, 1024], 
                 head_dim=256, 
                 width=1.0, 
                 num_classes=80, 
                 num_anchors=3,
                 depthwise=False,
                 act='silu', 
                 init_bias=True):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.head_dim = int(head_dim * width)
        self.width = width

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
        reg_preds = []
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
            reg_preds.append(reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_anchors, 4))

        return obj_preds, cls_preds, reg_preds
