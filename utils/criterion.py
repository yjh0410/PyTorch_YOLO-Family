import torch.nn as nn
import torch.nn.functional as F


class MSEWithLogitsLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, targets, target_pos):
        inputs = logits.sigmoid()
        # mse loss
        loss = F.mse_loss(input=inputs, 
                          target=targets,
                          reduction="none")
        pos_loss = loss * target_pos * 5.0
        neg_loss = loss * (1.0 - target_pos) * 1.0
        loss = pos_loss + neg_loss

        if self.reduction == 'mean':
            loss = loss.mean()

        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


class BCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=1.0, neg_weight=0.25, reduction='mean'):
        super().__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.reduction = reduction

    def forward(self, logits, targets, target_pos):
        # bce loss
        loss = F.binary_cross_entropy_with_logits(input=logits, target=targets, reduction="none")
        pos_loss = loss * target_pos * self.pos_weight
        neg_loss = loss * (1.0 - target_pos) * self.neg_weight
        loss = pos_loss + neg_loss

        if self.reduction == 'mean':
            loss = loss.mean()

        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


class Criterion(nn.Module):
    def __init__(self,
                 args,
                 cfg,
                 loss_obj_weight=1.0, 
                 loss_cls_weight=1.0, 
                 loss_reg_weight=1.0, 
                 num_classes=80):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self.loss_obj_weight = loss_obj_weight
        self.loss_cls_weight = loss_cls_weight
        self.loss_reg_weight = loss_reg_weight

        # objectness loss
        try:
            if cfg['loss_obj'] == 'mse':
                self.obj_loss_f = MSEWithLogitsLoss(reduction='none')
            elif cfg['loss_obj'] == 'bce':
                self.obj_loss_f = BCEWithLogitsLoss(reduction='none')
        except:
            self.obj_loss_f = MSEWithLogitsLoss(reduction='none')
        # class loss
        self.cls_loss_f = nn.CrossEntropyLoss(reduction='none')


    def loss_objectness(self, pred_obj, target_obj, target_pos):
        """
            pred_obj: (FloatTensor) [B, HW, 1]
            target_obj: (FloatTensor) [B, HW,]
            target_pos: (FloatTensor) [B, HW,]
        """
        # obj loss: [B, HW,]
        loss_obj = self.obj_loss_f(pred_obj[..., 0], target_obj, target_pos)

        if self.args.scale_loss == 'batch':
            # scale loss by batch size
            batch_size = pred_obj.size(0)
            loss_obj = loss_obj.sum() / batch_size
        elif self.args.scale_loss == 'positive':
            # scale loss by number of positive samples
            num_pos = target_pos.sum().clamp(1.0)
            loss_obj = loss_obj.sum() / num_pos

        return loss_obj


    def loss_class(self, pred_cls, target_cls, target_pos):
        """
            pred_cls: (FloatTensor) [B, HW, C]
            target_cls: (LongTensor) [B, HW,]
            target_pos: (FloatTensor) [B, HW,]
        """
        # [B, HW, C] -> [B, C, HW]
        pred_cls = pred_cls.permute(0, 2, 1)
        # reg loss: [B, HW, ]
        loss_cls = self.cls_loss_f(pred_cls, target_cls)
        # valid loss. Here we only compute the loss of positive samples
        loss_cls = loss_cls * target_pos

        if self.args.scale_loss == 'batch':
            # scale loss by batch size
            batch_size = pred_cls.size(0)
            loss_cls = loss_cls.sum() / batch_size
        elif self.args.scale_loss == 'positive':
            # scale loss by number of positive samples
            num_pos = target_pos.sum().clamp(1.0)
            loss_cls = loss_cls.sum() / num_pos

        return loss_cls


    def loss_bbox(self, pred_iou, target_pos, target_scale):
        """
            pred_iou: (FloatTensor) [B, HW, ]
            target_pos: (FloatTensor) [B, HW,]
            target_scale: (FloatTensor) [B, HW,]
        """

        # bbox loss: [B, HW,]
        loss_reg = 1. - pred_iou
        loss_reg = loss_reg * target_scale
        # valid loss. Here we only compute the loss of positive samples
        loss_reg = loss_reg * target_pos

        if self.args.scale_loss == 'batch':
            # scale loss by batch size
            batch_size = pred_iou.size(0)
            loss_reg = loss_reg.sum() / batch_size
        elif self.args.scale_loss == 'positive':
            # scale loss by number of positive samples
            num_pos = target_pos.sum().clamp(1.0)
            loss_reg = loss_reg.sum() / num_pos

        return loss_reg


    def forward(self, pred_obj, pred_cls, pred_iou, targets):
        """
            pred_obj: (Tensor) [B, HW, 1]
            pred_cls: (Tensor) [B, HW, C]
            pred_iou: (Tensor) [B, HW,]
            targets: (Tensor) [B, HW, 1+1+1+4]
        """
        # groundtruth
        target_obj = targets[..., 0].float()     # [B, HW,]
        target_pos = targets[..., 1].float()     # [B, HW,]
        target_cls = targets[..., 2].long()      # [B, HW,]
        target_scale = targets[..., -1].float()  # [B, HW,]

        # objectness loss
        loss_obj = self.loss_objectness(pred_obj, target_obj, target_pos)

        # class loss
        loss_cls = self.loss_class(pred_cls, target_cls, target_pos)

        # regression loss
        loss_reg = self.loss_bbox(pred_iou, target_pos, target_scale)

        # total loss
        losses = self.loss_obj_weight * loss_obj + \
                 self.loss_cls_weight * loss_cls + \
                 self.loss_reg_weight * loss_reg

        return loss_obj, loss_cls, loss_reg, losses


def build_criterion(args, cfg, num_classes=80):
    criterion = Criterion(args=args,
                          cfg=cfg,
                          loss_obj_weight=args.loss_obj_weight,
                          loss_cls_weight=args.loss_cls_weight,
                          loss_reg_weight=args.loss_reg_weight,
                          num_classes=num_classes)
    return criterion


if __name__ == "__main__":
    pass
