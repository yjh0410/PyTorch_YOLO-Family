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


class Criterion(nn.Module):
    def __init__(self, 
                 loss_obj_weight=1.0, 
                 loss_cls_weight=1.0, 
                 loss_reg_weight=1.0, 
                 num_classes=80, 
                 scale_loss='batch'):
        super().__init__()
        self.num_classes = num_classes
        self.loss_obj_weight = loss_obj_weight
        self.loss_cls_weight = loss_cls_weight
        self.loss_reg_weight = loss_reg_weight

        self.obj_loss_f = MSEWithLogitsLoss(reduction='none')
        self.cls_loss_f = nn.CrossEntropyLoss(reduction='none')
        self.reg_loss_f = None

        self.scale_loss = scale_loss
        if scale_loss == 'batch':
            print('Scale loss by batch size.')
        elif scale_loss == 'pos':
            print('Scale loss by number of positive samples.')


    def loss_objectness(self, pred_obj, target_obj, target_pos):
        """
            pred_obj: (FloatTensor) [B, HW, 1]
            target_obj: (FloatTensor) [B, HW,]
            target_pos: (FloatTensor) [B, HW,]
        """
        # obj loss: [B, HW,]
        loss_obj = self.obj_loss_f(pred_obj[..., 0], target_obj, target_pos)

        # scale loss by number of total positive samples
        if self.scale_loss == 'pos':
            loss_obj = loss_obj.sum() / (target_pos.sum().clamp(1.0))
        
        # scale loss by batch size
        elif self.scale_loss == 'batch':
            batch_size = pred_obj.size(0)
            loss_obj = loss_obj.sum() / batch_size

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

        # scale loss by number of total positive samples
        if self.scale_loss == 'pos':
            loss_cls = loss_cls.sum() / (target_pos.sum().clamp(1.0))

        # scale loss by batch size
        elif self.scale_loss == 'batch':
            batch_size = pred_cls.size(0)
            loss_cls = loss_cls.sum() / batch_size

        return loss_cls


    def loss_giou(self, pred_giou, target_pos):
        """
            pred_giou: (FloatTensor) [B, HW, ]
            target_pos: (FloatTensor) [B, HW,]
        """

        # reg loss: [B, HW,]
        loss_reg = 1. - pred_giou
        # valid loss. Here we only compute the loss of positive samples
        loss_reg = loss_reg * target_pos

        # scale loss by number of total positive samples
        if self.scale_loss == 'pos':
            loss_reg = loss_reg.sum() / (target_pos.sum().clamp(1.0))

        # scale loss by batch size
        elif self.scale_loss == 'batch':
            batch_size = pred_giou.size(0)
            loss_reg = loss_reg.sum() / batch_size

        return loss_reg


    def forward(self, pred_obj, pred_cls, pred_giou, targets):
        """
            pred_obj: (Tensor) [B, HW, 1]
            pred_cls: (Tensor) [B, HW, C]
            pred_giou: (Tensor) [B, HW,]
            targets: (Tensor) [B, HW, 1+1+4]
        """
        # groundtruth
        target_obj = targets[..., 0].float()  # [B, HW,]
        target_pos = targets[..., 1].float()  # [B, HW,]
        target_cls = targets[..., 2].long()   # [B, HW,]

        # objectness loss
        loss_obj = self.loss_objectness(pred_obj, target_obj, target_pos)

        # class loss
        loss_cls = self.loss_class(pred_cls, target_cls, target_pos)

        # regression loss
        loss_reg = self.loss_giou(pred_giou, target_pos)

        # total loss
        losses = self.loss_obj_weight * loss_obj + \
                 self.loss_cls_weight * loss_cls + \
                 self.loss_reg_weight * loss_reg

        return loss_obj, loss_cls, loss_reg, losses


def build_criterion(args, num_classes=80):
    criterion = Criterion(loss_obj_weight=args.loss_obj_weight,
                          loss_cls_weight=args.loss_cls_weight,
                          loss_reg_weight=args.loss_reg_weight,
                          num_classes=num_classes,
                          scale_loss=args.scale_loss)
    return criterion


if __name__ == "__main__":
    pass
