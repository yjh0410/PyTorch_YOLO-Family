import numpy
import torch.nn as nn
import torch.nn.functional as F


class MSEWithLogitsLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits, targets, gt_pos):
        inputs = logits.sigoid()
        # mse loss
        loss = F.mse_loss(input=logits, 
                          target=targets,
                          reduction="none")
        pos_loss = loss * gt_pos * 5.0
        neg_loss = loss * (1.0 - gt_pos) * 1.0
        loss = pos_loss + neg_loss

        if self.reduction == 'mean':
            batch_size = logits.size(0)
            loss = loss.sum() / batch_size

        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


def loss(pred_obj, pred_cls, pred_giou, targets):
    # loss func
    conf_loss_function = MSEWithLogitsLoss(reduction='mean')
    cls_loss_function = nn.CrossEntropyLoss(reduction='none')

    # gt
    gt_obj = targets[..., 0].float()
    gt_pos = targets[..., 1].float()
    gt_cls = targets[..., 2].long()

    batch_size = pred_obj.size(0)
    # obj loss
    obj_loss = conf_loss_function(pred_obj[..., 0], gt_obj, gt_pos)
    
    # cls loss
    cls_loss = (cls_loss_function(pred_cls.permute(0, 2, 1), gt_cls) * gt_pos).sum() / batch_size
    
    # reg loss
    reg_loss = ((1. - pred_giou) * gt_pos).sum() / batch_size

    # total loss
    total_loss = 1.0 * obj_loss + 1.0 * cls_loss + 1.0 * reg_loss

    return obj_loss, cls_loss, reg_loss, total_loss


if __name__ == "__main__":
    pass
