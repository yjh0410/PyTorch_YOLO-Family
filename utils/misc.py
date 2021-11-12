import torch
import torch.nn as nn
import numpy as np
import math
from copy import deepcopy


def nms(dets, scores, nms_thresh=0.4):
    """"Pure Python NMS baseline."""
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
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(1e-28, xx2 - xx1)
        h = np.maximum(1e-28, yy2 - yy1)
        inter = w * h

        # Cross Area / (bbox + particular area - Cross Area)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        #reserve all the boundingbox whose ovr less than thresh
        inds = np.where(ovr <= nms_thresh)[0]
        order = order[inds + 1]

    return keep


def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets


# Model EMA
class ModelEMA(object):
    def __init__(self, model, decay=0.9999, updates=0):
        # create EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()  # FP32 EMA
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000.))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if is_parallel(model) else model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()


# test time augmentation(TTA)
class TestTimeAugmentation(object):
    def __init__(self, num_classes=80, nms_thresh=0.4, scale_range=[320, 640, 32]):
        self.nms = nms
        self.num_classes = num_classes
        self.nms_thresh = nms_thresh
        self.scales = np.arange(scale_range[0], scale_range[1]+1, scale_range[2])
        
    def __call__(self, x, model):
        # x: Tensor -> [B, C, H, W]
        bboxes_list = []
        scores_list = []
        labels_list = []

        # multi scale
        for s in self.scales:
            if x.size(-1) == s and x.size(-2) == s:
                x_scale = x
            else:
                x_scale =torch.nn.functional.interpolate(
                                        input=x, 
                                        size=(s, s), 
                                        mode='bilinear', 
                                        align_corners=False)
            model.set_grid(s)
            bboxes, scores, labels = model(x_scale)
            bboxes_list.append(bboxes)
            scores_list.append(scores)
            labels_list.append(labels)

            # Flip
            x_flip = torch.flip(x_scale, [-1])
            bboxes, scores, labels = model(x_flip)
            bboxes = bboxes.copy()
            bboxes[:, 0::2] = 1.0 - bboxes[:, 2::-2]
            bboxes_list.append(bboxes)
            scores_list.append(scores)
            labels_list.append(labels)

        bboxes = np.concatenate(bboxes_list)
        scores = np.concatenate(scores_list)
        labels = np.concatenate(labels_list)

        # nms
        keep = np.zeros(len(bboxes), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(labels == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores, self.nms_thresh)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        return bboxes, scores, labels
