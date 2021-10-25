import numpy as np
import torch


def compute_iou(anchor_boxes, gt_box):
    """
    Input:
        anchor_boxes : ndarray -> [[xc_s, yc_s, anchor_w, anchor_h], ..., [xc_s, yc_s, anchor_w, anchor_h]].
        gt_box : ndarray -> [xc_s, yc_s, anchor_w, anchor_h].
    Output:
        iou : ndarray -> [iou_1, iou_2, ..., iou_m], and m is equal to the number of anchor boxes.
    """
    # compute the iou between anchor box and gt box
    # First, change [xc_s, yc_s, anchor_w, anchor_h] ->  [x1, y1, x2, y2]
    # anchor box :
    ab_x1y1_x2y2 = np.zeros([len(anchor_boxes), 4])
    ab_x1y1_x2y2[:, 0] = anchor_boxes[:, 0] - anchor_boxes[:, 2] / 2  # x1
    ab_x1y1_x2y2[:, 1] = anchor_boxes[:, 1] - anchor_boxes[:, 3] / 2  # y1
    ab_x1y1_x2y2[:, 2] = anchor_boxes[:, 0] + anchor_boxes[:, 2] / 2  # x2
    ab_x1y1_x2y2[:, 3] = anchor_boxes[:, 1] + anchor_boxes[:, 3] / 2  # y2
    w_ab, h_ab = anchor_boxes[:, 2], anchor_boxes[:, 3]
    
    # gt_box : 
    # We need to expand gt_box(ndarray) to the shape of anchor_boxes(ndarray), in order to compute IoU easily. 
    gt_box_expand = np.repeat(gt_box, len(anchor_boxes), axis=0)

    gb_x1y1_x2y2 = np.zeros([len(anchor_boxes), 4])
    gb_x1y1_x2y2[:, 0] = gt_box_expand[:, 0] - gt_box_expand[:, 2] / 2 # x1
    gb_x1y1_x2y2[:, 1] = gt_box_expand[:, 1] - gt_box_expand[:, 3] / 2 # y1
    gb_x1y1_x2y2[:, 2] = gt_box_expand[:, 0] + gt_box_expand[:, 2] / 2 # x2
    gb_x1y1_x2y2[:, 3] = gt_box_expand[:, 1] + gt_box_expand[:, 3] / 2 # y1
    w_gt, h_gt = gt_box_expand[:, 2], gt_box_expand[:, 3]

    # Then we compute IoU between anchor_box and gt_box
    S_gt = w_gt * h_gt
    S_ab = w_ab * h_ab
    I_w = np.minimum(gb_x1y1_x2y2[:, 2], ab_x1y1_x2y2[:, 2]) - np.maximum(gb_x1y1_x2y2[:, 0], ab_x1y1_x2y2[:, 0])
    I_h = np.minimum(gb_x1y1_x2y2[:, 3], ab_x1y1_x2y2[:, 3]) - np.maximum(gb_x1y1_x2y2[:, 1], ab_x1y1_x2y2[:, 1])
    S_I = I_h * I_w
    U = S_gt + S_ab - S_I + 1e-20
    IoU = S_I / U
    
    return IoU


def set_anchors(anchor_size):
    """
    Input:
        anchor_size : list -> [[h_1, w_1], [h_2, w_2], ..., [h_n, w_n]].
    Output:
        anchor_boxes : ndarray -> [[0, 0, anchor_w, anchor_h],
                                   [0, 0, anchor_w, anchor_h],
                                   ...
                                   [0, 0, anchor_w, anchor_h]].
    """
    num_anchors = len(anchor_size)
    anchor_boxes = np.zeros([num_anchors, 4])
    for index, size in enumerate(anchor_size): 
        anchor_w, anchor_h = size
        anchor_boxes[index] = np.array([0, 0, anchor_w, anchor_h])
    
    return anchor_boxes


def gt_creator(img_size, strides, label_lists, anchor_size, center_sample=False):
    """creator gt"""
    # prepare
    batch_size = len(label_lists)
    img_h = img_w = img_size
    num_scale = len(strides)
    gt_tensor = []
    KA = len(anchor_size) // num_scale

    for s in strides:
        fmp_h, fmp_w = img_h // s, img_w // s
        # [B, H, W, KA, obj+cls+box+pos]
        gt_tensor.append(np.zeros([batch_size, fmp_h, fmp_w, KA, 1+1+4+1]))
        
    # generate gt datas  
    for bi in range(batch_size):
        label = label_lists[bi]
        for box_cls in label:
            # get a bbox coords
            cls_id = int(box_cls[-1])
            x1, y1, x2, y2 = box_cls[:-1]
            # compute the center, width and height
            xc = (x2 + x1) / 2 * img_w
            yc = (y2 + y1) / 2 * img_h
            bw = (x2 - x1) * img_w
            bh = (y2 - y1) * img_h

            if bw < 1. or bh < 1.:
                # print('A dirty data !!!')
                continue    

            # compute the IoU
            anchor_boxes = set_anchors(anchor_size)
            gt_box = np.array([[0, 0, bw, bh]])
            iou = compute_iou(anchor_boxes, gt_box)

            # We assign the anchor box with highest IoU score.
            index = np.argmax(iou)
            # s_indx, ab_ind = index // num_scale, index % num_scale
            s_indx = index // KA
            ab_ind = index - s_indx * KA
            # get the corresponding stride
            s = strides[s_indx]
            # compute the gride cell
            xc_s = xc / s
            yc_s = yc / s
            grid_x = int(xc_s)
            grid_y = int(yc_s)

            if center_sample:
                for j in range(grid_y-1, grid_y+2):
                    for i in range(grid_x-1, grid_x+2):
                        if (j >= 0 and j < gt_tensor[s_indx].shape[0]) and (i >= 0 and i < gt_tensor[s_indx].shape[1]):
                            gt_tensor[s_indx][bi, j, i, ab_ind, 0] = 1.0
                            gt_tensor[s_indx][bi, j, i, ab_ind, 1] = cls_id
                            gt_tensor[s_indx][bi, j, i, ab_ind, 2:-1] = np.array([x1, y1, x2, y2])
            else:
                if (j >= 0 and j < gt_tensor[s_indx].shape[0]) and (i >= 0 and i < gt_tensor[s_indx].shape[1]):
                    gt_tensor[s_indx][bi, grid_y, grid_x, ab_ind, 0] = 1.0
                    gt_tensor[s_indx][bi, grid_y, grid_x, ab_ind, 1] = cls_id
                    gt_tensor[s_indx][bi, grid_y, grid_x, ab_ind, 2:] = np.array([x1, y1, x2, y2])


    gt_tensor = [gt.reshape(batch_size, -1, 1+1+4) for gt in gt_tensor]
    gt_tensor = np.concatenate(gt_tensor, axis=0)
    
    return torch.from_numpy(gt_tensor).float()


if __name__ == "__main__":
    gt_box = np.array([[0.0, 0.0, 10, 10]])
    anchor_boxes = np.array([[0.0, 0.0, 10, 10], 
                             [0.0, 0.0, 4, 4], 
                             [0.0, 0.0, 8, 8], 
                             [0.0, 0.0, 16, 16]
                             ])
    iou = compute_iou(anchor_boxes, gt_box)
    print(iou)
