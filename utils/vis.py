import numpy as np
import cv2


def vis_data(images, targets):
    """
        images: (tensor) [B, 3, H, W]
        targets: (list) a list of targets
    """
    batch_size = images.size(0)
    # vis data
    rgb_mean=np.array((0.406, 0.456, 0.485), dtype=np.float32)
    rgb_std=np.array((0.225, 0.224, 0.229), dtype=np.float32)

    for bi in range(batch_size):
        # to numpy
        image = images[bi].permute(1, 2, 0).cpu().numpy()
        # to BGR
        image = image[..., (2, 1, 0)]
        # denormalize
        image = ((image * rgb_std + rgb_mean)*255).astype(np.uint8)
        image = image.copy()
        img_h, img_w = image.shape[:2]

        targets_i = targets[bi]
        for target in targets_i:
            x1, y1, x2, y2 = target[:-1]
            x1 = int(x1 * img_w)
            y1 = int(y1 * img_h)
            x2 = int(x2 * img_w)
            y2 = int(y2 * img_h)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        cv2.imshow('groundtruth', image)
        cv2.waitKey(0)


def vis_targets(images, targets, anchor_sizes=None, strides=[8, 16, 32]):
    """
        images: (tensor) [B, 3, H, W]
        targets: (tensor) [B, HW*KA, 1+1+4+1]
        anchor_sizes: (List) 
        strides: (List[Int]) output stride of network
    """
    batch_size = images.size(0)
    KA = len(anchor_sizes) // len(strides) if anchor_sizes is not None else 1
    # vis data
    rgb_mean=np.array((0.485, 0.456, 0.406), dtype=np.float32)
    rgb_std=np.array((0.229, 0.224, 0.225), dtype=np.float32)

    for bi in range(batch_size):
        # to numpy
        image = images[bi].permute(1, 2, 0).cpu().numpy()
        # denormalize
        image = ((image * rgb_std + rgb_mean)*255).astype(np.uint8)
        # to BGR
        image = image[..., (2, 1, 0)]
        image = image.copy()
        img_h, img_w = image.shape[:2]

        target_i = targets[bi] # [HW*KA, 1+1+4+1]
        N = 0
        for si, s in enumerate(strides):
            fmp_h, fmp_w = img_h // s, img_w // s
            HWKA = fmp_h * fmp_w * KA
            targets_i_s = target_i[N:N+HWKA]
            N += HWKA
            # [HW*KA, 1+1+4+1] -> [H, W, KA, 1+1+4+1]
            targets_i_s = targets_i_s.reshape(fmp_h, fmp_w, KA, -1)
            for j in range(fmp_h):
                for i in range(fmp_w):
                    for k in range(KA):
                        target = targets_i_s[j, i, k] # [1+1+4+1,]
                        if target[0] > 0.:
                            # gt box
                            box = target[2:6]
                            x1, y1, x2, y2 = box
                            # denormalize bbox
                            x1 = int(x1 * img_w)
                            y1 = int(y1 * img_h)
                            x2 = int(x2 * img_w)
                            y2 = int(y2 * img_h)
                            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

                            if anchor_sizes is not None:
                                # anchor box
                                anchor_size = anchor_sizes[si*KA + k]
                                x_anchor = (i) * s
                                y_anchor = (j) * s
                                w_anchor, h_anchor = anchor_size
                                anchor_box = [x_anchor, y_anchor, w_anchor, h_anchor]
                                print('stride: {} - anchor box: ({}, {}, {}, {})'.format(s, *anchor_box))
                                x1_a = int(x_anchor - w_anchor * 0.5)
                                y1_a = int(y_anchor - h_anchor * 0.5)
                                x2_a = int(x_anchor + w_anchor * 0.5)
                                y2_a = int(y_anchor + h_anchor * 0.5)
                                cv2.rectangle(image, (x1_a, y1_a), (x2_a, y2_a), (255, 0, 0), 2)
                            else:
                                x_anchor = (i) * s
                                y_anchor = (j) * s
                                anchor_point = (x_anchor, y_anchor)
                                print('stride: {} - anchor point: ({}, {})'.format(s, *anchor_point))
                                cv2.circle(image, anchor_point, 10, (255, 0, 0), -1)

        cv2.imshow('assignment', image)
        cv2.waitKey(0)
