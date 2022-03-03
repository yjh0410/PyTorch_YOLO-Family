# YOLO config


yolo_config = {
    'yolov1': {
        # backbone
        'backbone': 'r50',
        # neck
        'neck': 'dilated_encoder',
        # anchor size
        'anchor_size': None
    },
    'yolov2': {
        # backbone
        'backbone': 'r50',
        # neck
        'neck': 'dilated_encoder',
        # anchor size: P5-640
        'anchor_size': [[10, 13],   [16, 30],   [33, 23],
                        [30, 61],   [62, 45],   [59, 119],
                        [116, 90],  [156, 198], [373, 326]]
    },
    'yolov3': {
        # backbone
        'backbone': 'd53',
        # neck
        'neck': 'conv_blocks',
        # anchor size: P5-640
        'anchor_size': [[10, 13],   [16, 30],   [33, 23],
                        [30, 61],   [62, 45],   [59, 119],
                        [116, 90],  [156, 198], [373, 326]]
    },
    'yolov3_spp': {
        # backbone
        'backbone': 'd53',
        # neck
        'neck': 'spp',
        # anchor size: P5-640
        'anchor_size': [[10, 13],   [16, 30],   [33, 23],
                        [30, 61],   [62, 45],   [59, 119],
                        [116, 90],  [156, 198], [373, 326]]
    },
    'yolov3_de': {
        # backbone
        'backbone': 'd53',
        # neck
        'neck': 'dilated_encoder',
        # anchor size: P5-640
        'anchor_size': [[10, 13],   [16, 30],   [33, 23],
                        [30, 61],   [62, 45],   [59, 119],
                        [116, 90],  [156, 198], [373, 326]]
    },
    'yolov4': {
        # backbone
        'backbone': 'cspd53',
        # neck
        'neck': 'spp',
        # anchor size: P5-640
        'anchor_size': [[10, 13],   [16, 30],   [33, 23],
                        [30, 61],   [62, 45],   [59, 119],
                        [116, 90],  [156, 198], [373, 326]],
        # loss
        'loss_obj': 'mse',  # optional: mse, qfl
        'loss_box': 'giou'  # optional: iou, giou, ciou
    },
    'yolov5_s': {
        # backbone
        'backbone': 'csp_s',
        'width': 0.5,
        'depth': 0.33,
        'depthwise': False,
        'freeze': False,
        # neck
        'neck': 'yolopafpn',
        # head
        'head_dim': 256,
        # anchor size: P5-640
        'anchor_size': [[10, 13],   [16, 30],   [33, 23],
                        [30, 61],   [62, 45],   [59, 119],
                        [116, 90],  [156, 198], [373, 326]],
        # loss
        'loss_obj': 'mse',  # optional: mse, bce
        'loss_box': 'giou'  # optional: iou, giou, ciou
    },
    'yolov5_m': {
        # backbone
        'backbone': 'csp_m',
        'width': 0.75,
        'depth': 0.67,
        'depthwise': False,
        'freeze': False,
        # neck
        'neck': 'yolopafpn',
        # head
        'head_dim': 256,
        # anchor size: P5-640
        'anchor_size': [[10, 13],   [16, 30],   [33, 23],
                        [30, 61],   [62, 45],   [59, 119],
                        [116, 90],  [156, 198], [373, 326]],
        # loss
        'loss_obj': 'mse',  # optional: mse, bce
        'loss_box': 'giou'  # optional: iou, giou, ciou
    },
    'yolov5_l': {
        # backbone
        'backbone': 'csp_l',
        'width': 1.0,
        'depth': 1.0,
        'depthwise': False,
        'freeze': False,
        # neck
        'neck': 'yolopafpn',
        # head
        'head_dim': 256,
        # anchor size: P5-640
        'anchor_size': [[10, 13],   [16, 30],   [33, 23],
                        [30, 61],   [62, 45],   [59, 119],
                        [116, 90],  [156, 198], [373, 326]],
        # loss
        'loss_obj': 'mse',  # optional: mse, bce
        'loss_box': 'giou'  # optional: iou, giou, ciou
    },
    'yolov5_x': {
        # backbone
        'backbone': 'csp_x',
        'width': 1.25,
        'depth': 1.33,
        'depthwise': False,
        'freeze': False,
        # neck
        'neck': 'yolopafpn',
        # head
        'head_dim': 256,
        # anchor size: P5-640
        'anchor_size': [[10, 13],   [16, 30],   [33, 23],
                        [30, 61],   [62, 45],   [59, 119],
                        [116, 90],  [156, 198], [373, 326]],
        # loss
        'loss_obj': 'mse',  # optional: mse, bce
        'loss_box': 'giou'  # optional: iou, giou, ciou
    },
    'yolov5_t': {
        # backbone
        'backbone': 'csp_t',
        'width': 0.375,
        'depth': 0.33,
        'depthwise': False,
        'freeze': False,
        # neck
        'neck': 'yolopafpn',
        # head
        'head_dim': 256,
        # anchor size: P5-640
        'anchor_size': [[10, 13],   [16, 30],   [33, 23],
                        [30, 61],   [62, 45],   [59, 119],
                        [116, 90],  [156, 198], [373, 326]],
        # loss
        'loss_obj': 'mse',  # optional: mse, bce
        'loss_box': 'giou'  # optional: iou, giou, ciou
    },
    'yolov5_n': {
        # backbone
        'backbone': 'csp_n',
        'width': 0.25,
        'depth': 0.33,
        'depthwise': True,
        'freeze': False,
        # neck
        'neck': 'yolopafpn',
        # head
        'head_dim': 256,
        # anchor size: P5-640
        'anchor_size': [[10, 13],   [16, 30],   [33, 23],
                        [30, 61],   [62, 45],   [59, 119],
                        [116, 90],  [156, 198], [373, 326]],
        # loss
        'loss_obj': 'mse',  # optional: mse, bce
        'loss_box': 'giou'  # optional: iou, giou, ciou
    },    
    'yolo_tiny': {
        # backbone
        'backbone': 'cspd_tiny',
        # neck
        'neck': 'spp-csp',
        # anchor size: P5-640
        'anchor_size': [[10, 13],   [16, 30],   [33, 23],
                        [30, 61],   [62, 45],   [59, 119],
                        [116, 90],  [156, 198], [373, 326]]
    },
    'yolo_nano': {
        # backbone
        'backbone': 'sfnet_v2',
        # neck
        'neck': 'spp-dw',
        # anchor size: P5-640
        'anchor_size': [[10, 13],   [16, 30],   [33, 23],
                        [30, 61],   [62, 45],   [59, 119],
                        [116, 90],  [156, 198], [373, 326]],
        # loss
        'loss_obj': 'mse',  # optional: mse, qfl
        'loss_box': 'giou'  # optional: iou, giou, ciou
    },
    'yolo_nano_plus': {
        # backbone
        'backbone': 'csp_n',
        'depthwise': True,
        # neck
        'neck': 'yolopafpn',
        # anchor size: P5-640
        'anchor_size': [[10, 13],   [16, 30],   [33, 23],
                        [30, 61],   [62, 45],   [59, 119],
                        [116, 90],  [156, 198], [373, 326]],
        # loss
        'loss_obj': 'mse',  # optional: mse, qfl
        'loss_box': 'giou'  # optional: iou, giou, ciou
    },
    'yolotr': {
        # backbone
        'backbone': 'vit_b',
        # neck
        'neck': 'dilated_encoder',
        # anchor size: P5-640
        'anchor_size': [[10, 13],   [16, 30],   [33, 23],
                        [30, 61],   [62, 45],   [59, 119],
                        [116, 90],  [156, 198], [373, 326]]
    }
}