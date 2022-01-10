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
    'yolov4_exp': {
        # backbone
        'backbone': 'd53',
        # neck
        'neck': 'spp',
        # anchor size: P5-640
        'anchor_size': [[10, 13],   [16, 30],   [33, 23],
                        [30, 61],   [62, 45],   [59, 119],
                        [116, 90],  [156, 198], [373, 326]]
    },
    'yolo_tiny': {
        # backbone
        'backbone': 'cspd_tiny',
        # neck
        'neck': 'dilated_encoder',
        # anchor size: P5-640
        'anchor_size': [[10, 13],   [16, 30],   [33, 23],
                        [30, 61],   [62, 45],   [59, 119],
                        [116, 90],  [156, 198], [373, 326]]
    },
    'yolo_nano': {
        # backbone
        'backbone': 'sfnet_v2',
        # neck
        'neck': 'spp',
        # anchor size: P5-640
        'anchor_size': [[10, 13],   [16, 30],   [33, 23],
                        [30, 61],   [62, 45],   [59, 119],
                        [116, 90],  [156, 198], [373, 326]]
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