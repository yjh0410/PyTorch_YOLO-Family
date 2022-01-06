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
        'backbone': 'cspd53',
        # neck
        'neck': 'dilated_encoder',
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
    },
    # The following YOLOv4 is coming soon ...
    'yolov4_s': {
        # model cfg params
        'depth': 0.33,
        'width': 0.5,
        'depthwise': False,
        'act': 'silu',
        # backbone
        'backbone': 'yolox_csp_s',
        # neck
        'neck': 'spp',
        # anchor size: P5-640
        'anchor_size': [[10, 13],   [16, 30],   [33, 23],
                        [30, 61],   [62, 45],   [59, 119],
                        [116, 90],  [156, 198], [373, 326]]
    },
    'yolov4_m': {
        # model cfg params
        'depth': 0.67,
        'width': 0.75,
        'depthwise': False,
        'act': 'silu',
        # backbone
        'backbone': 'yolox_csp_m',
        # neck
        'neck': 'spp',
        # anchor size: P5-640
        'anchor_size': [[10, 13],   [16, 30],   [33, 23],
                        [30, 61],   [62, 45],   [59, 119],
                        [116, 90],  [156, 198], [373, 326]]
    },
    'yolov4_l': {
        # model cfg params
        'depth': 1.0,
        'width': 1.0,
        'depthwise': False,
        'act': 'silu',
        # backbone
        'backbone': 'yolox_csp_l',
        # neck
        'neck': 'spp',
        # anchor size: P5-640
        'anchor_size': [[10, 13],   [16, 30],   [33, 23],
                        [30, 61],   [62, 45],   [59, 119],
                        [116, 90],  [156, 198], [373, 326]]
    },
    'yolov4_x': {
        # model cfg params
        'depth': 1.33,
        'width': 1.25,
        'depthwise': False,
        'act': 'silu',
        # backbone
        'backbone': 'yolox_csp_x',
        # neck
        'neck': 'spp',
        # anchor size: P5-640
        'anchor_size': [[10, 13],   [16, 30],   [33, 23],
                        [30, 61],   [62, 45],   [59, 119],
                        [116, 90],  [156, 198], [373, 326]]
    },
    'yolov4_tiny': {
        # model cfg params
        'depth': 0.33,
        'width': 0.375,
        'depthwise': False,
        'act': 'silu',
        # backbone
        'backbone': 'yolox_csp_tiny',
        # neck
        'neck': 'spp',
        # anchor size: P5-640
        'anchor_size': [[10, 13],   [16, 30],   [33, 23],
                        [30, 61],   [62, 45],   [59, 119],
                        [116, 90],  [156, 198], [373, 326]]
    },
    'yolov4_nano': {
        # model cfg params
        'depth': 0.33,
        'width': 0.25,
        'depthwise': True,
        'act': 'silu',
        # backbone
        'backbone': 'yolox_csp_nano',
        # neck
        'neck': 'spp',
        # anchor size: P5-640
        'anchor_size': [[10, 13],   [16, 30],   [33, 23],
                        [30, 61],   [62, 45],   [59, 119],
                        [116, 90],  [156, 198], [373, 326]]
    },
}