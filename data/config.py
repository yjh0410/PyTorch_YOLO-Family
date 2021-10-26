# config.py

yolo_cfg = {
    'train_size': 640,
    'val_size': 640,
    # for multi-scale trick
    'random_size_range': [10, 20],
    # anchor size
    'anchor_size': [[10, 13],   [16, 30],   [33, 23],
                    [30, 61],   [62, 45],   [59, 119],
                    [116, 90],  [156, 198], [373, 326]],
}
