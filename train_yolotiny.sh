python train.py \
        --cuda \
        -v yolo_tiny \
        -d coco \
        --batch_size 16 \
        --img_size 640 \
        --multi_scale_range 10 20 \
        -ms \
        --ema \
        --max_epoch 250 \
        --lr_epoch 130 180 \
        --mosaic \
        --multi_anchor \
        --center_sample
        