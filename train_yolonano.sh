python train.py \
        --cuda \
        -v yolo_nano \
        -d coco \
        --batch_size 16 \
        --img_size 512 \
        --multi_scale_range 9 16 \
        -ms \
        --ema \
        --max_epoch 250 \
        --lr_epoch 130 180 \
        --mosaic \
        --multi_anchor \
        --center_sample
        