python train.py \
        --cuda \
        -d coco \
        --batch_size 64 \
        -v yolo_nano \
        -ms \
        --ema \
        --max_epoch 250 \
        --lr_epoch 130 180 \
        --mosaic \
        --multi_anchor \
        --center_sample
        