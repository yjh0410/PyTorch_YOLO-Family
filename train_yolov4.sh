python train.py \
        --cuda \
        -d coco \
        --batch_size 16 \
        -v yolov4 \
        -ms \
        --ema \
        --max_epoch 250 \
        --lr_epoch 130 180 \
        --mosaic \
        --multi_anchor \
        --center_sample
        