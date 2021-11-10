python train.py \
        --cuda \
        -v yolov4 \
        -ms \
        --ema \
        --batch_size 16 \
        --max_epoch 250 \
        --lr_epoch 130 180 \
        -d coco \
        --mosaic \
        --multi_anchor \
        --center_sample
        