python train.py \
        --cuda \
        -d coco \
        --batch_size 16 \
        --img_size 640 \
        --multi_scale_range 10 20 \
        -v yolov4 \
        -ms \
        --ema \
        --max_epoch 250 \
        --lr_epoch 130 180 \
        --mosaic \
        --multi_anchor \
        --center_sample
        