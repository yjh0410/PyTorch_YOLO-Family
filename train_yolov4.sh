python train.py \
        --cuda \
        -v yolov4 \
        -ms \
        --ema \
        --mosaic \
        --batch_size 16 \
        --max_epoch 250 \
        --lr_epoch 130 180 \
        --center_sample \
        -d coco
        