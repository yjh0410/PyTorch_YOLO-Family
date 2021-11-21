python train.py \
        --cuda \
        -d coco \
        --root /mnt/share/ssd2/dataset \
        --batch_size 16 \
        --img_size 640 \
        --multi_scale_range 10 20 \
        -v yolox \
        -ms \
        --ema \
        --max_epoch 250 \
        --lr_epoch 130 180 \
        --mosaic \
        --center_sample
        