python train.py \
        --cuda \
        -d coco \
        --root /mnt/share/ssd2/dataset \
        -m yolov4 \
        --batch_size 16 \
        --img_size 640 \
        --max_epoch 250 \
        --lr_epoch 130 180 \
        --multi_scale_range 10 20 \
        --multi_scale \
        --mosaic \
        --center_sample \
        --ema
        