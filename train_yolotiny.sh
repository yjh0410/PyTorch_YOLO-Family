python train.py \
        --cuda \
        -m yolo_tiny \
        -d coco \
        --root /mnt/share/ssd2/dataset \
        --batch_size 16 \
        --img_size 512 \
        --max_epoch 200 \
        --lr_epoch 100 150 \
        --multi_scale \
        --multi_scale_range 10 16 \
        --mosaic \
        --center_sample \
        --ema
        