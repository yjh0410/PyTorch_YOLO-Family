python train.py \
        --cuda \
        -v yolo_nano \
        -d coco \
        --root /mnt/share/ssd2/dataset \
        --batch_size 64 \
        --img_size 512 \
        --multi_scale_range 10 16 \
        -ms \
        --ema \
        --max_epoch 160 \
        --lr_epoch 100 130
        