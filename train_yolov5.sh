python train.py \
        --cuda \
        -d coco \
        -m yolov5_l \
        --root /mnt/share/ssd2/dataset \
        --batch_size 15 \
        --lr 0.001 \
        --img_size 640 \
        --max_epoch 150 \
        --lr_epoch 90 120 \
        --multi_scale \
        --multi_scale_range 10 20 \
        --scale_loss batch \
        --accumulate 1 \
        --mosaic \
        --multi_anchor \
        --ema
                