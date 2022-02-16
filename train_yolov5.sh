python train.py \
        --cuda \
        -d coco \
        -m yolov5_l \
        --root /mnt/share/ssd2/dataset \
        --batch_size 16 \
        --lr 0.001 \
        --img_size 608 \
        --max_epoch 200 \
        --lr_epoch 100 150 \
        --multi_scale \
        --multi_scale_range 10 19 \
        --scale_loss batch \
        --accumulate 1 \
        --mosaic \
        --multi_anchor \
        --ema
                