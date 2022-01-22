python train.py \
        --cuda \
        -d coco \
        -m yolo_nano \
        --root /mnt/share/ssd2/dataset \
        --batch_size 64 \
        --lr 0.001 \
        --img_size 512 \
        --max_epoch 200 \
        --lr_epoch 100 150
        --multi_scale \
        --multi_scale_range 10 16 \
        --mosaic \
        --multi_anchor \
        --ema
                        