python train.py \
        --cuda \
        -d coco \
        -m yolo_tiny \
        --root /home/jxk/object-detection/dataset \
        --batch_size 16 \
        --lr 0.001 \
        --img_size 512 \
        --max_epoch 250 \
        --lr_epoch 130 180 \
        --multi_scale \
        --multi_scale_range 10 16 \
        --accumulate 4 \
        --mosaic \
        --multi_anchor \
        --ema
                        