python train.py \
        --cuda \
        -d coco \
        -m yolov5_l \
        --root /home/jxk/object-detection/dataset \
        --batch_size 16 \
        --lr 0.001 \
        --img_size 608 \
        --max_epoch 160 \
        --lr_epoch 100 130 \
        --multi_scale \
        --multi_scale_range 10 19 \
        --scale_loss batch \
        --accumulate 4 \
        --mosaic \
        --multi_anchor \
        --ema
                