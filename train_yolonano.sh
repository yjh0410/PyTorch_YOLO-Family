python train.py \
        --cuda \
        -d coco \
        -m yolo_nano \
        --root /home/jxk/object-detection/dataset \
        --batch_size 64 \
        --lr 0.001 \
        --img_size 512 \
        --max_epoch 160 \
        --lr_epoch 100 130 \
        --multi_scale \
        --multi_scale_range 10 16 \
        --multi_anchor \
        --ema
                        