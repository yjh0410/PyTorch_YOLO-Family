python train.py \
        --cuda \
        -d coco \
        --root /mnt/share/ssd2/dataset \
        --batch_size 16 \
        --img_size 640 \
        -v yolov3 \
        --max_epoch 200 \
        --lr_epoch 100 150 \
        --multi_scale \
        --multi_scale_range 10 20 \
        --ema
        