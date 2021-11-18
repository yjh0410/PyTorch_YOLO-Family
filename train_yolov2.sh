python train.py \
        --cuda \
        -d coco \
        --root /mnt/share/ssd2/dataset \
        --batch_size 16 \
        --img_size 640 \
        --multi_scale_range 10 20 \
        -v yolov2 \
        -ms \
        --ema
        