python train.py \
        --cuda \
        -d coco \
        --root /mnt/share/ssd2/dataset \
        --batch_size 16 \
        --img_size 640 \
        -v yolov2 \
        --multi_scale \
        --multi_scale_range 10 20 \
        --center_sample \
        --ema
        