python train.py \
        --cuda \
        -d coco \
        --batch_size 16 \
        --img_size 640 \
        --multi_scale_range 10 20 \
        -v yolov1 \
        -ms \
        --ema
        