python train.py \
        --cuda \
        -d coco \
        --root /mnt/share/ssd2/dataset \
        --batch_size 16 \
        --img_size 224 \
        -v yolo_tr \
        -ms \
        --accumulate 1 \
        --ema
        