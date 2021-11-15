python train.py \
        --cuda \
        -d coco \
        --batch_size 16 \
        -v yolo_nano \
        -ms \
        --ema \
        --max_epoch 200 \
        --lr_epoch 100 150 \
        --multi_anchor \
        --center_sample
        