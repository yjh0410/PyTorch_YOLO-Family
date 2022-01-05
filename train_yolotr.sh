python train.py \
        --cuda \
        -d coco \
        -m yolo_tr \
        --root /mnt/share/ssd2/dataset \
        --batch_size 16 \
        --lr 0.001 \
        --img_size 224 \
        --max_epoch 150 \
        --lr_epoch 90 120 \
        --scale_loss pos \
        --multi_anchor \
        --center_sample \
        --ema
                