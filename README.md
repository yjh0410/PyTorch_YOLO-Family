# Update: 2022-05-31
Recently, I have released an anchor-free YOLO:

https://github.com/yjh0410/FreeYOLO

# A new and strong YOLO family
Recently, I rebuild my YOLO-Family project !!

# Requirements
- We recommend you to use Anaconda to create a conda environment:
```Shell
conda create -n yolo python=3.6
```

- Then, activate the environment:
```Shell
conda activate yolo
```

- Requirements:
```Shell
pip install -r requirements.txt 
```
PyTorch >= 1.1.0 and Torchvision >= 0.3.0

# Visualize positive samples
You can run following command to visualize positiva sample:
```Shell
python train.py \
        -d voc \
        --root path/to/your/dataset \
        -m yolov2 \
        --batch_size 2 \
        --vis_targets
```

# Come soon
My better YOLO family


# This project
In this project, you can enjoy: 
- a new and stronger YOLOv1
- a new and stronger YOLOv2
- a stronger YOLOv3
- a stronger YOLOv3 with SPP
- a stronger YOLOv3 with DilatedEncoder
- YOLOv4 (I'm trying to make it better)
- YOLO-Tiny
- YOLO-Nano


# Future work
- Try to make my YOLO-v4 better.
- Train my YOLOv1/YOLOv2 with ViT-Base (pretrained by MaskAutoencoder)

# Weights
You can download all weights including my DarkNet-53, CSPDarkNet-53, MAE-ViT and YOLO weights from the following links.

## Backbone
My Backbone:
- DarkNet53: https://github.com/yjh0410/PyTorch_YOLO-Family/releases/download/yolo-weight/darknet53.pth
- CSPDarkNet-53: https://github.com/yjh0410/PyTorch_YOLO-Family/releases/download/yolo-weight/cspdarknet53.pth
- CSPDarkNet-Tiny: https://github.com/yjh0410/PyTorch_YOLO-Family/releases/download/yolo-weight/cspdarknet_tiny.pth

YOLOX-Backbone:
- CSPDarkNet-S: https://github.com/yjh0410/YOLOX-Backbone/releases/download/YOLOX-Backbone/yolox_cspdarknet_s.pth
- CSPDarkNet-M: https://github.com/yjh0410/YOLOX-Backbone/releases/download/YOLOX-Backbone/yolox_cspdarknet_m.pth
- CSPDarkNet-L: https://github.com/yjh0410/YOLOX-Backbone/releases/download/YOLOX-Backbone/yolox_cspdarknet_l.pth
- CSPDarkNet-X: https://github.com/yjh0410/YOLOX-Backbone/releases/download/YOLOX-Backbone/yolox_cspdarknet_x.pth
- CSPDarkNet-Tiny: https://github.com/yjh0410/YOLOX-Backbone/releases/download/YOLOX-Backbone/yolox_cspdarknet_tiny.pth
- CSPDarkNet-Nano: https://github.com/yjh0410/YOLOX-Backbone/releases/download/YOLOX-Backbone/yolox_cspdarknet_nano.pth

## YOLO
- YOLOv1: https://github.com/yjh0410/PyTorch_YOLO-Family/releases/download/yolo-weight/yolov1_35.22_54.7.pth
- YOLOv2: https://github.com/yjh0410/PyTorch_YOLO-Family/releases/download/yolo-weight/yolov2_36.4_56.6.pth
- YOLOv3: https://github.com/yjh0410/PyTorch_YOLO-Family/releases/download/yolo-weight/yolov3_36.9_59.0.pth
- YOLOv3-SPP: https://github.com/yjh0410/PyTorch_YOLO-Family/releases/download/yolo-weight/yolov3_spp_38.2_60.1.pth
- YOLOv3-DE: https://github.com/yjh0410/PyTorch_YOLO-Family/releases/download/yolo-weight/yolov3_de_38.7_60.2.pth
- YOLOv4: https://github.com/yjh0410/PyTorch_YOLO-Family/releases/download/yolo-weight/yolov4_exp_43.0_63.4.pth
- YOLO-Tiny: https://github.com/yjh0410/PyTorch_YOLO-Family/releases/download/yolo-weight/yolo_tiny_28.8_48.6.pth
- YOLO-Nano: https://github.com/yjh0410/PyTorch_YOLO-Family/releases/download/yolo-weight/yolo_nano_22.4_40.7.pth


# Experiments
## Tricks
Tricks in this project:
- [x] Augmentations: Flip + Color jitter + RandomCrop
- [x] Model EMA
- [x] Mosaic Augmentation
- [x] Multi Scale training
- [ ] Gradient accumulation
- [ ] MixUp Augmentation
- [ ] Cosine annealing learning schedule
- [ ] AdamW
- [ ] Scale loss by number of positive samples


# Experiments
All experiment results are evaluated on COCO val. All FPS results except YOLO-Nano's are measured on a 2080ti GPU. 
We will measure the speed of YOLO-Nano on a CPU.

## YOLOv1
<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8>           </th><td bgcolor=white> FPS </td><td bgcolor=white> AP   </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white>  APs  </td><td bgcolor=white>  APm  </td><td bgcolor=white>  APl  </td><td bgcolor=white>  GFLOPs  </td><td bgcolor=white>  Params  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv1-320</th><td bgcolor=white> 151 </td><td bgcolor=white> 25.4 </td><td bgcolor=white> 41.5 </td><td bgcolor=white> 26.0 </td><td bgcolor=white> 4.2   </td><td bgcolor=white> 25.0 </td><td bgcolor=white> 49.8 </td><td bgcolor=white> 10.49 </td><td bgcolor=white> 44.54M </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv1-416</th><td bgcolor=white> 128 </td><td bgcolor=white> 30.1 </td><td bgcolor=white> 47.8 </td><td bgcolor=white> 30.9 </td><td bgcolor=white> 7.8   </td><td bgcolor=white> 31.9 </td><td bgcolor=white> 53.3 </td><td bgcolor=white> 17.73 </td><td bgcolor=white> 44.54M </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv1-512</th><td bgcolor=white> 114 </td><td bgcolor=white> 33.1 </td><td bgcolor=white> 52.2 </td><td bgcolor=white> 34.0 </td><td bgcolor=white> 10.8  </td><td bgcolor=white> 35.9 </td><td bgcolor=white> 54.9 </td><td bgcolor=white> 26.85 </td><td bgcolor=white> 44.54M </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv1-640</th><td bgcolor=white> 75 </td><td bgcolor=white> 35.2 </td><td bgcolor=white> 54.7 </td><td bgcolor=white> 37.1 </td><td bgcolor=white>  14.3 </td><td bgcolor=white>  39.5 </td><td bgcolor=white>  53.4 </td><td bgcolor=white> 41.96 </td><td bgcolor=white> 44.54M </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv1-800 </th><td bgcolor=white>     </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white> 65.56 </td><td bgcolor=white> 44.54M </td></tr>

</table></tbody>

## YOLOv2
<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8>           </th><td bgcolor=white> FPS </td><td bgcolor=white> AP   </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white>  APs  </td><td bgcolor=white>  APm  </td><td bgcolor=white>  APl  </td><td bgcolor=white>  GFLOPs  </td><td bgcolor=white>  Params  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv2-320 </th><td bgcolor=white> 147 </td><td bgcolor=white> 26.8 </td><td bgcolor=white> 44.1 </td><td bgcolor=white> 27.1 </td><td bgcolor=white> 4.7  </td><td bgcolor=white> 27.6 </td><td bgcolor=white> 50.8 </td><td bgcolor=white> 10.53 </td><td bgcolor=white> 44.89M </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv2-416 </th><td bgcolor=white> 123 </td><td bgcolor=white> 31.6 </td><td bgcolor=white> 50.3 </td><td bgcolor=white> 32.4 </td><td bgcolor=white> 9.1  </td><td bgcolor=white> 33.8 </td><td bgcolor=white> 54.0 </td><td bgcolor=white> 17.79 </td><td bgcolor=white> 44.89M </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv2-512 </th><td bgcolor=white> 108 </td><td bgcolor=white> 34.3 </td><td bgcolor=white> 54.0 </td><td bgcolor=white> 35.4 </td><td bgcolor=white> 12.3 </td><td bgcolor=white> 37.8 </td><td bgcolor=white> 55.2 </td><td bgcolor=white> 26.94 </td><td bgcolor=white> 44.89M </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv2-640 </th><td bgcolor=white> 73 </td><td bgcolor=white> 36.3 </td><td bgcolor=white> 56.6 </td><td bgcolor=white> 37.7 </td><td bgcolor=white> 15.1 </td><td bgcolor=white>  41.1 </td><td bgcolor=white>  54.0 </td><td bgcolor=white> 42.10 </td><td bgcolor=white> 44.89M </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv2-800 </th><td bgcolor=white>     </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>     </td><td bgcolor=white>     </td><td bgcolor=white> 65.78 </td><td bgcolor=white> 44.89M </td></tr>

</table></tbody>

## YOLOv3

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8>           </th><td bgcolor=white> FPS </td><td bgcolor=white> AP   </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white>  APs  </td><td bgcolor=white>  APm  </td><td bgcolor=white>  APl  </td><td bgcolor=white>  GFLOPs  </td><td bgcolor=white>  Params  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-320</th><td bgcolor=white> 111 </td><td bgcolor=white> 30.8 </td><td bgcolor=white> 50.3 </td><td bgcolor=white> 31.8 </td><td bgcolor=white> 10.0 </td><td bgcolor=white> 33.1 </td><td bgcolor=white> 50.0 </td><td bgcolor=white> 19.57 </td><td bgcolor=white> 61.97M </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-416</th><td bgcolor=white> 89 </td><td bgcolor=white> 34.8 </td><td bgcolor=white> 55.8 </td><td bgcolor=white> 36.1 </td><td bgcolor=white> 14.6 </td><td bgcolor=white> 37.5 </td><td bgcolor=white> 52.9 </td><td bgcolor=white> 33.08 </td><td bgcolor=white> 61.97M </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-512</th><td bgcolor=white> 77 </td><td bgcolor=white> 36.9 </td><td bgcolor=white> 58.1 </td><td bgcolor=white> 39.3 </td><td bgcolor=white> 18.0 </td><td bgcolor=white> 40.3 </td><td bgcolor=white> 52.2 </td><td bgcolor=white> 50.11 </td><td bgcolor=white> 61.97M </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-608</th><td bgcolor=white> 51 </td><td bgcolor=white> 37.0 </td><td bgcolor=white> 58.9 </td><td bgcolor=white> 39.3 </td><td bgcolor=white> 20.5 </td><td bgcolor=white> 41.2 </td><td bgcolor=white> 49.0 </td><td bgcolor=white> 70.66 </td><td bgcolor=white> 61.97M </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-640</th><td bgcolor=white> 49 </td><td bgcolor=white> 36.9 </td><td bgcolor=white> 59.0 </td><td bgcolor=white> 39.7 </td><td bgcolor=white> 21.6 </td><td bgcolor=white> 41.6 </td><td bgcolor=white> 47.7 </td><td bgcolor=white> 78.30 </td><td bgcolor=white> 61.97M </td></tr>
</table></tbody>

## YOLOv3 with SPP

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8>           </th><td bgcolor=white> FPS </td><td bgcolor=white> AP   </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white>  APs  </td><td bgcolor=white>  APm  </td><td bgcolor=white>  APl  </td><td bgcolor=white>  GFLOPs  </td><td bgcolor=white>  Params  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-SPP-320</th><td bgcolor=white> 110 </td><td bgcolor=white> 31.0 </td><td bgcolor=white> 50.8 </td><td bgcolor=white> 32.0 </td><td bgcolor=white> 10.5 </td><td bgcolor=white> 33.0 </td><td bgcolor=white> 50.4 </td><td bgcolor=white> 19.68 </td><td bgcolor=white> 63.02M </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-SPP-416</th><td bgcolor=white> 88 </td><td bgcolor=white> 35.0 </td><td bgcolor=white> 56.1 </td><td bgcolor=white> 36.4 </td><td bgcolor=white> 14.9 </td><td bgcolor=white> 37.7 </td><td bgcolor=white> 52.8 </td><td bgcolor=white> 33.26 </td><td bgcolor=white> 63.02M </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-SPP-512</th><td bgcolor=white> 75 </td><td bgcolor=white> 37.2 </td><td bgcolor=white> 58.7 </td><td bgcolor=white> 39.1 </td><td bgcolor=white> 19.1 </td><td bgcolor=white> 40.0 </td><td bgcolor=white> 53.0 </td><td bgcolor=white> 50.38 </td><td bgcolor=white> 63.02M </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-SPP-608</th><td bgcolor=white> 50 </td><td bgcolor=white> 38.3 </td><td bgcolor=white> 60.1 </td><td bgcolor=white> 40.7 </td><td bgcolor=white> 20.9 </td><td bgcolor=white> 41.1 </td><td bgcolor=white> 51.2 </td><td bgcolor=white> 71.04 </td><td bgcolor=white>  63.02M </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-SPP-640</th><td bgcolor=white> 48 </td><td bgcolor=white> 38.2 </td><td bgcolor=white> 60.1 </td><td bgcolor=white> 40.4 </td><td bgcolor=white> 21.6 </td><td bgcolor=white> 41.1 </td><td bgcolor=white> 50.5 </td><td bgcolor=white> 78.72 </td><td bgcolor=white> 63.02M </td></tr>
</table></tbody>

## YOLOv3 with Dilated Encoder
The DilatedEncoder is proposed by YOLOF.

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8>           </th><td bgcolor=white> FPS </td><td bgcolor=white> AP   </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white>  APs  </td><td bgcolor=white>  APm  </td><td bgcolor=white>  APl  </td><td bgcolor=white>  GFLOPs  </td><td bgcolor=white>  Params  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-DE-320</th><td bgcolor=white> 109 </td><td bgcolor=white> 31.1 </td><td bgcolor=white> 51.1 </td><td bgcolor=white> 31.7 </td><td bgcolor=white> 10.2 </td><td bgcolor=white> 32.6 </td><td bgcolor=white> 51.2 </td><td bgcolor=white> 19.10 </td><td bgcolor=white> 57.25M </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-DE-416</th><td bgcolor=white> 88 </td><td bgcolor=white> 35.0 </td><td bgcolor=white> 56.1 </td><td bgcolor=white> 36.3 </td><td bgcolor=white> 14.6 </td><td bgcolor=white> 37.4 </td><td bgcolor=white> 53.7 </td><td bgcolor=white> 32.28 </td><td bgcolor=white> 57.25M </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-DE-512</th><td bgcolor=white> 74 </td><td bgcolor=white> 37.7 </td><td bgcolor=white> 59.3 </td><td bgcolor=white> 39.6 </td><td bgcolor=white> 17.9 </td><td bgcolor=white> 40.4 </td><td bgcolor=white> 54.4 </td><td bgcolor=white> 48.90 </td><td bgcolor=white> 57.25M </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-DE-608</th><td bgcolor=white> 50 </td><td bgcolor=white> 38.7 </td><td bgcolor=white> 60.5 </td><td bgcolor=white> 40.8 </td><td bgcolor=white> 20.6 </td><td bgcolor=white> 41.7 </td><td bgcolor=white> 53.1 </td><td bgcolor=white> 68.96 </td><td bgcolor=white> 57.25M </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-DE-640</th><td bgcolor=white> 48 </td><td bgcolor=white> 38.7 </td><td bgcolor=white> 60.2 </td><td bgcolor=white> 40.7 </td><td bgcolor=white>  21.3 </td><td bgcolor=white> 41.7 </td><td bgcolor=white> 51.7  </td><td bgcolor=white> 76.41 </td><td bgcolor=white> 57.25M </td></tr>
</table></tbody>

## YOLOv4
I'm still trying to make it better.

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8>           </th><td bgcolor=white> FPS </td><td bgcolor=white> AP   </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white>  APs  </td><td bgcolor=white>  APm  </td><td bgcolor=white>  APl  </td><td bgcolor=white>  GFLOPs  </td><td bgcolor=white>  Params  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv4-320</th><td bgcolor=white> 89 </td><td bgcolor=white> 39.2 </td><td bgcolor=white> 58.6 </td><td bgcolor=white> 40.9 </td><td bgcolor=white> 16.9 </td><td bgcolor=white> 44.1 </td><td bgcolor=white> 59.2 </td><td bgcolor=white> 16.38 </td><td bgcolor=white> 58.14M </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv4-416</th><td bgcolor=white> 84 </td><td bgcolor=white> 41.7 </td><td bgcolor=white> 61.6 </td><td bgcolor=white> 44.2 </td><td bgcolor=white> 22.0 </td><td bgcolor=white> 46.6 </td><td bgcolor=white> 57.7 </td><td bgcolor=white> 27.69 </td><td bgcolor=white> 58.14M </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv4-512</th><td bgcolor=white> 70 </td><td bgcolor=white> 42.9 </td><td bgcolor=white> 63.1 </td><td bgcolor=white> 46.1 </td><td bgcolor=white> 24.5 </td><td bgcolor=white> 48.3 </td><td bgcolor=white> 56.5 </td><td bgcolor=white> 41.94 </td><td bgcolor=white> 58.14M </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv4-608</th><td bgcolor=white> 51 </td><td bgcolor=white> 43.0 </td><td bgcolor=white> 63.4 </td><td bgcolor=white> 46.1 </td><td bgcolor=white> 26.7 </td><td bgcolor=white> 48.6 </td><td bgcolor=white> 53.9 </td><td bgcolor=white> 59.14 </td><td bgcolor=white> 58.14M </td></tr>

</table></tbody>

## YOLO-Tiny
<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8>           </th><td bgcolor=white> FPS </td><td bgcolor=white> AP   </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white>  APs  </td><td bgcolor=white>  APm  </td><td bgcolor=white>  APl  </td><td bgcolor=white>  GFLOPs  </td><td bgcolor=white>  Params  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLO-Tiny-320</th><td bgcolor=white> 143 </td><td bgcolor=white> 26.4 </td><td bgcolor=white> 44.5 </td><td bgcolor=white> 26.8 </td><td bgcolor=white> 8.8 </td><td bgcolor=white> 28.2 </td><td bgcolor=white> 42.4 </td><td bgcolor=white> 2.17 </td><td bgcolor=white> 7.66M </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLO-Tiny-416</th><td bgcolor=white> 130 </td><td bgcolor=white> 28.2 </td><td bgcolor=white> 47.6 </td><td bgcolor=white> 28.8 </td><td bgcolor=white> 11.6 </td><td bgcolor=white> 31.5 </td><td bgcolor=white> 41.4 </td><td bgcolor=white> 3.67 </td><td bgcolor=white> 7.82M </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLO-Tiny-512</th><td bgcolor=white> 118 </td><td bgcolor=white> 28.8 </td><td bgcolor=white> 48.6 </td><td bgcolor=white> 29.4 </td><td bgcolor=white> 13.3 </td><td bgcolor=white> 33.4 </td><td bgcolor=white> 38.3 </td><td bgcolor=white> 5.57 </td><td bgcolor=white> 7.82M </td></tr>

</table></tbody>

## YOLO-Nano
The FPS is measured on i5-1135G& CPU. Any accelerated deployments that would help speed up detection are not done.

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8>           </th><td bgcolor=white> FPS </td><td bgcolor=white> AP   </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white>  APs  </td><td bgcolor=white>  APm  </td><td bgcolor=white>  APl  </td><td bgcolor=white>  GFLOPs  </td><td bgcolor=white>  Params  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLO-Nano-320</th><td bgcolor=white> 25 </td><td bgcolor=white> 18.4 </td><td bgcolor=white> 33.7 </td><td bgcolor=white> 17.8 </td><td bgcolor=white> 3.9 </td><td bgcolor=white> 17.5 </td><td bgcolor=white> 33.1 </td><td bgcolor=white> 0.64 </td><td bgcolor=white> 1.86M </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLO-Nano-416</th><td bgcolor=white> 15 </td><td bgcolor=white> 21.4 </td><td bgcolor=white> 38.5 </td><td bgcolor=white> 20.9 </td><td bgcolor=white> 6.5 </td><td bgcolor=white> 21.4 </td><td bgcolor=white> 34.8 </td><td bgcolor=white> 0.99 </td><td bgcolor=white> 1.86M </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLO-Nano-512</th><td bgcolor=white> 10 </td><td bgcolor=white> 22.4 </td><td bgcolor=white> 40.7 </td><td bgcolor=white> 22.1 </td><td bgcolor=white> 8.0 </td><td bgcolor=white> 24.0 </td><td bgcolor=white> 33.2 </td><td bgcolor=white> 1.65 </td><td bgcolor=white> 1.86M </td></tr>

</table></tbody>


# Dataset

## VOC Dataset
### My BaiduYunDisk
- BaiduYunDisk: https://pan.baidu.com/s/1tYPGCYGyC0wjpC97H-zzMQ Password：4la9

### Download VOC2007 trainval & test

```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2007.sh # <directory>
```

### Download VOC2012 trainval
```Shell
# specify a directory for dataset to be downloaded into, else default is ~/data/
sh data/scripts/VOC2012.sh # <directory>
```
### My BaiduYunDisk
- BaiduYunDisk: https://pan.baidu.com/s/1xAPk8fnaWMMov1VEjr8-zA Password：6vhp

On Ubuntu system, you might use the command `jar xvf xxx.zip` to unzip the `train2017.zip` and `test2017.zip` files
since they are larger than 2G (As far as I know, `unzip` operation can't process the zip file which is larger than 2G.).

## MSCOCO Dataset

### Download MSCOCO 2017 dataset
Just run ```sh data/scripts/COCO2017.sh```. You will get COCO train2017, val2017, test2017.


# Train
For example:

```Shell
python train.py --cuda \
                -d coco \
                -m yolov1 \
                -ms \
                --ema \
                --batch_size 16 \
                --root path/to/dataset/
```

You can run ```python train.py -h``` to check all optional argument. Or you can just run the shell file, for example:
```Shell
sh train_yolov1.sh
```

If you have multi gpus like 8, and you put 4 images on each gpu:
```Shell
python -m torch.distributed.launch --nproc_per_node=8 train.py -d coco \
                                                               --cuda \
                                                               -m yolov1 \
                                                               -ms \
                                                               --ema \
                                                               -dist \
                                                               --sybn \
                                                               --num_gpu 8 \
                                                               --batch_size 4 \
                                                               --root path/to/dataset/
```
Attention, `--batch_size` is the number of batchsize on per GPU, not all GPUs.

I have upload all training log files. For example, `1-v1.txt` contains all the output information during the training YOLOv1.

It is strongly recommended that you open the training shell file to check how I train each YOLO detector.

# Test
For example:

```Shell
python test.py -d coco \
               --cuda \
               -m yolov4 \
               --weight path/to/weight \
               --img_size 640 \
               --root path/to/dataset/ \
               --show
```

# Evaluation
For example

```Shell
python eval.py -d coco-val \
               --cuda \
               -m yolov1 \
               --weight path/to/weight \
               --img_size 640 \
               --root path/to/dataset/
```

# Evaluation on COCO-test-dev
To run on COCO_test-dev(You must be sure that you have downloaded test2017):
```Shell
python eval.py -d coco-test \
               --cuda \
               -m yolov1 \
               --weight path/to/weight \
               --img_size 640 \
                --root path/to/dataset/
```
You will get a `coco_test-dev.json` file. 
Then you should follow the official requirements to compress it into zip format 
and upload it the official evaluation server.
