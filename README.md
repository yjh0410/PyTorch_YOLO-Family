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
        --batch_size 2 \
        -m yolov2 \
        --vis_targets
```

# Come soon
My better YOLO family


# This project
In this project, you can enjoy: 
- a new and stronger YOLOv1
- a new and stronger YOLOv2
- YOLOv3 with DilatedEncoder
- YOLOv4-Exp ~ (I'm try to make it better)
- YOLO-Tiny
- YOLO-Nano


# Future work
- Try to make my YOLO-v4 better.
- Train my YOLOv1/YOLOv2 with ViT-Base (pretrained by MaskAutoencoder)

# Weights
You can download all weights including my DarkNet-53, CSPDarkNet-53, MAE-ViT and YOLO weights from the following links.

## Google Drive
Link: Hold on ...

## BaiDuYun Disk

Link：https://pan.baidu.com/s/1Cin9R52wfubD4xZUHHCRjg 

Password：aigz

# Experiments
## Tricks
Tricks in this project:
- [x] Augmentations: Flip + Color jitter + RandomCrop + Multi-scale
- [x] Model EMA
- [x] GIoU
- [x] Mosaic Augmentation for my YOLOv4-Exp
- [x] Multiple positive samples(`--center_sample`) for my YOLOv4-Exp


On the COCO-val:

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8>       </th><td bgcolor=white> Backbone </td><td bgcolor=white> Size </td><td bgcolor=white> FPS </td><td bgcolor=white> AP   </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white>  APs  </td><td bgcolor=white>  APm  </td><td bgcolor=white>  APl  </td><td bgcolor=white>  GFLOPs  </td><td bgcolor=white>  Params  </td></tr>
<tr><th align="left" bgcolor=#f8f8f8> YOLO-Nano</th><td bgcolor=white> ShuffleNetv2-1.0x </td><td bgcolor=white> 512 </td><td bgcolor=white>     </td><td bgcolor=white> 21.6 </td><td bgcolor=white> 40.0 </td><td bgcolor=white> 20.5 </td><td bgcolor=white> 7.4 </td><td bgcolor=white> 22.7 </td><td bgcolor=white> 32.3 </td><td bgcolor=white> 1.65 </td><td bgcolor=white> 1.86M </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLO-Tiny</th><td bgcolor=white> CSPDarkNet-Tiny </td><td bgcolor=white> 512 </td><td bgcolor=white>     </td><td bgcolor=white> 26.6 </td><td bgcolor=white> 46.1 </td><td bgcolor=white> 26.9 </td><td bgcolor=white> 13.5 </td><td bgcolor=white> 30.0 </td><td bgcolor=white> 35.0 </td><td bgcolor=white> 5.52 </td><td bgcolor=white> 7.66M </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLO-TR</th><td bgcolor=white> ViT-B </td><td bgcolor=white> 384 </td><td bgcolor=white>     </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>   </td><td bgcolor=white>   </td><td bgcolor=white>  </td><td bgcolor=white>  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv1</th><td bgcolor=white> ResNet50 </td><td bgcolor=white> 640 </td><td bgcolor=white>     </td><td bgcolor=white> 35.2 </td><td bgcolor=white> 54.7 </td><td bgcolor=white> 37.1 </td><td bgcolor=white>  14.3 </td><td bgcolor=white>  39.5 </td><td bgcolor=white>  53.4 </td><td bgcolor=white>  41.96   </td><td bgcolor=white> 44.54M </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv2</th><td bgcolor=white> ResNet50 </td><td bgcolor=white> 640 </td><td bgcolor=white>     </td><td bgcolor=white> 36.3 </td><td bgcolor=white> 56.6 </td><td bgcolor=white> 37.7 </td><td bgcolor=white>  15.1 </td><td bgcolor=white>  41.1 </td><td bgcolor=white>  54.0 </td><td bgcolor=white>  42.10   </td><td bgcolor=white> 44.89M </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3 </th><td bgcolor=white> DarkNet53 </td><td bgcolor=white> 640 </td><td bgcolor=white>  </td><td bgcolor=white> 36.9 </td><td bgcolor=white> 59.0 </td><td bgcolor=white> 39.7 </td><td bgcolor=white> 21.6 </td><td bgcolor=white> 41.6 </td><td bgcolor=white> 47.7 </td><td bgcolor=white> 78.30 </td><td bgcolor=white> 61.97M </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-SPP </th><td bgcolor=white> DarkNet53 </td><td bgcolor=white> 640 </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white> 78.72 </td><td bgcolor=white> 63.01M </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-DE </th><td bgcolor=white> DarkNet53 </td><td bgcolor=white> 640 </td><td bgcolor=white>     </td><td bgcolor=white> 38.7 </td><td bgcolor=white> 60.2 </td><td bgcolor=white> 40.7 </td><td bgcolor=white>  21.3 </td><td bgcolor=white> 41.7 </td><td bgcolor=white> 51.7  </td><td bgcolor=white>  76.41   </td><td bgcolor=white> 57.25M </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv4-Exp</th><td bgcolor=white> CSPDarkNet53 </td><td bgcolor=white> 640 </td><td bgcolor=white>     </td><td bgcolor=white> 40.5 </td><td bgcolor=white> 60.4 </td><td bgcolor=white> 43.5 </td><td bgcolor=white> 24.2 </td><td bgcolor=white> 44.8 </td><td bgcolor=white> 52.0 </td><td bgcolor=white>  60.55   </td><td bgcolor=white> 52.00M </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv4-Exp </th><td bgcolor=white> DarkNet53 </td><td bgcolor=white> 640 </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td></tr>

</table></tbody>

The FPS of all YOLO detectors are measured on a one 2080ti GPU with 640 × 640 size.

My CSPDarkNet53 is not good.

# Visualization
I will upload some visualization results:

## YOLO-Nano
<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8>           </th><td bgcolor=white> FPS </td><td bgcolor=white> AP   </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white>  APs  </td><td bgcolor=white>  APm  </td><td bgcolor=white>  APl  </td><td bgcolor=white>  GFLOPs  </td><td bgcolor=white>  Params  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLO-Nano-320</th><td bgcolor=white>     </td><td bgcolor=white> 17.2 </td><td bgcolor=white> 32.9 </td><td bgcolor=white> 15.8 </td><td bgcolor=white> 3.5 </td><td bgcolor=white> 15.7 </td><td bgcolor=white> 31.4 </td><td bgcolor=white> 0.64 </td><td bgcolor=white> 1.86M </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLO-Nano-416</th><td bgcolor=white>     </td><td bgcolor=white> 20.2 </td><td bgcolor=white> 37.7 </td><td bgcolor=white> 19.3 </td><td bgcolor=white> 5.5 </td><td bgcolor=white> 19.7 </td><td bgcolor=white> 33.5 </td><td bgcolor=white> 1.09 </td><td bgcolor=white> 1.86M </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLO-Nano-512</th><td bgcolor=white>     </td><td bgcolor=white> 21.6 </td><td bgcolor=white> 40.0 </td><td bgcolor=white> 20.5 </td><td bgcolor=white> 7.4 </td><td bgcolor=white> 22.7 </td><td bgcolor=white> 32.3 </td><td bgcolor=white> 1.65 </td><td bgcolor=white> 1.86M </td></tr>

</table></tbody>

## YOLO-Tiny
<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8>           </th><td bgcolor=white> FPS </td><td bgcolor=white> AP   </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white>  APs  </td><td bgcolor=white>  APm  </td><td bgcolor=white>  APl  </td><td bgcolor=white>  GFLOPs  </td><td bgcolor=white>  Params  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLO-Tiny-320</th><td bgcolor=white>     </td><td bgcolor=white> 24.5 </td><td bgcolor=white> 42.4 </td><td bgcolor=white> 24.5 </td><td bgcolor=white> 8.9 </td><td bgcolor=white> 26.1 </td><td bgcolor=white> 38.8 </td><td bgcolor=white> 2.16 </td><td bgcolor=white> 7.66M </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLO-Tiny-416</th><td bgcolor=white>     </td><td bgcolor=white> 25.7 </td><td bgcolor=white> 44.4 </td><td bgcolor=white> 25.9 </td><td bgcolor=white> 11.7 </td><td bgcolor=white> 27.8 </td><td bgcolor=white> 36.7 </td><td bgcolor=white> 3.64 </td><td bgcolor=white> 7.66M </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLO-Tiny-512</th><td bgcolor=white>     </td><td bgcolor=white> 26.6 </td><td bgcolor=white> 46.1 </td><td bgcolor=white> 26.9 </td><td bgcolor=white> 13.5 </td><td bgcolor=white> 30.0 </td><td bgcolor=white> 35.0 </td><td bgcolor=white> 5.52 </td><td bgcolor=white> 7.66M </td></tr>

</table></tbody>

## YOLO-TR
<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8>           </th><td bgcolor=white> FPS </td><td bgcolor=white> AP   </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white>  APs  </td><td bgcolor=white>  APm  </td><td bgcolor=white>  APl  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLO-TR-224</th><td bgcolor=white>     </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLO-TR-320</th><td bgcolor=white>     </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLO-TR-384</th><td bgcolor=white>     </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td></tr>

</table></tbody>

## YOLOv1
<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8>           </th><td bgcolor=white> FPS </td><td bgcolor=white> AP   </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white>  APs  </td><td bgcolor=white>  APm  </td><td bgcolor=white>  APl  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv1-320</th><td bgcolor=white>     </td><td bgcolor=white> 25.4 </td><td bgcolor=white> 41.5 </td><td bgcolor=white> 26.0 </td><td bgcolor=white> 4.2   </td><td bgcolor=white> 25.0 </td><td bgcolor=white> 49.8 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv1-416</th><td bgcolor=white>     </td><td bgcolor=white> 30.1 </td><td bgcolor=white> 47.8 </td><td bgcolor=white> 30.9 </td><td bgcolor=white> 7.8   </td><td bgcolor=white> 31.9 </td><td bgcolor=white> 53.3 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv1-512</th><td bgcolor=white>     </td><td bgcolor=white> 33.1 </td><td bgcolor=white> 52.2 </td><td bgcolor=white> 34.0 </td><td bgcolor=white> 10.8  </td><td bgcolor=white> 35.9 </td><td bgcolor=white> 54.9 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv1-640</th><td bgcolor=white>     </td><td bgcolor=white> 35.2 </td><td bgcolor=white> 54.7 </td><td bgcolor=white> 37.1 </td><td bgcolor=white>  14.3 </td><td bgcolor=white>  39.5 </td><td bgcolor=white>  53.4 </td></tr>
</table></tbody>

## YOLOv2
<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8>           </th><td bgcolor=white> FPS </td><td bgcolor=white> AP   </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white>  APs  </td><td bgcolor=white>  APm  </td><td bgcolor=white>  APl  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv2-320</th><td bgcolor=white>     </td><td bgcolor=white> 26.8 </td><td bgcolor=white> 44.1 </td><td bgcolor=white> 27.1 </td><td bgcolor=white> 4.7  </td><td bgcolor=white> 27.6 </td><td bgcolor=white> 50.8 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv2-416</th><td bgcolor=white>     </td><td bgcolor=white> 31.6 </td><td bgcolor=white> 50.3 </td><td bgcolor=white> 32.4 </td><td bgcolor=white> 9.1  </td><td bgcolor=white> 33.8 </td><td bgcolor=white> 54.0 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv2-512</th><td bgcolor=white>     </td><td bgcolor=white> 34.3 </td><td bgcolor=white> 54.0 </td><td bgcolor=white> 35.4 </td><td bgcolor=white> 12.3 </td><td bgcolor=white> 37.8 </td><td bgcolor=white> 55.2 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv2-640</th><td bgcolor=white>     </td><td bgcolor=white> 36.3 </td><td bgcolor=white> 56.6 </td><td bgcolor=white> 37.7 </td><td bgcolor=white> 15.1 </td><td bgcolor=white>  41.1 </td><td bgcolor=white>  54.0 </td></tr>
</table></tbody>

## YOLOv3
Coming soon.

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8>           </th><td bgcolor=white> FPS </td><td bgcolor=white> AP   </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white>  APs  </td><td bgcolor=white>  APm  </td><td bgcolor=white>  APl  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-320</th><td bgcolor=white>     </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-416</th><td bgcolor=white>     </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-512</th><td bgcolor=white>     </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-608</th><td bgcolor=white>     </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>   </td><td bgcolor=white>  </td><td bgcolor=white>   </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-640</th><td bgcolor=white> </td><td bgcolor=white> 36.9 </td><td bgcolor=white> 59.0 </td><td bgcolor=white> 39.7 </td><td bgcolor=white> 21.6 </td><td bgcolor=white> 41.6 </td><td bgcolor=white> 47.7 </td></tr>
</table></tbody>

## YOLOv3 with SPP
Coming soon.

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8>           </th><td bgcolor=white> FPS </td><td bgcolor=white> AP   </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white>  APs  </td><td bgcolor=white>  APm  </td><td bgcolor=white>  APl  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-SPP-320</th><td bgcolor=white>     </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-SPP-416</th><td bgcolor=white>     </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-SPP-512</th><td bgcolor=white>     </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-SPP-608</th><td bgcolor=white>     </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>   </td><td bgcolor=white>  </td><td bgcolor=white>   </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-SPP-640</th><td bgcolor=white>     </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>   </td><td bgcolor=white>  </td><td bgcolor=white>   </td></tr>
</table></tbody>

## YOLOv3 with Dilated Encoder
The DilatedEncoder is proposed by YOLOF.

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8>           </th><td bgcolor=white> FPS </td><td bgcolor=white> AP   </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white>  APs  </td><td bgcolor=white>  APm  </td><td bgcolor=white>  APl  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-320</th><td bgcolor=white>     </td><td bgcolor=white> 31.1 </td><td bgcolor=white> 51.1 </td><td bgcolor=white> 31.7 </td><td bgcolor=white> 10.2 </td><td bgcolor=white> 32.6 </td><td bgcolor=white> 51.2 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-416</th><td bgcolor=white>     </td><td bgcolor=white> 35.0 </td><td bgcolor=white> 56.1 </td><td bgcolor=white> 36.3 </td><td bgcolor=white> 14.6 </td><td bgcolor=white> 37.4 </td><td bgcolor=white> 53.7 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-512</th><td bgcolor=white>     </td><td bgcolor=white> 37.7 </td><td bgcolor=white> 59.3 </td><td bgcolor=white> 39.6 </td><td bgcolor=white> 17.9 </td><td bgcolor=white> 40.4 </td><td bgcolor=white> 54.4 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-640</th><td bgcolor=white>     </td><td bgcolor=white> 38.7 </td><td bgcolor=white> 60.2 </td><td bgcolor=white> 40.7 </td><td bgcolor=white>  21.3 </td><td bgcolor=white> 41.7 </td><td bgcolor=white> 51.7  </td></tr>
</table></tbody>

## YOLOv4-exp
This is an experimental model, not the final version.

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8>           </th><td bgcolor=white> FPS </td><td bgcolor=white> AP   </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white>  APs  </td><td bgcolor=white>  APm  </td><td bgcolor=white>  APl  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv4-320</th><td bgcolor=white>     </td><td bgcolor=white> 36.7 </td><td bgcolor=white> 55.4 </td><td bgcolor=white> 38.2 </td><td bgcolor=white> 15.7 </td><td bgcolor=white> 39.9 </td><td bgcolor=white> 57.5 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv4-416</th><td bgcolor=white>     </td><td bgcolor=white> 39.2 </td><td bgcolor=white> 58.6 </td><td bgcolor=white> 41.4 </td><td bgcolor=white> 20.1 </td><td bgcolor=white> 43.3 </td><td bgcolor=white> 56.8 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv4-512</th><td bgcolor=white>     </td><td bgcolor=white> 40.5 </td><td bgcolor=white> 60.1 </td><td bgcolor=white> 43.1 </td><td bgcolor=white> 22.8 </td><td bgcolor=white> 44.5 </td><td bgcolor=white> 56.1 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv4-640</th><td bgcolor=white>     </td><td bgcolor=white> 40.5 </td><td bgcolor=white> 60.4 </td><td bgcolor=white> 43.5 </td><td bgcolor=white> 24.2 </td><td bgcolor=white> 44.8 </td><td bgcolor=white> 52.0 </td></tr>
</table></tbody>

# Dataset

## VOC Dataset
I copy the download files from the following excellent project:
https://github.com/amdegroot/ssd.pytorch

I have uploaded the VOC2007 and VOC2012 to BaiDuYunDisk, so for researchers in China, you can download them from BaiDuYunDisk:

Link：https://pan.baidu.com/s/1tYPGCYGyC0wjpC97H-zzMQ 

Password：4la9

You will get a ```VOCdevkit.zip```, then what you need to do is just to unzip it and put it into ```data/```. After that, the whole path to VOC dataset is ```data/VOCdevkit/VOC2007``` and ```data/VOCdevkit/VOC2012```.

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

## MSCOCO Dataset
### Download MSCOCO 2017 dataset
Just run ```sh data/scripts/COCO2017.sh```. You will get COCO train2017, val2017, test2017.


# Train
For example:

```Shell
python train.py --cuda \
                -d coco \
                -v yolov1 \
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
                                                               -v yolov1 \
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
               -v yolov1 \
               --weight path/to/weight \
               --img_size 640 \
               --root path/to/dataset/ \
               --show
```
Note that if you try to run my YOLOv4-Exp, please add `--center_sample` because I use this trick in training phase.
For more details, please check the code `models/yolo/yolov4_exp.py`.

# Evaluation
For example

```Shell
python eval.py -d coco-val \
               --cuda \
               -v yolov1 \
               --weight path/to/weight \
               --img_size 640 \
               --root path/to/dataset/
```

# Evaluation on COCO-test-dev
To run on COCO_test-dev(You must be sure that you have downloaded test2017):
```Shell
python eval.py -d coco-test \
               --cuda \
               -v yolov1 \
               --weight path/to/weight \
               --img_size 640 \
                --root path/to/dataset/
```
You will get a `coco_test-dev.json` file. 
Then you should follow the official requirements to compress it into zip format 
and upload it the official evaluation server.
