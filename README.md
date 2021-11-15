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


# This project
In this project, you can enjoy: 
- a new and stronger YOLOv1 !
- a new and stronger YOLOv2 !
- a new and stronger YOLOv3 !
- a new YOLOv4 !
- a new YOLO-Tiny !
- a new YOLO-Nano !

# Weights
I will upload all weight files to Google Drive.

# Experiments
## Tricks
Tricks in this project:
- [x] Augmentations: Flip + Color jitter + RandomCrop + Multi-scale
- [x] Model EMA
- [x] GIoU
- [x] Mosaic Augmentation for my YOLOv4
- [x] Multiple positive samples for my YOLOv4


On the COCO-val:

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8>       </th><td bgcolor=white> Backbone </td><td bgcolor=white> FPS </td><td bgcolor=white> AP   </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white>  APs  </td><td bgcolor=white>  APm  </td><td bgcolor=white>  APl  </td><td bgcolor=white>  GFLOPs  </td><td bgcolor=white>  Params  </td></tr>
<tr><th align="left" bgcolor=#f8f8f8> YOLO-Tiny</th><td bgcolor=white> CSPDarkNet-Tiny </td><td bgcolor=white>     </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>   </td><td bgcolor=white>   </td><td bgcolor=white>   </td><td bgcolor=white>   </td><td bgcolor=white>    </td><td bgcolor=white> </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLO-Nano</th><td bgcolor=white> ShuffleNetv2-1.0x </td><td bgcolor=white>     </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>   </td><td bgcolor=white>   </td><td bgcolor=white>   </td><td bgcolor=white>   </td><td bgcolor=white>    </td><td bgcolor=white> </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv1</th><td bgcolor=white> ResNet50 </td><td bgcolor=white>     </td><td bgcolor=white> 35.2 </td><td bgcolor=white> 54.7 </td><td bgcolor=white> 37.1 </td><td bgcolor=white>  14.3 </td><td bgcolor=white>  39.5 </td><td bgcolor=white>  53.4 </td><td bgcolor=white>  41.96   </td><td bgcolor=white> 44.54M </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv2</th><td bgcolor=white> ResNet50 </td><td bgcolor=white>     </td><td bgcolor=white> 36.3 </td><td bgcolor=white> 56.6 </td><td bgcolor=white> 37.7 </td><td bgcolor=white>  15.1 </td><td bgcolor=white>  41.1 </td><td bgcolor=white>  54.0 </td><td bgcolor=white>  42.10   </td><td bgcolor=white> 44.89M </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3</th><td bgcolor=white> DarkNet53 </td><td bgcolor=white>     </td><td bgcolor=white> 38.7 </td><td bgcolor=white> 60.2 </td><td bgcolor=white> 40.7 </td><td bgcolor=white>  21.3 </td><td bgcolor=white> 41.7 </td><td bgcolor=white> 51.7  </td><td bgcolor=white>  76.41   </td><td bgcolor=white> 57.25M </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv4</th><td bgcolor=white> CSPDarkNet53 </td><td bgcolor=white>     </td><td bgcolor=white>    </td><td bgcolor=white>      </td><td bgcolor=white>      </td><td bgcolor=white>       </td><td bgcolor=white>       </td><td bgcolor=white>       </td><td bgcolor=white>  60.55   </td><td bgcolor=white> 52.00M </td></tr>

</table></tbody>

The FPS of all YOLO detectors are measured on a one 2080ti GPU with 640 × 640 size.

# Visualization
I will upload some visualization results:

## YOLO-Tiny
<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8>           </th><td bgcolor=white> FPS </td><td bgcolor=white> AP   </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white>  APs  </td><td bgcolor=white>  APm  </td><td bgcolor=white>  APl  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLO-Tiny-320</th><td bgcolor=white>     </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLO-Tiny-416</th><td bgcolor=white>     </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLO-Tiny-512</th><td bgcolor=white>     </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLO-Tiny-640</th><td bgcolor=white>     </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td></tr>
</table></tbody>


## YOLO-Nano
<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8>           </th><td bgcolor=white> FPS </td><td bgcolor=white> AP   </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white>  APs  </td><td bgcolor=white>  APm  </td><td bgcolor=white>  APl  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLO-Nano-320</th><td bgcolor=white>     </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLO-Nano-416</th><td bgcolor=white>     </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLO-Nano-512</th><td bgcolor=white>     </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLO-Nano-640</th><td bgcolor=white>     </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td></tr>
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
<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8>           </th><td bgcolor=white> FPS </td><td bgcolor=white> AP   </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white>  APs  </td><td bgcolor=white>  APm  </td><td bgcolor=white>  APl  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-320</th><td bgcolor=white>     </td><td bgcolor=white> 31.1 </td><td bgcolor=white> 51.1 </td><td bgcolor=white> 31.7 </td><td bgcolor=white> 10.2 </td><td bgcolor=white> 32.6 </td><td bgcolor=white> 51.2 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-416</th><td bgcolor=white>     </td><td bgcolor=white> 35.0 </td><td bgcolor=white> 56.1 </td><td bgcolor=white> 36.3 </td><td bgcolor=white> 14.6 </td><td bgcolor=white> 37.4 </td><td bgcolor=white> 53.7 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-512</th><td bgcolor=white>     </td><td bgcolor=white> 37.7 </td><td bgcolor=white> 59.3 </td><td bgcolor=white> 39.6 </td><td bgcolor=white> 17.9 </td><td bgcolor=white> 40.4 </td><td bgcolor=white> 54.4 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-640</th><td bgcolor=white>     </td><td bgcolor=white> 38.7 </td><td bgcolor=white> 60.2 </td><td bgcolor=white> 40.7 </td><td bgcolor=white>  21.3 </td><td bgcolor=white> 41.7 </td><td bgcolor=white> 51.7  </td></tr>
</table></tbody>

## YOLOv4
<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8>           </th><td bgcolor=white> FPS </td><td bgcolor=white> AP   </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white>  APs  </td><td bgcolor=white>  APm  </td><td bgcolor=white>  APl  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv4-320</th><td bgcolor=white>     </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>   </td><td bgcolor=white>   </td><td bgcolor=white>   </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv4-416</th><td bgcolor=white>     </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>   </td><td bgcolor=white>   </td><td bgcolor=white>   </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv4-512</th><td bgcolor=white>     </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>   </td><td bgcolor=white>   </td><td bgcolor=white>   </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv4-640</th><td bgcolor=white>     </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>   </td><td bgcolor=white>   </td><td bgcolor=white>   </td></tr>
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
```Shell
python train.py --cuda \
                -d [select a dataset: voc or coco] \
                -v [select a model] \
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
                                                               --cuda -v [select a model] \
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
```Shell
python test.py -d [select a dataset: voc or coco] \
               --cuda \
               -v [select a model] \
               --trained_model [ Please input the path to model dir. ] \
               --img_size 640 \
                --root path/to/dataset/
```

# Evaluation
```Shell
python eval.py -d [select a dataset: voc or coco-val] \
               --cuda \
               -v [select a model] \
               --trained_model [ Please input the path to model dir. ] \
               --img_size 640 \
                --root path/to/dataset/
```

# Evaluation on COCO-test-dev
To run on COCO_test-dev(You must be sure that you have downloaded test2017):
```Shell
python eval.py -d coco-test \
               --cuda \
               -v [select a model] \
               --trained_model [ Please input the path to model dir. ] \
               --img_size 640 \
                --root path/to/dataset/
```
You will get a `coco_test-dev.json` file. 
Then you should follow the official requirements to compress it into zip format 
and upload it the official evaluation server.
