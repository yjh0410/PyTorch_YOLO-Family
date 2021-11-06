# A new and strong YOLO family
Recently, I rebuild my YOLO-Family project !!

# Installation
- Pytorch >= 1.1.0 (My Pytorch is torch-1.9.1)
- Python3

# This project
In this project, you can enjoy: 
- a new YOLOv1 !
- a new YOLOv2 !
- a new YOLOv3 !
- a new YOLOv4 !

# Weights
I will upload all weight files to Google Drive.

# Experiments
## Tricks
Tricks in this project:
- [x] Augmentations: Flip + Color jitter + RandomCrop + Multi-scale
- [x] Model EMA
- [x] GIoU

I also tried Mosaic Augmentation but it doesn't work.

On the COCO-val:

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8>       </th><td bgcolor=white> FPS </td><td bgcolor=white> AP   </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white>  APs  </td><td bgcolor=white>  APm  </td><td bgcolor=white>  APl  </td><td bgcolor=white>  GFLOPs  </td><td bgcolor=white>  Params  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv1</th><td bgcolor=white>     </td><td bgcolor=white> 35.2 </td><td bgcolor=white> 54.7 </td><td bgcolor=white> 37.1 </td><td bgcolor=white>  14.3 </td><td bgcolor=white>  39.5 </td><td bgcolor=white>  53.4 </td><td bgcolor=white>  41.96   </td><td bgcolor=white> 44.54M </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv2</th><td bgcolor=white>     </td><td bgcolor=white> 36.3 </td><td bgcolor=white> 56.6 </td><td bgcolor=white> 37.7 </td><td bgcolor=white>  15.1 </td><td bgcolor=white>  41.1 </td><td bgcolor=white>  54.0 </td><td bgcolor=white>  42.10   </td><td bgcolor=white> 44.89M </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3</th><td bgcolor=white>     </td><td bgcolor=white>    </td><td bgcolor=white>      </td><td bgcolor=white>      </td><td bgcolor=white>       </td><td bgcolor=white>       </td><td bgcolor=white>       </td><td bgcolor=white>  76.41   </td><td bgcolor=white> 57.25M </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv4</th><td bgcolor=white>     </td><td bgcolor=white>    </td><td bgcolor=white>      </td><td bgcolor=white>      </td><td bgcolor=white>       </td><td bgcolor=white>       </td><td bgcolor=white>       </td><td bgcolor=white>  60.55   </td><td bgcolor=white> 52.00M </td></tr>

</table></tbody>

The FPS of all YOLO detectors are measured on a one 2080ti GPU with 640 × 640 size.

# Visualization
I will upload some visualization results:

## YOLOv1
<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8>           </th><td bgcolor=white> FPS </td><td bgcolor=white> AP   </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white>  APs  </td><td bgcolor=white>  APm  </td><td bgcolor=white>  APl  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv1-320</th><td bgcolor=white>     </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>   </td><td bgcolor=white>   </td><td bgcolor=white>   </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv1-416</th><td bgcolor=white>     </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>   </td><td bgcolor=white>   </td><td bgcolor=white>   </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv1-512</th><td bgcolor=white>     </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>   </td><td bgcolor=white>   </td><td bgcolor=white>   </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv1-640</th><td bgcolor=white>     </td><td bgcolor=white> 35.2 </td><td bgcolor=white> 54.7 </td><td bgcolor=white> 37.1 </td><td bgcolor=white>  14.3 </td><td bgcolor=white>  39.5 </td><td bgcolor=white>  53.4 </td></tr>
</table></tbody>

## YOLOv2
<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8>           </th><td bgcolor=white> FPS </td><td bgcolor=white> AP   </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white>  APs  </td><td bgcolor=white>  APm  </td><td bgcolor=white>  APl  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv2-320</th><td bgcolor=white>     </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>   </td><td bgcolor=white>   </td><td bgcolor=white>   </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv2-416</th><td bgcolor=white>     </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>   </td><td bgcolor=white>   </td><td bgcolor=white>   </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv2-512</th><td bgcolor=white>     </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>   </td><td bgcolor=white>   </td><td bgcolor=white>   </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv2-640</th><td bgcolor=white>     </td><td bgcolor=white> 36.3 </td><td bgcolor=white> 56.6 </td><td bgcolor=white> 37.7 </td><td bgcolor=white>  15.1 </td><td bgcolor=white>  41.1 </td><td bgcolor=white>  54.0 </td></tr>
</table></tbody>

## YOLOv3
<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8>           </th><td bgcolor=white> FPS </td><td bgcolor=white> AP   </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white>  APs  </td><td bgcolor=white>  APm  </td><td bgcolor=white>  APl  </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-320</th><td bgcolor=white>     </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>   </td><td bgcolor=white>   </td><td bgcolor=white>   </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-416</th><td bgcolor=white>     </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>   </td><td bgcolor=white>   </td><td bgcolor=white>   </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-512</th><td bgcolor=white>     </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>   </td><td bgcolor=white>   </td><td bgcolor=white>   </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> YOLOv3-640</th><td bgcolor=white>     </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>  </td><td bgcolor=white>   </td><td bgcolor=white>   </td><td bgcolor=white>   </td></tr>
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

You can run ```python train.py -h``` to check all optional argument.

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
