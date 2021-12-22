import argparse
import numpy as np
import time
import os
import torch
import torch.backends.cudnn as cudnn

from data import config
import data
from data.transforms import ValTransforms
from data.coco import COCODataset, coco_class_index, coco_class_labels
from utils.com_flops_params import FLOPs_and_Params
from utils import fuse_conv_bn


parser = argparse.ArgumentParser(description='Benchmark')
# Model
parser.add_argument('-v', '--version', default='yolov1',
                    help='yolov1, yolov2, yolov3, yolov4, yolo_tiny, yolo_nano')
parser.add_argument('--fuse_conv_bn', action='store_true', default=False,
                    help='fuse conv and bn')
parser.add_argument('--conf_thresh', default=0.1, type=float,
                    help='confidence threshold')
parser.add_argument('--nms_thresh', default=0.45, type=float,
                    help='NMS threshold')
# data root
parser.add_argument('--root', default='/mnt/share/ssd2/dataset',
                    help='data root')
# basic
parser.add_argument('-size', '--img_size', default=640, type=int or list,
                    help='img_size')
parser.add_argument('--weight', default=None,
                    type=str, help='Trained state_dict file path to open')
# cuda
parser.add_argument('--cuda', action='store_true', default=False, 
                    help='use cuda.')

args = parser.parse_args()


def test(net, device, img_size, testset, transform):
    # Step-1: Compute FLOPs and Params
    FLOPs_and_Params(net, img_size, device)

    # Step-2: Compute FPS
    num_images = 2002
    total_time = 0
    count = 0
    with torch.no_grad():
        for index in range(num_images):
            if index % 500 == 0:
                print('Testing image {:d}/{:d}....'.format(index+1, num_images))
            image, _ = testset.pull_image(index)

            h, w, _ = image.shape
            size = np.array([[w, h, w, h]])

            # prepare
            x, _, _, scale, offset = transform(image)
            x = x.unsqueeze(0).to(device)

            # star time
            torch.cuda.synchronize()
            start_time = time.perf_counter()    

            # inference
            bboxes, scores, cls_inds = net(x)
            
            # rescale
            bboxes -= offset
            bboxes /= scale
            bboxes *= size

            # end time
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time

            # print("detection time used ", elapsed, "s")
            if index > 1:
                total_time += elapsed
                count += 1
            
        print('- FPS :', 1.0 / (total_time / count))



if __name__ == '__main__':
    # get device
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # dataset
    print('test on coco-val ...')
    data_dir = os.path.join(args.root, 'COCO')
    class_names = coco_class_labels
    class_indexs = coco_class_index
    num_classes = 80
    dataset = COCODataset(
                data_dir=data_dir,
                image_set='val2017',
                img_size=args.img_size)

    model_name = args.version
    print('Model: ', model_name)

    # load model and config file
    if model_name == 'yolov1':
        from models.yolov1 import YOLOv1 as yolo_net

    elif model_name == 'yolov2':
        from models.yolov2 import YOLOv2 as yolo_net

    elif model_name == 'yolov3':
        from models.yolov3 import YOLOv3 as yolo_net

    elif model_name == 'yolov4':
        from models.yolov4 import YOLOv4 as yolo_net

    elif model_name == 'yolo_tiny':
        from models.yolo_tiny import YOLOTiny as yolo_net

    elif model_name == 'yolo_nano':
        from models.yolo_nano import YOLONano as yolo_net

    else:
        print('Unknown model name...')
        exit(0)
    # YOLO Config
    cfg = config.yolo_cfg

    # build model
    anchor_size = cfg['anchor_size']
    model = yolo_net(device=device, 
                   img_size=args.img_size, 
                   num_classes=num_classes, 
                   trainable=False,
                   conf_thresh=args.conf_thresh,
                   nms_thresh=args.nms_thresh, 
                   anchor_size=anchor_size)

    # load weight
    if args.weight:
        model.load_state_dict(torch.load(args.weight, map_location=device))
        print('Finished loading model!')
    else:
        print('The path to weight file is None !')
        exit(0)
    model = model.to(device).eval()

    # fuse conv bn
    if args.fuse_conv_bn:
        print('fuse conv and bn ...')
        model = fuse_conv_bn(model)

    # run
    test(net=model, 
        img_size=args.img_size,
        device=device, 
        testset=dataset,
        transform=ValTransforms(args.img_size)
        )
