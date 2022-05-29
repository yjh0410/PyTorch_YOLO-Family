import argparse
import cv2
import os
import time
import numpy as np
import torch

from config.yolo_config import yolo_config
from data.voc import VOC_CLASSES, VOCDetection
from data.coco import coco_class_index, coco_class_labels, COCODataset
from data.transforms import ValTransforms
from utils.misc import TestTimeAugmentation

from models.yolo import build_model


parser = argparse.ArgumentParser(description='YOLO Detection')
# basic
parser.add_argument('-size', '--img_size', default=640, type=int,
                    help='img_size')
parser.add_argument('--show', action='store_true', default=False,
                    help='show the visulization results.')
parser.add_argument('-vs', '--visual_threshold', default=0.35, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', action='store_true', default=False, 
                    help='use cuda.')
parser.add_argument('--save_folder', default='det_results/', type=str,
                    help='Dir to save results')
# model
parser.add_argument('-m', '--model', default='yolov1',
                    help='yolov1, yolov2, yolov3, yolov3_spp, yolov3_de, '
                            'yolov4, yolo_tiny, yolo_nano')
parser.add_argument('--weight', default='weight/',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--conf_thresh', default=0.1, type=float,
                    help='NMS threshold')
parser.add_argument('--nms_thresh', default=0.45, type=float,
                    help='NMS threshold')
parser.add_argument('--center_sample', action='store_true', default=False,
                    help='center sample trick.')
# dataset
parser.add_argument('--root', default='/mnt/share/ssd2/dataset',
                    help='data root')
parser.add_argument('-d', '--dataset', default='coco',
                    help='coco.')
# TTA
parser.add_argument('-tta', '--test_aug', action='store_true', default=False,
                    help='use test augmentation.')

args = parser.parse_args()



def plot_bbox_labels(img, bbox, label=None, cls_color=None, text_scale=0.4):
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
    # plot bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), cls_color, 2)
    
    if label is not None:
        # plot title bbox
        cv2.rectangle(img, (x1, y1-t_size[1]), (int(x1 + t_size[0] * text_scale), y1), cls_color, -1)
        # put the test on the title bbox
        cv2.putText(img, label, (int(x1), int(y1 - 5)), 0, text_scale, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    return img


def visualize(img, 
              bboxes, 
              scores, 
              cls_inds, 
              vis_thresh, 
              class_colors, 
              class_names, 
              class_indexs=None, 
              dataset_name='voc'):
    ts = 0.4
    for i, bbox in enumerate(bboxes):
        if scores[i] > vis_thresh:
            cls_id = int(cls_inds[i])
            if dataset_name == 'coco':
                cls_color = class_colors[cls_id]
                cls_id = class_indexs[cls_id]
            else:
                cls_color = class_colors[cls_id]
                
            if len(class_names) > 1:
                mess = '%s: %.2f' % (class_names[cls_id], scores[i])
            else:
                cls_color = [255, 0, 0]
                mess = None
            img = plot_bbox_labels(img, bbox, mess, cls_color, text_scale=ts)

    return img
        

def test(args,
         net, 
         device, 
         dataset,
         transforms=None,
         vis_thresh=0.4, 
         class_colors=None, 
         class_names=None, 
         class_indexs=None, 
         show=False,
         test_aug=None, 
         dataset_name='coco'):
    num_images = len(dataset)
    save_path = os.path.join('det_results/', args.dataset, args.model)
    os.makedirs(save_path, exist_ok=True)

    for index in range(num_images):
        print('Testing image {:d}/{:d}....'.format(index+1, num_images))
        image, _ = dataset.pull_image(index)

        h, w, _ = image.shape
        size = np.array([[w, h, w, h]])

        # prepare
        x, _, _, scale, offset = transforms(image)
        x = x.unsqueeze(0).to(device)

        t0 = time.time()
        # forward
        # test augmentation:
        if test_aug is not None:
            bboxes, scores, cls_inds = test_aug(x, net)
        else:
            # inference
            bboxes, scores, cls_inds = net(x)
        print("detection time used ", time.time() - t0, "s")
        
        # rescale
        bboxes -= offset
        bboxes /= scale
        bboxes *= size

        # vis detection
        img_processed = visualize(
                            img=image,
                            bboxes=bboxes,
                            scores=scores,
                            cls_inds=cls_inds,
                            vis_thresh=vis_thresh,
                            class_colors=class_colors,
                            class_names=class_names,
                            class_indexs=class_indexs,
                            dataset_name=dataset_name
                            )
        if show:
            cv2.imshow('detection', img_processed)
            cv2.waitKey(0)
        # save result
        cv2.imwrite(os.path.join(save_path, str(index).zfill(6) +'.jpg'), img_processed)


if __name__ == '__main__':
    args = parser.parse_args()
    # cuda
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model_name = args.model
    print('Model: ', model_name)

    # dataset and evaluator
    if args.dataset == 'voc':
        data_dir = os.path.join(args.root, 'VOCdevkit')
        class_names = VOC_CLASSES
        class_indexs = None
        num_classes = 20
        dataset = VOCDetection(
                        data_dir=data_dir,
                        img_size=args.img_size,
                        image_sets=[('2007', 'test')])

    elif args.dataset == 'coco':
        data_dir = os.path.join(args.root, 'COCO')
        class_names = coco_class_labels
        class_indexs = coco_class_index
        num_classes = 80
        dataset = COCODataset(
                    data_dir=data_dir,
                    img_size=args.img_size,
                    image_set='val2017')
    
    else:
        print('unknow dataset !! Only support voc and coco !!')
        exit(0)

    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(num_classes)]

    # YOLO Config
    cfg = yolo_config[args.model]
    # build model
    model = build_model(args=args, 
                        cfg=cfg, 
                        device=device, 
                        num_classes=num_classes, 
                        trainable=False)

    # load weight
    model.load_state_dict(torch.load(args.weight, map_location='cpu'), strict=False)
    model = model.to(device).eval()
    print('Finished loading model!')

    # TTA
    test_aug = TestTimeAugmentation(num_classes=num_classes) if args.test_aug else None


    # run
    test(args=args,
        net=model, 
        device=device, 
        dataset=dataset,
        transforms=ValTransforms(args.img_size),
        vis_thresh=args.visual_threshold,
        class_colors=class_colors,
        class_names=class_names,
        class_indexs=class_indexs,
        show=args.show,
        test_aug=test_aug,
        dataset_name=args.dataset)
