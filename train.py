from __future__ import division

import os
import argparse
import time
import cv2
import numpy as np

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from data.voc import VOCDetection
from data.coco import COCODataset
from data import config
from data.transforms import TrainTransforms, ValTransforms

from utils.com_flops_params import FLOPs_and_Params
from utils.misc import detection_collate
from utils.misc import ModelEMA

from evaluator.cocoapi_evaluator import COCOAPIEvaluator
from evaluator.vocapi_evaluator import VOCAPIEvaluator

from engine import train_one_epoch, evaluate, set_lr


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Detection')
    # basic
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')
    parser.add_argument('--batch_size', default=16, type=int, 
                        help='Batch size for training')
    parser.add_argument('--lr', default=1e-3, type=float, 
                        help='initial learning rate')
    parser.add_argument('--max_epoch', type=int, default=200,
                        help='The upper bound of warm-up')
    parser.add_argument('--lr_epoch', nargs='+', default=[100, 150], type=int,
                        help='lr epoch to decay')
    parser.add_argument('--wp_epoch', type=int, default=2,
                        help='The upper bound of warm-up')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='start epoch to train')
    parser.add_argument('-r', '--resume', default=None, type=str, 
                        help='keep training')
    parser.add_argument('--num_workers', default=8, type=int, 
                        help='Number of workers used in dataloading')
    parser.add_argument('--num_gpu', default=1, type=int, 
                        help='Number of GPUs to train')
    parser.add_argument('--eval_epoch', type=int,
                            default=10, help='interval between evaluations')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--save_folder', default='weights/', type=str, 
                        help='Gamma update for SGD')
    parser.add_argument('--vis', action='store_true', default=False,
                        help='visualize target.')

    # model
    parser.add_argument('-v', '--version', default='yolov1',
                        help='yolov1, yolov2, yolov3, yolov4')

    # dataset
    parser.add_argument('--root', default='/mnt/share/ssd2/dataset',
                        help='data root')
    parser.add_argument('-d', '--dataset', default='coco',
                        help='coco, widerface, crowdhuman')
    
    # train trick
    parser.add_argument('--no_warmup', action='store_true', default=False,
                        help='do not use warmup')
    parser.add_argument('-ms', '--multi_scale', action='store_true', default=False,
                        help='use multi-scale trick')      
    parser.add_argument('--ema', action='store_true', default=False,
                        help='use ema training trick')
    parser.add_argument('--center_sample', action='store_true', default=False,
                        help='use center sample for labels')

    # DDP train
    parser.add_argument('-dist', '--distributed', action='store_true', default=False,
                        help='distributed training')
    parser.add_argument('--local_rank', type=int, default=0, 
                        help='local_rank')
    parser.add_argument('--sybn', action='store_true', default=False, 
                        help='use sybn.')

    return parser.parse_args()


def train():
    args = parse_args()
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")

    # path to save model
    path_to_save = os.path.join(args.save_folder, args.dataset, args.version)
    os.makedirs(path_to_save, exist_ok=True)

    # set distributed
    local_rank = 0
    if args.distributed:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = torch.distributed.get_rank()
        print(local_rank)
        torch.cuda.set_device(local_rank)

    # cuda
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

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

    else:
        print('Unknown model name...')
        exit(0)
    # YOLO Config
    cfg = config.yolo_cfg
    train_size = cfg['train_size']
    val_size = cfg['val_size']

    # dataset and evaluator
    if args.dataset == 'voc':
        data_dir = os.path.join(args.root, 'VOCdevkit')
        num_classes = 20
        dataset = VOCDetection(
                        data_dir=data_dir,
                        img_size=train_size,
                        transform=TrainTransforms(train_size))

        evaluator = VOCAPIEvaluator(
                        data_dir=data_dir,
                        img_size=val_size,
                        device=device,
                        transform=ValTransforms(val_size))

    elif args.dataset == 'coco':
        data_dir = os.path.join(args.root, 'COCO')
        num_classes = 80
        dataset = COCODataset(
                    data_dir=data_dir,
                    img_size=train_size,
                    transform=TrainTransforms(train_size))

        evaluator = COCOAPIEvaluator(
                        data_dir=data_dir,
                        img_size=val_size,
                        device=device,
                        transform=ValTransforms(val_size)
                        )
    
    else:
        print('unknow dataset !! Only support voc and coco !!')
        exit(0)
    
    # dataloader
    dataloader = build_dataloader(args, dataset, detection_collate)
    
    print('Training model on:', args.dataset)
    print('The dataset size:', len(dataset))
    print("----------------------------------------------------------")

    # build model
    anchor_size = None if args.version == 'yolov1' else cfg['anchor_size']
    net = yolo_net(device=device, 
                   img_size=train_size, 
                   num_classes=num_classes, 
                   trainable=True, 
                   anchor_size=anchor_size)
    model = net

    # SyncBatchNorm
    if args.sybn and args.cuda and args.num_gpu > 1:
        print('use SyncBatchNorm ...')
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = model.to(device).train()
    # compute FLOPs and Params
    FLOPs_and_Params(model=model, size=train_size)

    # DDP
    if args.distributed and args.num_gpu > 1:
        print('using DDP ...')
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
     
    # keep training
    if args.resume is not None:
        print('keep training model: %s' % (args.resume))
        model.load_state_dict(torch.load(args.resume, map_location=device))

    # EMA
    ema = ModelEMA(model) if args.ema else None

    # use tfboard
    tblogger = None
    if args.tfboard:
        print('use tensorboard')
        from torch.utils.tensorboard import SummaryWriter
        c_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        log_path = os.path.join('log/', args.dataset, c_time)
        os.makedirs(log_path, exist_ok=True)

        tblogger = SummaryWriter(log_path)
    
    # optimizer setup
    tmp_lr = args.lr
    optimizer = optim.SGD(model.parameters(), 
                            lr=tmp_lr, 
                            momentum=0.9,
                            weight_decay=5e-4)

    batch_size = args.batch_size
    epoch_size = len(dataset) // (batch_size * args.num_gpu)
    best_map = -100.

    # start training loop
    for epoch in range(args.start_epoch, args.max_epoch):
        if args.distributed:
            dataloader.sampler.set_epoch(epoch)        

        # use step lr decay
        if epoch in args.lr_epoch:
            tmp_lr = tmp_lr * 0.1
            set_lr(optimizer, tmp_lr)

        # train one epoch
        train_size, tmp_lr = train_one_epoch(
                                    args=args, 
                                    epoch=epoch,
                                    max_epoch=args.max_epoch,
                                    epoch_size=epoch_size,
                                    cfg=cfg,
                                    train_size=train_size,
                                    tmp_lr=tmp_lr,
                                    model=model, 
                                    dataloader=dataloader, 
                                    optimizer=optimizer,
                                    anchor_size=anchor_size,
                                    ema=ema,
                                    tblogger=tblogger)
        
        # evaluation
        if (epoch + 1) % args.eval_epoch == 0 or (epoch + 1) == args.max_epoch:
            if args.ema:
                model_eval = ema.ema
            else:
                model_eval = model.module if args.distributed else model
            
            # set eval mode
            model_eval.trainable = False
            model_eval.set_grid(val_size)
            model_eval.eval()

            if local_rank == 0:
                evaluate(args=args, 
                        epoch=epoch,
                        model=model_eval, 
                        evaluator=evaluator,
                        best_map=best_map,
                        path_to_save=path_to_save,
                        tblogger=tblogger)
            
            if args.distributed:
                # wait for all processes to synchronize
                dist.barrier()

            # set train mode.
            model_eval.trainable = True
            model_eval.set_grid(train_size)
            model_eval.eval()

    if args.tfboard:
        tblogger.close()


def build_dataloader(args, dataset, collate_fn=None):
    # distributed
    if args.distributed and args.num_gpu > 1:
        # dataloader
        dataloader = torch.utils.data.DataLoader(
                        dataset=dataset, 
                        batch_size=args.batch_size, 
                        collate_fn=collate_fn,
                        num_workers=args.num_workers,
                        pin_memory=True,
                        sampler=torch.utils.data.distributed.DistributedSampler(dataset)
                        )

    else:
        # dataloader
        dataloader = torch.utils.data.DataLoader(
                        dataset=dataset, 
                        shuffle=True,
                        batch_size=args.batch_size, 
                        collate_fn=collate_fn,
                        num_workers=args.num_workers,
                        pin_memory=True
                        )
    return dataloader


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def vis_data(images, targets, img_size):
    # vis data
    mean=(0.406, 0.456, 0.485)
    std=(0.225, 0.224, 0.229)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    img = images[0].permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
    img = ((img * std + mean)*255).astype(np.uint8)
    cv2.imwrite('1.jpg', img)

    img_ = cv2.imread('1.jpg')
    for box in targets[0]:
        xmin, ymin, xmax, ymax = box[:-1]
        # print(xmin, ymin, xmax, ymax)
        xmin *= img_size
        ymin *= img_size
        xmax *= img_size
        ymax *= img_size
        cv2.rectangle(img_, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)

    cv2.imshow('img', img_)
    cv2.waitKey(0)


if __name__ == '__main__':
    train()
