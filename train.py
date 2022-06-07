from __future__ import division

import os
import argparse
import time
import math
import random
from copy import deepcopy

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from config.yolo_config import yolo_config
from data.voc import VOCDetection
from data.coco import COCODataset
from data.transforms import TrainTransforms, ColorTransforms, ValTransforms

from utils import distributed_utils
from utils import create_labels
from utils.vis import vis_data, vis_targets
from utils.com_flops_params import FLOPs_and_Params
from utils.criterion import build_criterion
from utils.misc import detection_collate
from utils.misc import ModelEMA
from utils.criterion import build_criterion

from models.yolo import build_model

from evaluator.cocoapi_evaluator import COCOAPIEvaluator
from evaluator.vocapi_evaluator import VOCAPIEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Detection')
    # basic
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')
    parser.add_argument('--batch_size', default=16, type=int, 
                        help='Batch size for training')
    parser.add_argument('--lr', default=1e-3, type=float, 
                        help='initial learning rate')
    parser.add_argument('--img_size', type=int, default=640,
                        help='The upper bound of warm-up')
    parser.add_argument('--multi_scale_range', nargs='+', default=[10, 20], type=int,
                        help='lr epoch to decay')
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
                        help='path to save weight')
    parser.add_argument('--vis_data', action='store_true', default=False,
                        help='visualize images and labels.')
    parser.add_argument('--vis_targets', action='store_true', default=False,
                        help='visualize assignment.')

    # Optimizer & Schedule
    parser.add_argument('--optimizer', default='sgd', type=str,
                        help='sgd, adamw')
    parser.add_argument('--lr_schedule', default='step', type=str,
                        help='step, cos')
    parser.add_argument('--grad_clip', default=None, type=float,
                        help='clip gradient')

    # model
    parser.add_argument('-m', '--model', default='yolov1',
                        help='yolov1, yolov2, yolov3, yolov3_spp, yolov3_de, '
                             'yolov4, yolo_tiny, yolo_nano')
    parser.add_argument('--conf_thresh', default=0.001, type=float,
                        help='NMS threshold')
    parser.add_argument('--nms_thresh', default=0.6, type=float,
                        help='NMS threshold')

    # dataset
    parser.add_argument('--root', default='/mnt/share/ssd2/dataset',
                        help='data root')
    parser.add_argument('-d', '--dataset', default='coco',
                        help='coco, widerface, crowdhuman')
    
    # Loss
    parser.add_argument('--loss_obj_weight', default=1.0, type=float,
                        help='weight of obj loss')
    parser.add_argument('--loss_cls_weight', default=1.0, type=float,
                        help='weight of cls loss')
    parser.add_argument('--loss_reg_weight', default=1.0, type=float,
                        help='weight of reg loss')
    parser.add_argument('--scale_loss', default='batch', type=str,
                        help='scale loss: batch or positive samples')

    # train trick
    parser.add_argument('--no_warmup', action='store_true', default=False,
                        help='do not use warmup')
    parser.add_argument('-ms', '--multi_scale', action='store_true', default=False,
                        help='use multi-scale trick')      
    parser.add_argument('--ema', action='store_true', default=False,
                        help='use ema training trick')
    parser.add_argument('--mosaic', action='store_true', default=False,
                        help='use Mosaic Augmentation trick')
    parser.add_argument('--mixup', action='store_true', default=False,
                        help='use MixUp Augmentation trick')
    parser.add_argument('--multi_anchor', action='store_true', default=False,
                        help='use multiple anchor boxes as the positive samples')
    parser.add_argument('--center_sample', action='store_true', default=False,
                        help='use center sample for labels')
    parser.add_argument('--accumulate', type=int, default=1,
                        help='accumulate gradient')

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
    path_to_save = os.path.join(args.save_folder, args.dataset, args.model)
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

    # YOLO config
    cfg = yolo_config[args.model]
    train_size = val_size = args.img_size

    # dataset and evaluator
    dataset, evaluator, num_classes = build_dataset(args, train_size, val_size, device)
    # dataloader
    dataloader = build_dataloader(args, dataset, detection_collate)
    # criterioin
    criterion = build_criterion(args, cfg, num_classes)
    
    print('Training model on:', args.dataset)
    print('The dataset size:', len(dataset))
    print("----------------------------------------------------------")

    # build model
    net = build_model(args=args, 
                      cfg=cfg, 
                      device=device, 
                      num_classes=num_classes, 
                      trainable=True)
    model = net

    # SyncBatchNorm
    if args.sybn and args.cuda and args.num_gpu > 1:
        print('use SyncBatchNorm ...')
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = model.to(device).train()
    # compute FLOPs and Params
    if local_rank == 0:
        model_copy = deepcopy(model)
        model_copy.trainable = False
        model_copy.eval()
        FLOPs_and_Params(model=model_copy, size=train_size)
        model_copy.trainable = True
        model_copy.train()
        

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
    base_lr = args.lr
    tmp_lr = args.lr
    if args.optimizer == 'sgd':
        print('use SGD with momentum ...')
        optimizer = optim.SGD(model.parameters(), 
                                lr=tmp_lr, 
                                momentum=0.9,
                                weight_decay=5e-4)
    elif args.optimizer == 'adamw':
        print('use AdamW ...')
        optimizer = optim.AdamW(model.parameters(), 
                                lr=tmp_lr, 
                                weight_decay=5e-4)

    batch_size = args.batch_size
    epoch_size = len(dataset) // (batch_size * args.num_gpu)
    best_map = -100.
    warmup = not args.no_warmup

    t0 = time.time()
    # start training loop
    for epoch in range(args.start_epoch, args.max_epoch):
        if args.distributed:
            dataloader.sampler.set_epoch(epoch)            

        # use step lr decay
        if args.lr_schedule == 'step':
            if epoch in args.lr_epoch:
                tmp_lr = tmp_lr * 0.1
                set_lr(optimizer, tmp_lr)
        # use cos lr decay
        elif args.lr_schedule == 'cos' and not warmup:
            T_max = args.max_epoch - 15
            lr_min = base_lr * 0.1 * 0.1
            if epoch > T_max:
                # Cos decay is done
                print('Cosine annealing is over !!')
                args.lr_schedule == None
                tmp_lr = lr_min
                set_lr(optimizer, tmp_lr)
            else:
                tmp_lr = lr_min + 0.5*(base_lr - lr_min)*(1 + math.cos(math.pi*epoch / T_max))
                set_lr(optimizer, tmp_lr)

        # train one epoch
        for iter_i, (images, targets) in enumerate(dataloader):
            ni = iter_i + epoch * epoch_size
            # warmup
            if epoch < args.wp_epoch and warmup:
                nw = args.wp_epoch * epoch_size
                tmp_lr = base_lr * pow(ni / nw, 4)
                set_lr(optimizer, tmp_lr)

            elif epoch == args.wp_epoch and iter_i == 0 and warmup:
                # warmup is over
                print('Warmup is over !!')
                warmup = False
                tmp_lr = base_lr
                set_lr(optimizer, tmp_lr)

            # multi-scale trick
            if iter_i % 10 == 0 and iter_i > 0 and args.multi_scale:
                # randomly choose a new size
                r = args.multi_scale_range
                train_size = random.randint(r[0], r[1]) * 32
                model.set_grid(train_size)
            if args.multi_scale:
                # interpolate
                images = torch.nn.functional.interpolate(
                                    input=images, 
                                    size=train_size, 
                                    mode='bilinear', 
                                    align_corners=False)

            targets = [label.tolist() for label in targets]
            # visualize target
            if args.vis_data:
                vis_data(images, targets)
                continue
            # make labels
            targets = create_labels.gt_creator(
                                    img_size=train_size, 
                                    strides=net.stride, 
                                    label_lists=targets, 
                                    anchor_size=cfg["anchor_size"], 
                                    multi_anchor=args.multi_anchor,
                                    center_sample=args.center_sample)
            # visualize assignment
            if args.vis_targets:
                vis_targets(images, targets, cfg["anchor_size"], net.stride)
                continue

            # to device
            images = images.to(device)
            targets = targets.to(device)

            # inference
            pred_obj, pred_cls, pred_iou, targets = model(images, targets=targets)

            # compute loss
            loss_obj, loss_cls, loss_reg, total_loss = criterion(pred_obj, pred_cls, pred_iou, targets)

            # check loss
            if torch.isnan(total_loss):
                continue

            loss_dict = dict(
                loss_obj=loss_obj,
                loss_cls=loss_cls,
                loss_reg=loss_reg,
                total_loss=total_loss
            )
            loss_dict_reduced = distributed_utils.reduce_loss_dict(loss_dict)

            total_loss = total_loss / args.accumulate
            # Backward and Optimize
            total_loss.backward()
            if ni % args.accumulate == 0:
                if args.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

                # ema
                if args.ema:
                    ema.update(model)

            # display
            if iter_i % 10 == 0:
                if args.tfboard:
                    # viz loss
                    tblogger.add_scalar('loss obj',  loss_dict_reduced['loss_obj'].item(),  ni)
                    tblogger.add_scalar('loss cls',  loss_dict_reduced['loss_cls'].item(),  ni)
                    tblogger.add_scalar('loss reg',  loss_dict_reduced['loss_reg'].item(),  ni)
                
                t1 = time.time()
                print('[Epoch %d/%d][Iter %d/%d][lr %.6f][Loss: obj %.2f || cls %.2f || reg %.2f || size %d || time: %.2f]'
                        % (epoch+1, 
                           args.max_epoch, 
                           iter_i, 
                           epoch_size, 
                           tmp_lr,
                           loss_dict['loss_obj'].item(), 
                           loss_dict['loss_cls'].item(), 
                           loss_dict['loss_reg'].item(), 
                           train_size, 
                           t1-t0),
                        flush=True)

                t0 = time.time()

        # evaluation
        if (epoch + 1) % args.eval_epoch == 0 or (epoch + 1) == args.max_epoch:
            if evaluator is None:
                print('No evaluator ...')
                print('Saving state, epoch:', epoch + 1)
                torch.save(model_eval.state_dict(), os.path.join(path_to_save, 
                            args.model + '_' + repr(epoch + 1) + '.pth'))  
                print('Keep training ...')
            else:
                print('eval ...')
                # check ema
                if args.ema:
                    model_eval = ema.ema
                else:
                    model_eval = model.module if args.distributed else model

                # set eval mode
                model_eval.trainable = False
                model_eval.set_grid(val_size)
                model_eval.eval()

                if local_rank == 0:
                    # evaluate
                    evaluator.evaluate(model_eval)

                    cur_map = evaluator.map
                    if cur_map > best_map:
                        # update best-map
                        best_map = cur_map
                        # save model
                        print('Saving state, epoch:', epoch + 1)
                        torch.save(model_eval.state_dict(), os.path.join(path_to_save, 
                                    args.model + '_' + repr(epoch + 1) + '_' + str(round(best_map*100, 2)) + '.pth'))  
                    if args.tfboard:
                        if args.dataset == 'voc':
                            tblogger.add_scalar('07test/mAP', evaluator.map, epoch)
                        elif args.dataset == 'coco':
                            tblogger.add_scalar('val/AP50_95', evaluator.ap50_95, epoch)
                            tblogger.add_scalar('val/AP50', evaluator.ap50, epoch)

                if args.distributed:
                    # wait for all processes to synchronize
                    dist.barrier()

                # set train mode.
                model_eval.trainable = True
                model_eval.set_grid(train_size)
                model_eval.train()

        # close mosaic augmentation
        if args.mosaic and args.max_epoch - epoch == 15:
            print('close Mosaic Augmentation ...')
            dataloader.dataset.mosaic = False
        # close mixup augmentation
        if args.mixup and args.max_epoch - epoch == 15:
            print('close Mixup Augmentation ...')
            dataloader.dataset.mixup = False

    if args.tfboard:
        tblogger.close()


def build_dataset(args, train_size, val_size, device):
    if args.dataset == 'voc':
        data_dir = os.path.join(args.root, 'VOCdevkit')
        num_classes = 20
        dataset = VOCDetection(
                        data_dir=data_dir,
                        img_size=train_size,
                        transform=TrainTransforms(train_size),
                        color_augment=ColorTransforms(train_size),
                        mosaic=args.mosaic,
                        mixup=args.mixup)

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
                    image_set='train2017',
                    transform=TrainTransforms(train_size),
                    color_augment=ColorTransforms(train_size),
                    mosaic=args.mosaic,
                    mixup=args.mixup)

        evaluator = COCOAPIEvaluator(
                        data_dir=data_dir,
                        img_size=val_size,
                        device=device,
                        transform=ValTransforms(val_size)
                        )
    
    else:
        print('unknow dataset !! Only support voc and coco !!')
        exit(0)

    return dataset, evaluator, num_classes


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


if __name__ == '__main__':
    train()
