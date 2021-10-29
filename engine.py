import os
import time
import random
import numpy as np
import cv2
import torch

from utils import create_labels
from utils import distributed_utils


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
    img = img.copy()

    for box in targets[0]:
        xmin, ymin, xmax, ymax = box[:-1]
        # print(xmin, ymin, xmax, ymax)
        xmin *= img_size
        ymin *= img_size
        xmax *= img_size
        ymax *= img_size
        cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)

    cv2.imshow('img', img)
    cv2.waitKey(0)


def train_one_epoch(args, 
                    epoch,
                    max_epoch,
                    epoch_size,
                    cfg,
                    train_size,
                    tmp_lr,
                    model, 
                    dataloader, 
                    optimizer,
                    anchor_size=None,
                    ema=None,
                    tblogger=None):
    base_lr = args.lr
    warmup = not args.no_warmup
    t0 = time.time()
    # train one epoch
    for iter_i, (images, targets) in enumerate(dataloader):
        
        # warmup
        ni = iter_i+epoch*epoch_size
        if epoch < args.wp_epoch and warmup:
            nw = args.wp_epoch * epoch_size
            tmp_lr = base_lr * pow(ni / nw, 4)
            set_lr(optimizer, tmp_lr)

        elif epoch == args.wp_epoch and iter_i == 0 and warmup:
            # warmup is over
            warmup = False
            tmp_lr = base_lr
            set_lr(optimizer, tmp_lr)

        # multi-scale trick
        if iter_i % 10 == 0 and iter_i > 0 and args.multi_scale:
            # randomly choose a new size
            r = cfg['random_size_range']
            train_size = random.randint(r[0], r[1]) * 32
            model.set_grid(train_size)
        if args.multi_scale:
            # interpolate
            images = torch.nn.functional.interpolate(
                                input=images, 
                                size=train_size, 
                                mode='bilinear', 
                                align_corners=False)

        # make labels
        targets = [label.tolist() for label in targets]
        if args.vis:
            vis_data(images, targets, train_size)
            continue
        if args.version == 'yoloq':
            targets = create_labels.gt_creator_with_queris(
                                img_size=train_size, 
                                strides=model.module().stride if args.distributed else model.stride, 
                                label_lists=targets, 
                                center_sample=args.center_sample,
                                num_queries=args.num_queries)
        else:
            targets = create_labels.gt_creator(
                                img_size=train_size, 
                                strides=model.module().stride if args.distributed else model.stride, 
                                label_lists=targets, 
                                anchor_size=anchor_size, 
                                center_sample=args.center_sample)

        # to device
        try:
            images = images.cuda()
            targets = targets.cuda()
        except:
            images = images.cpu()
            targets = targets.cpu()

        # forward
        obj_loss, cls_loss, reg_loss, total_loss = model(images, targets=targets)

        loss_dict = dict(obj_loss=obj_loss,
                            cls_loss=cls_loss,
                            reg_loss=reg_loss,
                            total_loss=total_loss)
        loss_dict_reduced = distributed_utils.reduce_loss_dict(loss_dict)

        # check NAN for loss
        if torch.isnan(total_loss):
            continue

        # backprop
        total_loss.backward()        
        optimizer.step()
        optimizer.zero_grad()

        # ema
        if args.ema:
            ema.update(model)

        # display
        if iter_i % 10 == 0:
            if tblogger is not None:
                # viz loss
                tblogger.add_scalar('obj loss',    loss_dict_reduced['obj_loss'].item(),    iter_i + epoch * epoch_size)
                tblogger.add_scalar('cls loss',    loss_dict_reduced['cls_loss'].item(),    iter_i + epoch * epoch_size)
                tblogger.add_scalar('box loss',    loss_dict_reduced['reg_loss'].item(),    iter_i + epoch * epoch_size)
                tblogger.add_scalar('total loss',  loss_dict_reduced['total_loss'].item(),  iter_i + epoch * epoch_size)
            
            t1 = time.time()
            outstream = ('[Epoch %d/%d][Iter %d/%d][lr %.6f]'
                    '[Loss: obj %.2f || cls %.2f || reg %.2f || total %.2f || size %d || time: %.2f]'
                    % (epoch+1, 
                        max_epoch, 
                        iter_i, 
                        epoch_size, 
                        tmp_lr,
                        loss_dict_reduced['obj_loss'].item(),
                        loss_dict_reduced['cls_loss'].item(), 
                        loss_dict_reduced['reg_loss'].item(),
                        loss_dict_reduced['total_loss'].item(),
                        train_size, 
                        t1-t0))

            print(outstream, flush=True)
            t0 = time.time()

    return train_size, tmp_lr


def evaluate(args, 
             epoch,
             model, 
             evaluator,
             best_map=0.,
             path_to_save=None,
             tblogger=None):

    # evaluate
    evaluator.evaluate(model)

    cur_map = evaluator.map
    if cur_map > best_map:
        # update best-map
        best_map = cur_map
        # save model
        print('Saving state, epoch:', epoch + 1)
        torch.save(model.state_dict(), os.path.join(path_to_save, 
                    args.version + '_' + str(epoch + 1) + '_' + str(round(best_map*100, 2)) + '.pth'))  
    if tblogger is not None:
        if args.dataset == 'voc':
            tblogger.add_scalar('07test/mAP', evaluator.map, epoch)
        elif args.dataset == 'coco':
            tblogger.add_scalar('val/AP50_95', evaluator.ap50_95, epoch)
            tblogger.add_scalar('val/AP50', evaluator.ap50, epoch)
