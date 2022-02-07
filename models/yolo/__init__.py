from .yolov1 import YOLOv1
from .yolov2 import YOLOv2
from .yolov3 import YOLOv3
from .yolov4 import YOLOv4
from .yolov5 import YOLOv5
from .yolo_tiny import YOLOTiny
from .yolo_nano import YOLONano
from .yolo_tr import YOLOTR


# build YOLO detector
def build_model(args, cfg, device, num_classes=80, trainable=False):
    
    if args.model == 'yolov1':
        print('Build YOLOv1 ...')
        model = YOLOv1(cfg=cfg,
                        device=device, 
                        img_size=args.img_size, 
                        num_classes=num_classes, 
                        trainable=trainable,
                        conf_thresh=args.conf_thresh,
                        nms_thresh=args.nms_thresh,
                        center_sample=args.center_sample)
    elif args.model == 'yolov2':
        print('Build YOLOv2 ...')
        model = YOLOv2(cfg=cfg,
                        device=device, 
                        img_size=args.img_size, 
                        num_classes=num_classes, 
                        trainable=trainable,
                        conf_thresh=args.conf_thresh,
                        nms_thresh=args.nms_thresh,
                        center_sample=args.center_sample)
    elif args.model == 'yolov3':
        print('Build YOLOv3 ...')
        model = YOLOv3(cfg=cfg,
                        device=device, 
                        img_size=args.img_size, 
                        num_classes=num_classes, 
                        trainable=trainable,
                        conf_thresh=args.conf_thresh,
                        nms_thresh=args.nms_thresh,
                        center_sample=args.center_sample)
    elif args.model == 'yolov3_spp':
        print('Build YOLOv3 with SPP ...')
        model = YOLOv3(cfg=cfg,
                        device=device, 
                        img_size=args.img_size, 
                        num_classes=num_classes, 
                        trainable=trainable,
                        conf_thresh=args.conf_thresh,
                        nms_thresh=args.nms_thresh,
                        center_sample=args.center_sample)
    elif args.model == 'yolov3_de':
        print('Build YOLOv3 with DilatedEncoder ...')
        model = YOLOv3(cfg=cfg,
                        device=device, 
                        img_size=args.img_size, 
                        num_classes=num_classes, 
                        trainable=trainable,
                        conf_thresh=args.conf_thresh,
                        nms_thresh=args.nms_thresh,
                        center_sample=args.center_sample)
    elif args.model == 'yolov4':
        print('Build YOLOv4 ...')
        model = YOLOv4(cfg=cfg,
                        device=device, 
                        img_size=args.img_size, 
                        num_classes=num_classes, 
                        trainable=trainable,
                        conf_thresh=args.conf_thresh,
                        nms_thresh=args.nms_thresh,
                        center_sample=args.center_sample)
    elif args.model == 'yolov5_s':
        print('Build YOLOv5-S ...')
        model = YOLOv5(cfg=cfg,
                        device=device, 
                        img_size=args.img_size, 
                        num_classes=num_classes, 
                        trainable=trainable,
                        conf_thresh=args.conf_thresh,
                        nms_thresh=args.nms_thresh,
                        center_sample=args.center_sample)
    elif args.model == 'yolov5_m':
        print('Build YOLOv5-M ...')
        model = YOLOv5(cfg=cfg,
                        device=device, 
                        img_size=args.img_size, 
                        num_classes=num_classes, 
                        trainable=trainable,
                        conf_thresh=args.conf_thresh,
                        nms_thresh=args.nms_thresh,
                        center_sample=args.center_sample)
    elif args.model == 'yolov5_l':
        print('Build YOLOv5-L ...')
        model = YOLOv5(cfg=cfg,
                        device=device, 
                        img_size=args.img_size, 
                        num_classes=num_classes, 
                        trainable=trainable,
                        conf_thresh=args.conf_thresh,
                        nms_thresh=args.nms_thresh,
                        center_sample=args.center_sample)
    elif args.model == 'yolov5_x':
        print('Build YOLOv5-X ...')
        model = YOLOv5(cfg=cfg,
                        device=device, 
                        img_size=args.img_size, 
                        num_classes=num_classes, 
                        trainable=trainable,
                        conf_thresh=args.conf_thresh,
                        nms_thresh=args.nms_thresh,
                        center_sample=args.center_sample)
    elif args.model == 'yolov5_t':
        print('Build YOLOv5-Tiny ...')
        model = YOLOv5(cfg=cfg,
                        device=device, 
                        img_size=args.img_size, 
                        num_classes=num_classes, 
                        trainable=trainable,
                        conf_thresh=args.conf_thresh,
                        nms_thresh=args.nms_thresh,
                        center_sample=args.center_sample)
    elif args.model == 'yolov5_n':
        print('Build YOLOv5-Nano ...')
        model = YOLOv5(cfg=cfg,
                        device=device, 
                        img_size=args.img_size, 
                        num_classes=num_classes, 
                        trainable=trainable,
                        conf_thresh=args.conf_thresh,
                        nms_thresh=args.nms_thresh,
                        center_sample=args.center_sample)    
    elif args.model == 'yolo_tiny':
        print('Build YOLO-Tiny ...')
        model = YOLOTiny(cfg=cfg,
                        device=device, 
                        img_size=args.img_size, 
                        num_classes=num_classes, 
                        trainable=trainable,
                        conf_thresh=args.conf_thresh,
                        nms_thresh=args.nms_thresh,
                        center_sample=args.center_sample)
    elif args.model == 'yolo_nano':
        print('Build YOLO-Nano ...')
        model = YOLONano(cfg=cfg,
                        device=device, 
                        img_size=args.img_size, 
                        num_classes=num_classes, 
                        trainable=trainable,
                        conf_thresh=args.conf_thresh,
                        nms_thresh=args.nms_thresh,
                        center_sample=args.center_sample)
    elif args.model == 'yolo_tr':
        print('Build YOLO-TR ...')
        model = YOLOTR(cfg=cfg,
                        device=device, 
                        img_size=args.img_size, 
                        num_classes=num_classes, 
                        trainable=trainable,
                        conf_thresh=args.conf_thresh,
                        nms_thresh=args.nms_thresh,
                        center_sample=args.center_sample)
                        
    return model
