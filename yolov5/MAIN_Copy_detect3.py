#!/usr/bin/env python3
# YOLOv5  by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (macOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""
#print("Hello World3")
from utils.torch_utils import select_device, time_sync
from utils.plots import Annotator, colors, save_one_box
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from models.common import DetectMultiBackend
#from this import d
import requests
import argparse
import os
import sys
import shutil
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

#print("Hello World2")

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


msg_img = []
token = 'fWWhBE4UiRj1XTVcuknYJmOtAOOdp81V0RN7xJAHTG9'
is_BB_empty = True
name_of_pic = ""
name_of_BB = ""
name_of_confi = ""


@torch.no_grad()
def run(
        weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    source = str(source)
    save_img = not nosave and not source.endswith(
        '.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith(
        '.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name,
                              exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True,
                                                          exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(
        weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    flag = False
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], [0.0, 0.0, 0.0]
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(
            save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(
            pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            is_BB_empty = True
            parent_dir = "/home/leaf/Documents/melProject/yolov5/Summary/"  # Edit Here
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path

            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + \
                ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            # normalization gain whwh
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(
                im0, line_width=line_thickness, example=str(names))
            directory = file_name_collector(p.name)
            name_of_pic = file_name_collector(p.name) + ".jpg"
            name_of_BB = "Bounding_box" + ".txt"
            name_of_confi = "Confidence_value" + ".txt"
            print("NAME OF PIC = ", name_of_pic)
            str_labels = []
            txt=""
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    im.shape[2:], det[:, :4], im0.shape).round()
                txt = ""
                num_i = 1
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # add to string
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                    txt = txt + " " + str(n.item()) + names[int(c)] + '\n'
                    flag = True
                str_labels = []
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)
                                          ) / gn).view(-1).tolist()  # normalized xywh
                        # label format
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (
                            names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        # print(label)
                        str_labels.append(label)
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' /
                                     names[c] / f'{p.stem}.jpg', BGR=True)
                    BoundingBox(xyxy, file_name_collector(str(p)))
                    is_BB_empty = False
            # Stream results
            if(is_BB_empty == True):
                BoundingBox(0, file_name_collector(str(p)))
                is_BB_empty = False
            im0 = annotator.result()
         #   Label_Collector(str_labels, file_name_collector(
         #       str(p)))  # Get path of pic
            Label_Collector(str_labels, "Confidence_value")  # Get path of pic
            str_labels.clear()
            if view_img:
                if p not in windows:
                    windows.append(p)
                    # allow window resize (Linux)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL |
                                    cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            # release previous video writer
                            vid_writer[i].release()
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        # force *.mp4 suffix on results videos
                        save_path = str(Path(save_path).with_suffix('.mp4'))
                        vid_writer[i] = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)
                if(len(txt) == 0):
                    if(flag == True and int(n.item()) > 0):  # LINE notify
                        LINENotification(ResizeImage(save_path, str(p)), txt)
                    else:
                        ResizeImage(save_path, str(p))
                else:
                    if(flag == True and int(n.item()) > 0):  # LINE notify
                        LINENotification(ResizeImage(
                            save_path, str(p)), txt.rstrip(txt[-1]))
                    else:
                        ResizeImage(save_path, str(p))
                txt = ""
                # CREATE DIRECTORY AND MOVE FILES
                path_dir = os.path.join(parent_dir, directory)
                #os.mkdir(path_dir)
                #shutil.move("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
                shutil.move(parent_dir+name_of_pic, "/home/leaf/Documents/melProject/upload/")
                shutil.move(parent_dir+name_of_BB, "/home/leaf/Documents/melProject/upload/")
                shutil.move(parent_dir+name_of_confi, "/home/leaf/Documents/melProject/upload/")
        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(
        f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def LINENotification(imageName, txt):
    #picture = save_path
    payload = {'message': txt,
               'notificationDisabled': False}
    r = requests.post('https://notify-api.line.me/api/notify',
                      headers={'Authorization': 'Bearer {}'.format(token)}, params=payload, files={'imageFile': open(imageName, 'rb')})


def ResizeImage(imageName, p):
    path_to_save = "/home/leaf/Documents/melProject/yolov5/Summary/"  # images are saved #Edit Here

    img = cv2.imread(imageName, cv2.IMREAD_UNCHANGED)

    print('Original Dimensions : ', img.shape)

    scale_percent = 40  # percent of original size
    height = int(img.shape[0] * scale_percent / 100)
    width = int(img.shape[1] * scale_percent / 100)

    dim = (width, height)

    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    cv2.imwrite(path_to_save + file_name_collector(p) + ".jpg", resized)
    print('Resized Dimensions : ', resized.shape)
    return path_to_save + file_name_collector(p) + ".jpg"


def file_name_collector(text):
    t = text.split('/')
    for s in t:
        if(".jpg" in s):
            a = s.split('.')
            return a[0]


def bubbleSort(d):
    n = len(d)
    data = []
    for i in range(n):
        tmp = d[i]
        t = int(tmp[-2:])
        data.append(t)

    # optimize code, so if the array is already sorted, it doesn't need
    # to go through the entire process
    swapped = False
    # Traverse through all array elements
    for i in range(n-1):
        # range(n) also work but outer loop will
        # repeat one time more than needed.
        # Last i elements are already in place
        for j in range(0, n-i-1):
            # traverse the array from 0 to n-i-1
            # Swap if the element found is greater
            # than the next element
            if data[j] < data[j + 1]:
                swapped = True
                d[j], d[j + 1] = d[j + 1], d[j]
                data[j], data[j + 1] = data[j + 1], data[j]

        if not swapped:
            # if we haven't needed to make a single swap, we
            # can just exit the main loop.
            return


def Label_Collector(txt, namefile):
    path_of_file = "/home/leaf/Documents/melProject/yolov5/Summary/" + namefile + ".txt"  # Edit Here
    file = open(path_of_file, 'a+')
    bubbleSort(txt)
    if len(txt) > 0:
        for i in range(len(txt)):
            file.write(txt[i])
            if(i != len(txt)-1):
                file.write(" ")
    else:
        file.write("None")
    file.close()


def find_xy(list_xy):
    s = list_xy.split(",")
    XY = []
    for t in s:
        if "tensor(" in t:
            s1 = t[8:]
            s2 = s1[:-2]
            XY.append(s2)
    print(XY)
    return XY


def BoundingBox(xy, NameOfFile):
    print(NameOfFile)
    path_of_file = "/home/leaf/Documents/melProject/yolov5/Summary/" + \
        "Bounding_box" + ".txt"  # Edit Here
    if(xy == 0):
        file = open(path_of_file, 'a+')
        file.write("None")
    else:
        #print("Size of xy",len(xy))
        file = open(path_of_file, 'a+')
        x_y = find_xy(str(xy))
        for i in x_y:
            file.write(str(i) + " ")
        file.write("\n")
    file.close()


def Move_all_pics():
    i = 0


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default=ROOT / '/home/leaf/Documents/melProject/yolov5/runs/train/exp/weights/best.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT /
                        '/home/leaf/Documents/melProject/Cam1', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT /
                        'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+',
                        type=int, default=[416], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float,
                        default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000,
                        help='maximum detections per image')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true',
                        help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true',
                        help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true',
                        help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--visualize', action='store_true',
                        help='visualize features')
    parser.add_argument('--update', action='store_true',
                        help='update all models')
    parser.add_argument('--project', default=ROOT /
                        'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3,
                        type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False,
                        action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False,
                        action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true',
                        help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true',
                        help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    #print("Hello World1")
    opt = parse_opt()
    main(opt)
    print("yolo Detect Success")
