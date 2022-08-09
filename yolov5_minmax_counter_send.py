# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
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

import argparse
# from dis import _HaveCodeOrStringType
from ipaddress import ip_address
import os
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

import socket

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

# buffer_size = 4092

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((socket.gethostname(), 7010))

def socket_send(message):
    # s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # s.bind((socket.gethostname(), 7010))
    s.listen(8)
    clientsocket, address = s.accept()
    print(f"Connection from {address} has been established!")
    clientsocket.send(bytes(str(message), 'utf-8'))
    # clientsocket.send(str(message), 'utf-8')
    clientsocket.close()


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
        person_count = 0,
        car_count = 0,
        motorcycle_count = 0,
        traffic_light_count = 0,
        count = 0, # Frame Number Difference
        cpf = 50, # Frame Number Difference frame (per sec)
        i = 0, # Frame Number Difference
        difference = 0.0, # Frame Number Difference
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir #comment outed

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
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
    # vid_path, vid_writer = [None] * bs, [None] * bs # comment outed

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
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
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        # label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}') # label with probability
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]}') # label without probability <- me added
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                    # print(s) # me added
                    current_person_num = str(str(s).split(',')[0]).split(' ')[2] # The number of object which has the largest count. Person count seems to be here.
                    # current_person_num_former = str(str(s).split(',')[0]).split(' ')
                    # current_person_num = str(current_person_num_former).find('persons')
                    # current_person_num_use = current_person_num_former[:current_person_num]


                    # Frame Number Difference
                    if count%cpf == 0:
                        #画像のサイズ
                        # previous_frame = '/Users/yugakoba/Keio_Univ_SFC/CCLab-XMusic/DJ_Learning_AI/DJ_mixes_the_world/yolov5_youtube_osc/object_count_image/object_count_image.png'
                        previous_frame = './object_count_image/object_count_image.png'
                        # laplacian_thr = 800
                        previous_dataset = LoadImages(previous_frame, img_size=imgsz, stride=stride, auto=pt)
                        # print(previous_dataset)
                        for path, im, im007s, vid_cap, previous_s in previous_dataset:
                            t1 = time_sync()
                            im = torch.from_numpy(im).to(device)
                            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                            im /= 255  # 0 - 255 to 0.0 - 1.0
                            if len(im.shape) == 3:
                                im = im[None]  # expand for batch dim
                            t2 = time_sync()
                            dt[0] += t2 - t1

                            # Inference
                            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                            pred = model(im, augment=augment, visualize=visualize)
                            t3 = time_sync()
                            dt[1] += t3 - t2

                            # NMS
                            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
                            dt[2] += time_sync() - t3

                            # Process predictions
                            for i, det in enumerate(pred):  # per image
                                seen += 1
                                p, im007, previous_frame = path, im007s.copy(), getattr(dataset, 'frame', 0)

                                p = Path(p)  # to Path
                                save_path = str(save_dir / p.name)  # im.jpg
                                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                                previous_s += '%gx%g ' % im.shape[2:]  # print string
                                gn = torch.tensor(im007.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                                imc = im007.copy() if save_crop else im007  # for save_crop
                                annotator = Annotator(im007, line_width=line_thickness, example=str(names))
                                if len(det):
                                    # Rescale boxes from img_size to im007 size
                                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im007.shape).round()

                                    # Print results
                                    for c in det[:, -1].unique():
                                        n = (det[:, -1] == c).sum()  # detections per class
                                        previous_s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                                        print(previous_s)
                                    previous_person_num = str(str(previous_s).split(',')[0]).split(' ')[4] # The number of object which has the largest count. Person count seems to be here.
                                    # previous_person_num_former = str(str(previous_s).split(',')[0]).split(' ')
                                    # previous_person_num = str(previous_person_num_former).find('persons')
                                    # previous_person_num_use = previous_person_num_former[:previous_person_num]
                                    # print(previous_person_num_use)
                        try:
                            difference = float(current_person_num) - float(previous_person_num)
                            # difference = float(current_person_num_use) - float(previous_person_num_use)
                            print(s, previous_person_num)
                            print(str(difference))
                            # print('Reset because the diffrence number is: ' + str(difference))
                            if float(difference) < 0:
                                person_count = 0
                                print('Reset because the diffrence number is: ' + str(float(difference)))
                                # cv2.imwrite('/Users/yugakoba/Keio_Univ_SFC/CCLab-XMusic/DJ_Learning_AI/DJ_mixes_the_world/yolov5_youtube_osc/object_count_image/object_count_image.png', im0)
                            else:
                                print('Persons increased.')
                        except UnboundLocalError:
                            continue
                        # cv2.imwrite('/Users/yugakoba/Keio_Univ_SFC/CCLab-XMusic/DJ_Learning_AI/DJ_mixes_the_world/yolov5_youtube_osc/object_count_image/object_count_image.png', im0)
                        cv2.imwrite('./object_count_image/object_count_image.png', im0)
                    
                    # if float(difference) < 0:
                    #     person_count = 0
                    #     print('Reset because the diffrence number is: ' + str(float(difference)))
                    #     cv2.imwrite('/Users/yugakoba/Keio_Univ_SFC/CCLab-XMusic/DJ_Learning_AI/DJ_mixes_the_world/yolov5_youtube_osc/object_count_image/object_count_image.png', im0)
                    # else:
                    #     print('Persons increased.')

                    if label == 'traffic light':
                        traffic_light_count += 1

                    if label == 'person':
                        person_count += 1
                        if person_count % 100 == 0:
                            use_person_num = person_count - traffic_light_count
                            print('person=' + str(person_count))
                            if use_person_num < 0:
                                use_person_num = 1
                            socket_send(use_person_num)
                            print(use_person_num) # <-- send this to MusicVAE

                    # if label == 'car':
                    #     car_count += 1
                    #     if car_count % 100 == 0:
                    #         use_car_num = car_count - traffic_light_count
                    #         print('car=' + str(car_count))
                    #         # socket_send(use_car_num)
                    #         print(use_car_num) # <-- send this to MusicVAE

                    # if label == 'motorcycle':
                    #     motorcycle_count += 1
                    #     if motorcycle_count % 10 == 0:
                    #         use_motorcycle_num = motorcycle_count + traffic_light_count
                    #         print('motorcycle=' + str(motorcycle_count))
                    #         # socket_send(use_motorcycle_num)
                    #         print(use_motorcycle_num) # <-- send this to MusicVAE

            # Stream results
            count = count + 1
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
        # Print time (inference-only)
        # LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)') # Total Results Print

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    # LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    # parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--source', default='https://youtu.be/3kPH7kTphnE', help='file/dir/URL/glob, 0 for webcam') # me added
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
