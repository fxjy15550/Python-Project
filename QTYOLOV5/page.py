import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
import torch
from numpy import random

from torch.backends import cudnn

from yolov5.utils.datasets import LoadStreams, LoadImages
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords, increment_path, apply_classifier, \
    xyxy2xywh, save_one_box
from yolov5.utils.plots import plot_one_box
from yolov5.utils.torch_utils import select_device, time_synchronized, load_classifier


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1042, 921)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        # 显示摄像头画面
        self.cam_frame = QtWidgets.QFrame(self.centralwidget)
        self.cam_frame.setGeometry(QtCore.QRect(10, 110, 521, 571))
        self.cam_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.cam_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.cam_frame.setObjectName("cam_frame")

        self.label_img_show = QtWidgets.QLabel(self.cam_frame)
        self.label_img_show.setGeometry(QtCore.QRect(10, 10, 501, 551))
        self.label_img_show.setObjectName("label_img_show")
        # self.label_img_show.setStyleSheet(("border:2px solid red"))

        # 显示检测画面
        self.detect_frame = QtWidgets.QFrame(self.centralwidget)
        self.detect_frame.setGeometry(QtCore.QRect(540, 110, 491, 571))
        self.detect_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.detect_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.detect_frame.setObjectName("detect_frame")
        self.label_detect_show = QtWidgets.QLabel(self.detect_frame)
        self.label_detect_show.setGeometry(QtCore.QRect(10, 10, 481, 551))
        self.label_detect_show.setObjectName("label_detect_show")
        # self.label_detect_show.setStyleSheet(("border:2px solid green"))
        # 按钮框架
        self.btn_frame = QtWidgets.QFrame(self.centralwidget)
        self.btn_frame.setGeometry(QtCore.QRect(10, 20, 1021, 80))
        self.btn_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.btn_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.btn_frame.setObjectName("frame_3")

        # 按钮水平布局
        self.widget = QtWidgets.QWidget(self.btn_frame)
        self.widget.setGeometry(QtCore.QRect(20, 10, 501, 60))
        self.widget.setObjectName("widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setContentsMargins(0, 20, 20, 20)
        self.horizontalLayout.setSpacing(20)
        self.horizontalLayout.setObjectName("horizontalLayout")
        # 打开摄像头
        self.btn_opencam = QtWidgets.QPushButton(self.widget)
        self.btn_opencam.setObjectName("btn_opencam")
        self.horizontalLayout.addWidget(self.btn_opencam)
        # 加载模型文件
        self.btn_model_add_file = QtWidgets.QPushButton(self.widget)
        self.btn_model_add_file.setObjectName("btn_model_add_file")
        self.horizontalLayout.addWidget(self.btn_model_add_file)
        # 加载img文件
        self.btn_img_add_file = QtWidgets.QPushButton(self.widget)
        self.btn_img_add_file.setObjectName("btn_img_add_file")
        self.horizontalLayout.addWidget(self.btn_img_add_file)
        # 开始检测
        self.btn_detect = QtWidgets.QPushButton(self.widget)
        self.btn_detect.setObjectName("btn_detect")
        self.horizontalLayout.addWidget(self.btn_detect)
        # 退出
        self.btn_exit = QtWidgets.QPushButton(self.widget)
        self.btn_exit.setObjectName("btn_exit")
        self.horizontalLayout.addWidget(self.btn_exit)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1042, 17))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        # 这里将按钮和定义的动作相连，通过click信号连接openfile槽？
        # 加载模型文件
        self.btn_model_add_file.clicked.connect(self.open_model)
        # 加载img文件
        self.btn_img_add_file.clicked.connect(self.open_img)
        # 打开摄像头
        self.btn_opencam.clicked.connect(self.opencam)
        # 开始识别
        self.btn_detect.clicked.connect(self.detect(self.opts()))
        # 这里是将btn_exit按钮和Form窗口相连，点击按钮发送关闭窗口命令
        self.btn_exit.clicked.connect(MainWindow.close)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "目标检测"))
        self.label_img_show.setText(_translate("MainWindow", "摄像头原始画面"))
        self.label_detect_show.setText(_translate("MainWindow", "实时检测效果"))
        self.btn_opencam.setText(_translate("MainWindow", "打开摄像头"))
        self.btn_model_add_file.setText(_translate("MainWindow", "加载模型文件"))
        self.btn_img_add_file.setText(_translate("MainWindow", "加载img文件"))
        self.btn_detect.setText(_translate("MainWindow", "开始检测"))
        self.btn_exit.setText(_translate("MainWindow", "退出"))

    def open_model(self):
        global openfile_name_model
        openfile_name_model, _ = QFileDialog.getOpenFileName(self.btn_model_add_file, '选择模型文件',
                                                             '/home/ljw/桌面/demo/project_demo/pyqt5/yolov3/')
        print('加载模型文件地址为：' + str(openfile_name_model))

    def open_img(self):
        global openfile_name_img
        openfile_name_img, _ = QFileDialog.getOpenFileName(self.btn_img_add_file, '选择img文件',
                                                           '/home/ljw/桌面/demo/project_demo/pyqt5/yolov3/')
        print('加载图片文件地址为：' + str(openfile_name_img))

    def opencam(self):
        self.camcapture = cv2.VideoCapture(0)
        self.timer = QtCore.QTimer()
        self.timer.start()
        self.timer.setInterval(3)  # 0.1s刷新一次
        self.timer.timeout.connect(self.camshow)

    def camshow(self):
        # global self.camimg
        _, self.camimg = self.camcapture.read()
        camimg = cv2.cvtColor(self.camimg, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(camimg.data, camimg.shape[1], camimg.shape[0], QtGui.QImage.Format_RGB888)
        self.label_img_show.setPixmap(QtGui.QPixmap.fromImage(showImage))

    def opts(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
        parser.add_argument('--source', default='yolov5/data/images', help='source')  # file/folder, 0 for webcam
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default='runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        args = parser.parse_args()
        args.img_size = check_img_size(args.img_size)
        return args

    def detect(self, opt):
        source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
        save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
        webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

        # Directories
        save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        device = select_device(opt.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
        model.to(device).eval()
        if half:
            model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(
                device).eval()

        # Set Dataloader
        if webcam:
            view_img = True
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz)
        else:
            view_img = True
            dataset = LoadImages(source, img_size=imgsz)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        t0 = time.time()
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                       agnostic=opt.agnostic_nms)
            t2 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    if save_img or opt.save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = f'{names[c]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[c], line_thickness=3)
                        if opt.save_crop:
                            save_one_box(xyxy, im0s, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            print(f"Results saved to {save_dir}{s}")

        print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = QMainWindow()
    ui = Ui_MainWindow()
    # 向主窗口上添加控件
    ui.setupUi(mainWindow)
    mainWindow.show()
    sys.exit(app.exec_())
