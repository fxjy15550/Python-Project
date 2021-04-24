import argparse
import numpy
import math
import torch
from PyQt5.QtWidgets import *  # 这两个是pyqt5常用的库
from PyQt5.QtGui import QImage, QIcon, QPixmap  # 可以满足小白大多数功能
from PyQt5.QtCore import pyqtSignal, QThread, QMutex
from detcetUtil import *
from yolov5.utils.general import check_img_size
import os
import sys


class ChildClass(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.TModel = []  # 用来接收主界面的训练好的模型的序号
        self.openfile_name_image = ''  # 存储原始图像的地址
        self.result_name_image = ''  # 存储检测好的图像的地址

    def initUI(self):
        self.resize(1100, 450)  # 缩放界面大小
        self.setWindowTitle("目标检测")  # 设置界面标题
        self.setWindowIcon(QIcon("img/logo.png"))		#设置界面图标
        self.PModelSelectSignal = [0, 0]  # 设置需要预测模型的序号，在下拉框里选择

        myframe = QFrame(self)
        self.label1 = QLabel("检测模型", self)
        combol1 = QComboBox(myframe)
        combol1.addItem("选择检测模型")
        combol1.addItem("YOLOV3")
        combol1.addItem("YOLOV4")
        combol1.activated[str].connect(self.PModelSelect)  # 链接预测模型序号选择函数
        btn1 = QPushButton("选择检测图片", self)
        btn1.clicked.connect(self.select_image)  # 链接检测图片选择函数，本质是打开一个文件夹

        btn2 = QPushButton("开始检测", self)
        btn2.clicked.connect(self.PredictModel)  # 链接预测模型函数

        self.label2 = QLabel("", self)  # 创建一个label，可以存放文字或者图片，在这里是用来存放图片，文本参数为空就会显示为空，留出空白区域，选择好图片时会有函数展示图片
        self.label2.resize(400, 400)
        self.label3 = QLabel("", self)
        self.label3.resize(400, 400)
        label4 = QLabel("      原始图片", self)  # 用来放在图片底部表示这是哪一种图片
        label5 = QLabel("      检测图片", self)
        vlo2 = QHBoxLayout()  # 创建一个子布局，将图片水平排放
        vlo2.addWidget(label4)
        vlo2.addWidget(label5)

        vlo = QHBoxLayout()  # 创建一个子布局，将按钮水平排放
        vlo.addStretch()
        vlo.addWidget(self.label1)
        vlo.addWidget(combol1)
        vlo.addWidget(btn1)
        vlo.addWidget(btn2)
        vlo.addStretch(1)

        vlo1 = QHBoxLayout()  # 创建一个水平布局，将两个提示标签竖直排放
        vlo1.addWidget(self.label2)
        vlo1.addWidget(self.label3)

        hlo = QVBoxLayout(self)  # 创建一个总的垂直布局，将三个子布局垂直排放
        hlo.addLayout(vlo)
        hlo.addLayout(vlo1)
        hlo.addStretch(1)
        hlo.addLayout(vlo2)
        hlo.addStretch(0)
        hlo.addWidget(myframe)

    def GetTModel(self, a):
        self.TModel = a

    def closeEvent(self, event):
        result = QMessageBox.question(self, "提示：", "您真的要退出程序吗", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if result == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def select_image(self):
        self.openfile_name_image, _ = QFileDialog.getOpenFileName(self, "选择照片文件",
                                                                  r"./yolov3/imgtest/")
        # 弹出一个对话窗，是一个文件夹，可以选择一个文件然后返回地址到 self.openfile_name_image中
        print('加载照片文件地址为：' + str(self.openfile_name_image))
        self.label2.setPixmap(QPixmap(str(self.openfile_name_image)))  # 将选中的文件名字传入QPixmap（）中，括号内为文件地址，就会读取这个图片
        self.label2.resize(300, 400)
        self.label2.setScaledContents(True)  # 表示这个label可以可以自适应窗口大小，可以让图片随窗口大小而变化

    def PModelSelect(self, s):
        if s == 'YOLOV3':
            if self.TModel[0] == 1:
                self.PModelSelectSignal[0] = 1
                self.PModelSelectSignal[1] = 0
                print(self.PModelSelectSignal[0])
            else:
                print("模型YOLOV3未训练")  # 如果已经训练好的模型数组里对应的位置为0，则表示该模型未训练
                self.PModelSelectSignal[1] = 0  # 同时也要讲模型选择信号清零，以便下次可以继续选择赋值
        elif s == 'YOLOV4':
            if self.TModel[1] == 1:
                self.PModelSelectSignal[1] = 1
                self.PModelSelectSignal[0] = 0
                print(self.PModelSelectSignal[1])
            else:
                print("模型YOLOV4未训练")
                self.PModelSelectSignal[0] = 0

    def PredictModel(self):
        if self.PModelSelectSignal[0] == 1:
            parser = argparse.ArgumentParser()
            parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
            parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
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
            opt = parser.parse_args()
            print(opt)

            with torch.no_grad():
                detect.detect()
        elif self.PModelSelectSignal[1] == 1:
            print('YOLOV4正在检测')  # 这里应该放入另外一个模型
        else:
            print('没有该模型')

        a = self.openfile_name_image
        a = a.split('/')  # 将预测图片里的编号分离出来
        a = './yolov3/imgtestresult/' + a[-1]  # 将指定路径与图片编号组合，即可得到预测好的图片的路径
        self.label3.setPixmap(QPixmap(a))  # 直接读取预测好的图片
        self.label3.resize(300, 400)
        self.label3.setScaledContents(True)
        print(a)


class MyClass(QWidget):
    def __init__(self):
        super().__init__()  # 继承父类
        self.initUI()  # 自己定义的函数，初始化类界面，里面放着自己各种定义的按钮组件及布局
        self.child_window = ChildClass()  # 子界面的调用，本质和主界面一样是一个类，在这里将其声明为主界面的成员

    def initUI(self):
        self.setWindowTitle("YOLO小目标检测")  # 设置界面名称
        self.setWindowIcon(QIcon("img/logo.png"))		#设计界面的图标，图片放在项目文件夹的子文件夹里就不会出错，名字也要对应
        self.resize(350, 200)  # 设置界面大小
        self.TModelSelectSignal = [0, 0]  # 选择按钮对应的模型
        self.TModel = [0, 0]  # 表示已经训练好的模型编号

        myframe = QFrame(self)  # 实例化一个QFrame可以定义一下风格样式，相当于一个框架，可以移动，其内部组件也可以移动
        btn2 = QPushButton("开始训练模型", self)  # 定义一个按钮，括号里需要一个self，如果需要在类内传递，则应该定义为self.btn2
        btn2.clicked.connect(self.TestModel)  # 将点击事件与一个函数相连，clicked表示按钮的点击事件，还有其他的功能函数，后面连接的是一个类内函数，调用时无需加括号
        btn3 = QPushButton("上传数据集", self)
        btn3.clicked.connect(self.DataExplorerSelect)  # 连接一个选择文件夹的函数
        btn5 = QPushButton("退出程序", self)
        btn5.clicked.connect(self.close)  # 将按钮与关闭事件相连，这个关闭事件是重写的，它自带一个关闭函数，这里重写为点击关闭之后会弹窗提示是否需要关闭
        btn6 = QPushButton("检测", self)
        btn6.clicked.connect(self.show_child)  # 这里将联系弹出子界面函数，具体弹出方式在函数里说明

        combol1 = QComboBox(myframe)  # 定义为一个下拉框，括号里为这个下拉框从属的骨架（框架)
        combol1.addItem("   选择模型")  # 添加下拉选项的文本表示，这里因为没有找到文字对齐方式，所以采用直接打空格，网上说文字对齐需要重写展示函数
        combol1.addItem("   YOLOv3")
        combol1.addItem("   YOLOv4")
        combol1.activated[str].connect(self.TModelSelect)  # |--将选择好的模型序号存到模型选择数组里
        # |--后面的训练函数会根据这个数组判断需要训练哪个模型
        # |--[str]表示会将下拉框里的文字随着选择信号传过去
        # |--activated表示该选项可以被选中并传递信号
        vlo = QVBoxLayout()  # 创建一个垂直布局，需要将需要垂直布局的组件添加进去
        vlo.addWidget(combol1)  # 添加相关组件到垂直布局里
        vlo.addWidget(btn3)
        vlo.addWidget(btn2)
        vlo.addWidget(btn6)
        vlo.addWidget(btn5)
        vlo.addStretch(1)  # 一个伸缩函数，可以一定程度上防止界面放大之后排版不协调
        hlo = QVBoxLayout(self)  # 创建整体框架布局，即主界面的布局
        hlo.addLayout(vlo)  # 将按钮布局添加到主界面的布局之中
        hlo.addWidget(myframe)  # 将框架也加入到总体布局中，当然也可以不需要这框架，直接按照整体框架布局来排版
        self.show()  # 显示主界面

    @staticmethod
    def DataExplorerSelect():
        path = r'D:\pycharm\QTYOLOV3\yolov3\VOCdevkit\VOC2007'
        os.system("explorer.exe %s" % path)

    def show_child(self):
        TModel1 = self.TModel  # |--这是子界面的类内函数
        self.child_window.GetTModel(TModel1)  # |--将训练好的模型序号传到子界面的类内参数里面
        self.child_window.show()  # |--子界面相当于主界面的一个类内成员
        # |--但是本质还是一个界面类，也有show函数将其展示

    def TModelSelect(self, s):  # s是形参，表示传回来的选中的选项的文字
        if s == '   YOLOv3':
            self.TModelSelectSignal[0] = 1  # 如果选中的是YOLOv3-COC就将第一位置1
            # print(self.TModelSelectSignal[0])
        elif s == '   YOLOv4':
            self.TModelSelectSignal[1] = 1  # 如果选中的是YOLO-Efficientnet就将第二位置1
            # print(self.TModelSelectSignal[1])

    def TestModel(self):
        if self.TModelSelectSignal[0] == 1:
            # train.run()
            self.TModelSelectSignal[0] = 0
        else:
            print("没有该模型")

    def closeEvent(self, event):
        result = QMessageBox.question(self, "提示：", "您真的要退出程序吗", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if result == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


class ConsumeThread(QThread):
    sum_person = pyqtSignal(int)
    bbox_id = pyqtSignal(list)

    def __init__(self, qmut_1, queue, parent=None):
        super(ConsumeThread, self).__init__(parent)
        self.queue = queue
        self.Consuming = False
        self.qmut_1 = qmut_1

    def stop(self):
        self.Consuming = False

    def begin(self):
        self.Consuming = True

    def opts(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', type=str, default='yolov5/weights/yolov5x.pt', help='model.pt path')
        parser.add_argument('--source', type=str, default='rtsp://iscas:opqwer12@192.168.100.176:554/Streaming'
                                                          '/Channels/101', help='source')  # file/folder, 0 for webcam
        parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
        parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
        parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        # class 0 is person
        parser.add_argument('--classes', nargs='+', type=int, default=range(0, 1), help='filter by class')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
        args = parser.parse_args()
        args.img_size = check_img_size(args.img_size)

        return args

    def detect(self, opt):
        print("before detect lock")
        self.qmut_1.lock()
        print("after detect lock")
        out, source, weights, view_img, save_txt, imgsz = \
            opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
        webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

        # initialize deepsort
        cfg = get_config()
        cfg.merge_from_file(opt.config_deepsort)
        deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)

        # Initialize
        device = select_device(opt.device)
        if os.path.exists(out):
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
        model.to(device).eval()
        if half:
            model.half()  # to FP16

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

        # Run inference
        self.t0 = time.time()
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

        for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
            if not self.Consuming:
                # dataset.stop_cap()
                raise StopIteration
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

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if not self.Consuming:
                    # dataset.stop_cap()
                    raise StopIteration
                if webcam:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                else:
                    p, s, im0 = path, '', im0s

                s += '%gx%g ' % img.shape[2:]  # print string

                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    n = 0
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, names[int(c)])  # add to string
                    # 将当前帧的总人数发送给前端pyqt界面
                    self.sum_person.emit(n)
                    self.msleep(30)
                    bbox_xywh = []
                    confs = []

                    # Adapt detections to deep sort input format
                    for *xyxy, conf, cls in det:
                        img_h, img_w, _ = im0.shape
                        x_c, y_c, bbox_w, bbox_h = bbox_rel(img_w, img_h, *xyxy)
                        obj = [x_c, y_c, bbox_w, bbox_h]
                        bbox_xywh.append(obj)
                        confs.append([conf.item()])

                    xywhs = torch.Tensor(bbox_xywh)
                    confss = torch.Tensor(confs)

                    # Pass detections to deepsort
                    outputs = deepsort.update(xywhs, confss, im0)

                    # draw boxes for visualization
                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -1]
                        self.bbox_id.emit([bbox_xyxy, identities])
                        self.msleep(30)
                        draw_boxes(im0, bbox_xyxy, identities)

                # Print time (inference + NMS)
                print('%sDone. (%.3fs)' % (s, t2 - t1))
                # Stream results
                if view_img:
                    # self.detOut.emit(im0)
                    self.queue.put(im0)
                    # if self.queue.qsize() > 3:
                    self.qmut_1.unlock()
                    if self.queue.qsize() > 1:

                        self.queue.get(False)
                        self.queue.task_done()
                    else:
                        self.msleep(30)

        print('Done. (%.3fs)' % (time.time() - self.t0))

    def run(self):

        print("self.Consuming-->" + str(self.Consuming))
        if self.Consuming:
            try:
                print("Begin Detecting!!!!")
                with torch.no_grad():
                    self.detect(self.opts())
            except StopIteration:
                print("SOPT ITERATION!!!!!")
            finally:
                # print("Clear The Queue!!")
                # 不清理队列的时候反倒是不会有Bug....
                # 也不知道这个的原理是什么
                # 在这里mark一下，以后有机会了学习学习.....
                # Process finished with exit code 139 (interrupted by signal 11: SIGSEGV)
                # This Bug again..
                while True:
                    print("Clear The Queue!!")
                    if not self.queue.empty():
                        self.queue.get(False)
                        # self.msleep(30)
                        self.queue.task_done()
                    else:
                        break

                print('Done. (%.3fs)' % (time.time() - self.t0))
                print("Stop!!!!!!!!!!!!")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mc = MyClass()
    sys.exit(app.exec_())
