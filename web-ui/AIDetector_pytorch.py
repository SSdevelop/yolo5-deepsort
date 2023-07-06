import logging
import sys
from deep_sort.deep_sort import DeepSort
import torch
import numpy as np
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.BaseDetector import baseDet
from utils.torch_utils import select_device
from utils.datasets import letterbox
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import torch
import cv2
import re
palette = (2**11 - 1, 2**15 - 1, 2**20 - 1)
cfg = get_config()
cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
import glob
import tqdm
import cv2


deepsort = DeepSort(
    cfg.DEEPSORT.REID_CKPT,
    max_dist=cfg.DEEPSORT.MAX_DIST,
    min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
    max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
    max_age=cfg.DEEPSORT.MAX_AGE,
    n_init=cfg.DEEPSORT.N_INIT,
    nn_budget=cfg.DEEPSORT.NN_BUDGET,
    use_cuda=False,
)

model=None



class Detector(baseDet):

    def __init__(self,item_to_detect):
        super(Detector, self).__init__()
        self.init_model()
        self.build_config()
        self.item_to_detect = item_to_detect
        

    #TODO model should be singleton and only created once
    def init_model(self):
        global model
        self.weights = 'weights/yolov5s.pt'
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        self.device = select_device(self.device)
        if model is None:
            logging.info("Creating model")
            model = attempt_load(self.weights, map_location=self.device)
            model.to(self.device).eval()
            # model.half()
            model.float()
            logging.info('Model created')
        # torch.save(model, 'test.pt')
        self.m = model
        logging.info("Model loaded")
        self.names = model.module.names if hasattr(
            model, 'module') else model.names

    def preprocess(self, img):
        img0 = img.copy()
        img = letterbox(img, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        # img = img.half()
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        logging.info(f"Input image size: {img.shape}")
        return img0, img

#  检测乘客全身照的函数，其实是自带的原函数

    def detect_oringin(self, im):
        logging.info(f"Raw image : {im}")

        im0, img = self.preprocess(im)
        logging.info("Start forward")
        pred = self.m(img, augment=False)[0]
        pred = pred.float()
        pred = non_max_suppression(pred, self.threshold, 0.4)
        pred_boxes = []
        logging.info(f"YOLO result: {pred}")
        for det in pred:

            if det is not None and len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                for *x, conf, cls_id in det:
                    lbl = self.names[int(cls_id)]
                    if not lbl in self.item_to_detect:
                        continue
                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])
                    pred_boxes.append(
                        (x1, y1, x2, y2, lbl, conf))
        return im, pred_boxes
#  原本是自带的原函数，进行更改过后只会检测视频中置信度>0.8的人，降低运算成本

    def detect(self, im):

        im0, img = self.preprocess(im)
        pred = self.m(img, augment=False)[0]
        pred = pred.float()
        pred = non_max_suppression(pred, self.threshold, 0.4)

        pred_boxes = []
        for det in pred:

            if det is not None and len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                for *x, conf, cls_id in det:
                    a = conf.cpu().numpy()
                    lbl = self.names[int(cls_id)]
                    if not lbl in self.item_to_detect or ((lbl == 'person') & (a < 0.75)):
                        continue
                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])
                    pred_boxes.append(
                        (x1, y1, x2, y2, lbl, conf))
        logging.info(pred_boxes)
        return im, pred_boxes

    def loadIDFeats(self,img_name,img_content):
        '''
        加载目标人的特征
        '''
        known_img_input = []
        known_boxes_input = []
        # 遍历
        name_id_map = {}

        for index,person in enumerate(img_name):
            logging.info(f"Person: {person}")
            picture = img_content[index]
            im, pred_boxes = self.detect(picture)
            # 这里要用原始的检测函数，目的是确保每一张全身照的特征都被提取
            if im is None or len(pred_boxes)==0:
                logging.info('No person detected in {}. Skipped'.format(person))
                continue
            x1 = pred_boxes[0][0]
            y1 = pred_boxes[0][1]
            x2 = pred_boxes[0][2]
            y2 = pred_boxes[0][3]
            bbox_xyxy_person = [x1, y1, x2, y2]
            x = int((x1 + x2) / 2.0)
            y = int((y1 + y2) / 2.0)
            w = x2 - x1
            h = y2 - y1
            bbox_xywh_person = [x, y, w, h]
            known_boxes_input.append(bbox_xyxy_person)
            known_img_input.append(picture)
            # 得到所有的embedding
        logging.info(f"known box {known_boxes_input}")
        if len(known_img_input)==0:
            return img_name,[]
        known_embedding = deepsort._get_ID_features(known_boxes_input, known_img_input)  # 此函数需要xywh的格式
        return img_name, known_embedding

    def loadDetFeats(self, picture):
        '''
        加载目标人的特征
        '''

        # 输入网络的所有人图片
        known_img_input = []
        known_boxes_input = []
        im, pred_boxes = self.detect(picture)  # 这里要用更改过的检测函数detect，设置只会检测视频中置信度>0.8的人，降低运算成本
        for person in pred_boxes:
            x1 = person[0]
            y1 = person[1]
            x2 = person[2]
            y2 = person[3]
            bbox_xyxy_person = [x1, y1, x2, y2]
            known_boxes_input.append(bbox_xyxy_person)
            known_img_input.append(im)
            # 得到所有的embedding
        known_embedding = deepsort._get_ID_features(known_boxes_input, known_img_input)  # 此函数需要xywh的格式
        return known_embedding, known_img_input, known_boxes_input