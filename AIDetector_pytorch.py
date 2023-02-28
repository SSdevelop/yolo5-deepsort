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
class Detector(baseDet):

    def __init__(self,item_to_detect):
        super(Detector, self).__init__()
        self.init_model()
        self.build_config()
        self.item_to_detect = item_to_detect
        

    def init_model(self):

        self.weights = 'weights/yolov5s.pt'
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        self.device = select_device(self.device)

        model = attempt_load(self.weights, map_location=self.device)

        model.to(self.device).eval()

        # model.half()
        model.float()
        # torch.save(model, 'test.pt')
        self.m = model
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

        return img0, img


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
                    lbl = self.names[int(cls_id)]
                    if not lbl in self.item_to_detect:
                        continue
                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])
                    pred_boxes.append(
                        (x1, y1, x2, y2, lbl, conf))

        print(pred_boxes)
        return im, pred_boxes

    def loadIDFeats(self):
        '''
        加载目标人的特征
        '''
       # deepsort = DeepSort()
        # 记录名字和特征值
        name_list = []
        known_embedding = []
        # 输入网络的所有人图片
        known_img_input = []
        known_boxes_input = []
        # 遍历
        known_person_list = glob.glob('./images/origin/*')
        import pytest
        # pytest.set_trace()
        name_id_map = {}
        for person in tqdm.tqdm(known_person_list, desc='处理目标人物...'):
            picture = cv2.imread(person)
            name = person.split('/')[-1].split('.')[0]
            name_list.append(name)
            # pytest.set_trace()
            # 用YOLOv5识别人脸
            im, pred_boxes = self.detect(picture)
            if im is None:
                print('图片：{} 未检测到人，跳过'.format(person))
                continue
            # 预处理
           # img_input = self.imgPreprocess(im)

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

            #known_boxes_input.append(bbox_xywh_person)
            known_boxes_input.append(bbox_xyxy_person)
            known_img_input.append(picture)
            # 得到所有的embedding
        known_embedding = deepsort._get_ID_features(known_boxes_input, known_img_input)  # 此函数需要xywh的格式
            #known_embedding.append(embedding_person)
       # self.height, self.width = img_input.shape[:2]

        return name_list, known_embedding

    def loadDetFeats(self, picture):
        '''
        加载目标人的特征
        '''
    # deepsort = DeepSort()
        # 记录名字和特征值
        name_list = []
        known_embedding = []
        # 输入网络的所有人图片
        known_img_input = []
        known_boxes_input = []
        im, pred_boxes = self.detect(picture)
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
            #known_embedding.append(embedding_person)
    # self.height, self.width = img_input.shape[:2]

        return known_embedding, known_img_input, known_boxes_input