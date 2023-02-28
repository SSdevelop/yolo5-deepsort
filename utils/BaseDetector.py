from tracker import update_tracker
import cv2


class baseDet(object):

    def __init__(self):

        self.img_size = 640
        self.threshold = 0.3
        self.stride = 1
        self.targetTrackId=None

    def build_config(self):

        self.faceTracker = {}
        self.faceClasses = {}
        self.faceLocation1 = {}
        self.faceLocation2 = {}
        self.frameCounter = 0
        self.currentCarID = 0
        self.personAndSuitcaseLostCounter = 0
        self.isLost = False
        self.recorded = []

        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def feedCap(self, im, draw=True):

        retDict = {
            'frame': None,
            'faces': None,
            'list_of_ids': None,
            'face_bboxes': []
        }
        self.frameCounter += 1
        if draw:
            print("drawing")
            im, faces, face_bboxes = update_tracker(self, im, draw, self.targetTrackId)
            # print("face_bboxes",face_bboxes)
            # print("im",im)

            retDict['frame'] = im
            retDict['faces'] = faces
            retDict['face_bboxes'] = face_bboxes

            return retDict
        else:
            bboxes=update_tracker(self, im, draw)
            return bboxes

    def init_model(self):
        raise EOFError("Undefined model type.")

    def preprocess(self):
        raise EOFError("Undefined model type.")

    def detect(self):
        raise EOFError("Undefined model type.")
