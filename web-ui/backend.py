import logging

import cv2
import imutils
import numpy as np

from AIDetector_pytorch import Detector
from demo import _nn_euclidean_distance
logging_level = logging.INFO
logging.basicConfig(level=logging_level,format='[%(lineno)d]:[%(asctime)s]:%(message)s')

def exec_one_video(cap: cv2.VideoCapture, det: Detector, embeds,visualize=False):
    fps = int(cap.get(5))
    logging.info(f'fps: {fps}')
    frame_count = 0
    conf_index = 0
    trackingcounter = 0
    targetLocked = False
    minIndex = None
    videoWriter = None
    trackId = None
    lost_frame_index = []
    while True:
        success, im = cap.read()
        frame_count = frame_count + 1
        if im is None:
            logging.info(f"Read {frame_count} frames")
            break
        if frame_count % 5 != 0:
            continue
        DetFeatures, img_input, box_input = det.loadDetFeats(im)
        result, lost = det.feedCap(im)
        if lost:
            lost_frame_index.append(frame_count)
        current_ids = result['current_ids']
        logging.info(f"Current ids: {current_ids}")
        if len(DetFeatures) > 0 and not targetLocked:
            dist_matrix = _nn_euclidean_distance(embeds, DetFeatures, embeds[0])
            minimum = np.min(dist_matrix)
            minIndex = dist_matrix.argmin()
            if minimum > 0.5:
                minIndex = -2
        if (minIndex == conf_index) & (minIndex != -2):
            trackingcounter = trackingcounter + 1
            logging.info(f'conf_index: {conf_index}, minIndex: {minIndex}, trackingcounter, {trackingcounter}')
        else:
            conf_index = minIndex
            trackingcounter = 0
        if trackingcounter == 5:
            trackingcounter = 0
            if trackId is None:
                trackId = current_ids[conf_index]
                logging.info(f'trackId: {trackId}')
            det.targetTrackId = trackId
            targetLocked = True
        #save result
        result = result['frame']
        result = imutils.resize(result, height=500)
        # if videoWriter is None:
        #     fourcc = cv2.VideoWriter_fourcc(
        #         'm', 'p', '4', 'v')  # opencv3.0
        #     videoWriter = cv2.VideoWriter(
        #         'result.mp4', fourcc, fps, (result.shape[1], result.shape[0]))
        # videoWriter.write(result)
        frame_count += 1
        if visualize:
            #we display a video using opencv with imshow and set waitket w.r.t fps
            cv2.imshow('video',result)
            cv2.waitKey(int(100/fps))

    cap.release()
    #videoWriter.release()
    if visualize:
        cv2.destroyAllWindows()
    logging.info(f"Lost frames: {lost_frame_index}")
    return lost_frame_index


if __name__ == '__main__':
    det = Detector(['person', 'suitcase'])
    vid_names=['web-ui/IMG_6757.mp4']
    img_names=['images/origin/6757.PNG']
    cv2_img=[cv2.imread(name) for name in img_names]
    cv2_cap=[cv2.VideoCapture(name) for name in vid_names]
    name_list, known_embedding = det.loadIDFeats(img_names, cv2_img)
    result_list = []
    logging.info(f"know embedding: {known_embedding}")
    for index, vid_name in enumerate(vid_names):
        result_list.append(exec_one_video(cv2_cap[index], det, known_embedding,True))
