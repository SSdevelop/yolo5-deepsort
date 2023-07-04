import logging
import os.path

from progress_monitor import job_monitor
import cv2
import imutils
import numpy as np
video_result='web-ui/videos'
os.makedirs(video_result,exist_ok=True)
from AIDetector_pytorch import Detector
from demo import _nn_euclidean_distance
logging_level = logging.INFO
logging.basicConfig(level=logging_level,format='[%(lineno)d]:[%(asctime)s]:%(message)s')


#https://stackoverflow.com/questions/30103077/what-is-the-codec-for-mp4-videos-in-python-opencv
def exec_one_video(cap: cv2.VideoCapture, det: Detector,index:int, embeds,vid_name=None,visualize=False):
    fps = int(cap.get(5))
    logging.info(f'fps: {fps}')
    frame_count = 0
    conf_index = 0
    trackingcounter = 0
    targetLocked = False
    minIndex = None
    trackId = None
    lost_frame_index = []
    if vid_name is not None:
        #codec must be avc1 (h.264) to allowing playing in <video> element of html
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        name,suffix=vid_name.split('.')
        video_writer=cv2.VideoWriter(os.path.join(video_result,"__RESULT__"+name+'.'+suffix),fourcc,6,(int(cap.get(3)),int(cap.get(4))))
    lost_counter=20
    while True:
        # if job_monitor.canceled():
        #     return lost_frame_index
        success, im = cap.read()
        frame_count = frame_count + 1
        job_monitor.increase(index)
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

        if lost_counter>0 and lost and vid_name is not None:
            logging.info(f'Writing frame {lost_counter} to output')
            video_writer.write(result)
            lost_counter-=1
        result = imutils.resize(result, height=500)
        if visualize:
            #we display a video using opencv with imshow and set waitket w.r.t fps
            cv2.imshow('video',result)
            cv2.waitKey(int(100/fps))
    if vid_name is not None:
        video_writer.release()
    if visualize:
        cv2.destroyAllWindows()
    logging.info(f"Lost frames: {lost_frame_index}")
    return lost_frame_index


if __name__ == '__main__':
    det = Detector(['person', 'suitcase'])
    #you can change vid names and img names yourself (relative path from project base dir)
    vid_names=['IMG_1752.mp4']
    img_names=['images/origin/1752.PNG']
    cv2_img=[cv2.imread(name) for name in img_names]
    cv2_cap=[cv2.VideoCapture(name) for name in vid_names]
    name_list, known_embedding = det.loadIDFeats(img_names, cv2_img)
    result_list = []
    logging.info(f"know embedding: {known_embedding}")
    for index, vid_name in enumerate(vid_names):
        result_list.append(exec_one_video(cv2_cap[index], det, known_embedding,vid_name,True))
