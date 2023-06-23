import logging
import os.path

import cv2
from flask import Flask, request, jsonify,send_file
from backend import exec_one_video
from AIDetector_pytorch import Detector
from progress_monitor import job_monitor
from flask import Blueprint
blueprint = Blueprint('blueprint', __name__)

tmp_dir='web-ui/tmp-store'
@blueprint.after_request
def after_request(response):
    header = response.headers
    header['Access-Control-Allow-Origin'] = '*'
    # Other headers can be added here if needed
    return response

#$env:FLASK_APP = "web-ui/wsgi.py"
logging_level = logging.DEBUG
logging.basicConfig(level=logging_level,format='[%(lineno)d]:[%(asctime)s]:%(message)s')
#https://www.javatpoint.com/flask-file-uploading#:~:text=The%20server%2Dside%20flask%20script,saved%20to%20some%20desired%20location.


#https://stackoverflow.com/questions/57233053/chrome-fails-to-load-video-if-transferred-with-status-206-partial-content
@blueprint.route('/results/<filename>',methods=['GET'])
def get_file(filename):
    response=send_file(os.path.join('video-result',filename), mimetype='video/mp4')
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

@blueprint.route("/", methods=['GET'])
def index():
    return "<p>POST request to /upload for inference; GET /progress for inference progress.</p>"
@blueprint.route("/upload", methods=['POST'])
# https://www.geeksforgeeks.org/upload-multiple-files-with-flask/
def inference():
    if request.method == 'POST':
        #key is the name attribute of input element in frontend
        logging.info("POST request received on /upload")
        #files = request.files.getlist('file')
        # Iterate for each file in the files List, and Save them
        num_images,num_videos=int(request.form['num_images']),int(request.form['num_videos'])
        if num_videos==0 or num_videos==0:
            return 'Please upload an image and a video!'
        logging.info(f"Images:{num_images} Videos {num_videos}")
        logging.info(f"Files: {request.files.keys()}")
        images_cv2=[]
        images=[]
        video_cv2=[]
        videos=[]
        return_metadata=[]
        file_list=request.files
        for i in range(num_images):
            image=file_list['image{}'.format(i)]
            images.append(image.filename)
            image.save(os.path.join(tmp_dir,image.filename))
            images_cv2.append(cv2.imread(os.path.join(tmp_dir,image.filename)))
        logging.info(f'Images: {images}')
        for i in range(num_videos):
            video = file_list['video{}'.format(i)]
            videos.append(video.filename)
            video.save(os.path.join(tmp_dir, video.filename))
            video_cv2.append(cv2.VideoCapture(os.path.join(tmp_dir, video.filename)))
        logging.info(f'Videos: {videos}')
        det = Detector(['person', 'suitcase'])
        # you can change vid names and img names yourself (relative path from project base dir)
        name_list, known_embedding = det.loadIDFeats(images, images_cv2)
        logging.info(f"know embedding: {known_embedding}")
        for index, vid_name in enumerate(videos):
            frame_range=exec_one_video(video_cv2[index], det, known_embedding,vid_name,True)
            name, suffix = vid_name.split('.')
            return_metadata.append([name+'_result.'+suffix,frame_range])
            video_cv2[index].release()
        # clean up
        for i in range(num_videos):
            video = file_list['video{}'.format(i)]
            os.remove(os.path.join(tmp_dir, video.filename))
        for i in range(num_images):
            image = file_list['image{}'.format(i)]
            os.remove(os.path.join(tmp_dir, image.filename))
        return jsonify(return_metadata)
    return "Only support POST Method"

@blueprint.route("/progress", methods=['GET'])
def progress():
    logging.debug(f"Progress Queried: {job_monitor.get_progress()}/100")
    return jsonify({'progress':job_monitor.get_progress()})

app=Flask(__name__)
app.register_blueprint(blueprint)


if __name__ == '__main__':
    app.run()