#Unlike regular web app, this backend is not safe when accesssed from multipe clients
#Only one client should be communicating with the backend at any given time
import json
import logging
import os.path

import cv2
from flask import Flask, request, jsonify,send_file
from backend import exec_one_video
from AIDetector_pytorch import Detector
from progress_monitor import job_monitor
from flask import Blueprint
blueprint = Blueprint('blueprint', __name__)

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
tmp_dir="web-ui/tmp-store"
logging.info(f"Backend Running Dir: {os.getcwd()}")
#https://stackoverflow.com/questions/57233053/chrome-fails-to-load-video-if-transferred-with-status-206-partial-content
@blueprint.route('/files/<filename>',methods=['GET'])
def get_file(filename):
    logging.info(f"Route /files running on: {os.getcwd()}")
    #TODO: refactor tmp file naming
    try:
        response=send_file(os.path.join('web-ui','videos',filename), mimetype='video/mp4')
    except FileNotFoundError:
        return ""
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response
@blueprint.route('/thumbnails/<filename>',methods=['GET'])
def get_thumbnail(filename):
    logging.info(f"Route /files running on: {os.getcwd()}")
    name, suffix = filename.split(".")
    thumbnail_name=f'{name}_thumbnail.jpg'
    thumbnail_path=f"./web-ui/thumbnails/{thumbnail_name}"
    if not os.path.exists(thumbnail_path):
        logging.info(f"Creating thumbnail for {filename}")
        video_path=f"./web-ui/videos/{filename}"
        if os.path.exists(video_path):
            #create thumbnail for the video
            logging.info(f"Video located for {filename}")
            cap=cv2.VideoCapture(video_path)
            success,image=cap.read()
            height, width = image.shape[:2]
            size = min(height, width)
            cropped = image[:size, :size]
            cv2.imwrite(thumbnail_path, cropped)
            cap.release()
            logging.info(f"Thumbnail {thumbnail_name} created")
    response=send_file(f"./thumbnails/{thumbnail_name}", mimetype='image/jpeg')
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
        file_names=json.loads(request.form['file_names'])
        job_monitor.start_process(file_names)
        if num_videos==0 or num_videos==0:
            logging.info('Please at least upload a video and an image')
            return jsonify({'message':'Please at least upload a video and an image'})
        logging.info(f"Images:{num_images} Videos {num_videos}")
        logging.info(f"Files: {request.files.keys()}")
        images_cv2=[]
        images=[]
        video_cv2=[]
        frame_count=[]
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
            vid_cap=cv2.VideoCapture(os.path.join(tmp_dir, video.filename))
            video_cv2.append(vid_cap)
            frame_count.append(int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        job_monitor.set_frames(frame_count)
        logging.info(f'Videos: {videos}')
        det = Detector(['person', 'suitcase'])
        # you can change vid names and img names yourself (relative path from project base dir)
        name_list, known_embedding = det.loadIDFeats(images, images_cv2)
        logging.info(f"know embedding: {known_embedding}")
        for vid_index, vid_name in enumerate(videos):
            if len(known_embedding)==0:
                frame_range=[]
            else:
                frame_range=exec_one_video(video_cv2[vid_index], det,vid_index, known_embedding,vid_name,True)
            name, suffix = vid_name.split('.')
            return_metadata.append(["__RESULT__"+name+'.'+suffix,frame_range])
            video_cv2[vid_index].release()
        # clean up
        for i in range(num_videos):
            video = file_list['video{}'.format(i)]
            os.remove(os.path.join(tmp_dir, video.filename))
        for i in range(num_images):
            image = file_list['image{}'.format(i)]
            os.remove(os.path.join(tmp_dir, image.filename))
        job_monitor.end_process()
        response=jsonify(return_metadata)
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response

    return "Only support POST Method"

# @blueprint.route("/progress", methods=['POST'])
# def progress():
#     if request.method=='POST':
#         logging.debug(f"Progress Queried: {job_monitor.get_progress()}/100")
#         return jsonify({'progress':job_monitor.get_progress()})



@blueprint.route('/progress', methods=['GET'])
def progress():
    return jsonify(job_monitor.get_progress())

app=Flask(__name__)
app.register_blueprint(blueprint)


if __name__ == '__main__':
    app.run(debug=True)