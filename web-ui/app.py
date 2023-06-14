import io
from PIL import Image
import base64
import dash
from dash import dcc
from dash import html
import cv2
import tempfile
import imutils
import numpy as np
from dash.dependencies import Input, Output, State
import logging
from AIDetector_pytorch import Detector
from demo import _pdist


def _nn_euclidean_distance(x, y, single_embedding):
    distances = _pdist(x, y, single_embedding)
    return np.maximum(0.0, distances.min(axis=0))
logging_level = logging.INFO
logging.basicConfig(level=logging_level,format='[%(lineno)d]:[%(asctime)s]:%(message)s')
app = dash.Dash(__name__)
#TODO: use dash-bootstrap to format
app.layout = html.Div([
    dcc.Upload(
        id='upload-img',
        children=[html.Div([
            'Drag and Drop or ',
            html.A('Select Images')
        ]),html.Div(id='img-list')],
        multiple=True
    ),
    dcc.Upload(
        id='upload-vid',
        children=[html.Div([
            'Drag and Drop or ',
            html.A('Select Videos')
        ]),html.Div(id='vid-list')],
        multiple=True
    ),

    html.Button('Submit', id='submit-button',n_clicks=0),
    html.Div(id='exec-result')
])




@app.callback(Output('img-list', 'children'),
              Input('upload-img', 'filename'))
def img_list(files):
    if files is None:
        files=[]
    file_elements = [html.Li(filename) for filename in files]
    return html.Ul(file_elements)

@app.callback(Output('vid-list', 'children'),
              Input('upload-vid', 'filename'))
def vid_list(files):
    if files is None:
        files = []
    file_elements = [html.Li(filename) for filename in files]
    return html.Ul(file_elements)

def str_to_cv2(img_content:str):
    img_content=img_content.split('base64,')[1]
    data=np.frombuffer(base64.b64decode(img_content),dtype=np.uint8)
    image=cv2.imdecode(data,cv2.IMREAD_UNCHANGED)
    return image[:,:,0:3]

def str_to_video(vid_content:str):
    vid_content=vid_content.split('base64,')[1]
    data=base64.b64decode(vid_content)
    with tempfile.NamedTemporaryFile() as f:
        f.write(data)
        print(f.name)
        vid_cap=cv2.VideoCapture(f.name)
    return vid_cap
@app.callback(Output('exec-result', 'children'),
              Input('submit-button', 'n_clicks'),
              State('upload-img', 'filename'),
              State('upload-img', 'contents'),
              State('upload-vid', 'filename'),
              State('upload-vid', 'contents'))
def exec_back_end(n_clicks, img_names, img_contents, vid_names, vid_contents):
    #print(f'vids:{vid_names} imgs:{img_names}')
    logging.info(f"exec back end: {n_clicks}")
    det = Detector(['person', 'backpack'])
    #TODO: refactor loadIDFeats to take imgs as faces
    if img_contents is None:
        return "Please upload image of the person who lost belongings"
    cv2_img=[str_to_cv2(img_bytes) for img_bytes in img_contents]
    name_list, known_embedding = det.loadIDFeats(img_names,cv2_img)
    result_list=[]
    logging.info(f"know embedding: {known_embedding}")
    if vid_names is None:
        return "Please upload video footages"
    for index,vid_name in enumerate(vid_names):
        cap=str_to_video(vid_contents[index])
        logging.info(f"VID name: {vid_name} Content {vid_contents[index][:100]}")
        result_list.append(exec_one_video(cap,det,known_embedding))
    logging.info(f"Result List: {result_list}")
    return result_list

#https://stackoverflow.com/questions/73167161/error-when-trying-to-capture-some-frames-from-a-video-using-open-cv-on-windows
def exec_one_video(cap:cv2.VideoCapture,det:Detector,embeds):
    video_writer=cv2.VideoWriter()
    frame_count=0
    target_locked=False
    lost_frame_index=[]
    min_index=0
    conf_index=0
    #track_id=None
    while True:
        success, im = cap.read()
        if im is None:
            logging.info(f"exit after {frame_count} frames")
            break
        if frame_count % 5 == 0:
            logging.info(f'running det at {frame_count}')
            DetFeatures, img_input, box_input = det.loadDetFeats(im)
            result = det.feedCap(im)

            logging.info(f"Result of frame {frame_count}: {result}")
            #current_ids = result['current_ids']
            if len(DetFeatures) > 0 and not target_locked:
                dist_matrix = _nn_euclidean_distance(embeds, DetFeatures, embeds[0])
                minimum = np.min(dist_matrix)
                min_index = dist_matrix.argmin()
                if minimum > 0.12:
                    min_index = -2
                    # print('最小坐标：', minIndex)

                # if track_id is None:
                #     track_id = current_ids[conf_index]
                #     logging.info('trackId:', track_id)
                # det.targetTrackId = track_id
        frame_count=frame_count+1
    logging.info(f"Final Result: {lost_frame_index}")
    return lost_frame_index

if __name__ == '__main__':
    app.run_server(debug=True)