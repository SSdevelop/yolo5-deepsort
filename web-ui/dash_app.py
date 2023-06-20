import io
import os
import uuid

from PIL import Image
import base64
import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html
import cv2
import tempfile
from backend import exec_one_video
import imutils
import numpy as np
from dash.dependencies import Input, Output, State
import logging
from AIDetector_pytorch import Detector
from demo import _pdist
from progress_monitor import job_monitor

def _nn_euclidean_distance(x, y, single_embedding):
    distances = _pdist(x, y, single_embedding)
    return np.maximum(0.0, distances.min(axis=0))
logging_level = logging.INFO
logging.basicConfig(level=logging_level,format='[%(lineno)d]:[%(asctime)s]:%(message)s')
app = dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])
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
    dbc.Progress(id='prog',value=0, style={"height": "30px"}, className="mb-3"),
    dcc.Interval(id='interval',interval=500),
    html.Div(id='exec-result')
])

@app.callback(
    Output('prog','value'),
    Input('interval', 'n_intervals'),
)
def poll(s):
    return job_monitor.get_progress()


# @app.callback(Output('img-list', 'children'),
#               Input('cancel-button', 'filename'))
# def cancel(s):
#     pass



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
    file_name=f'tmp-{uuid.uuid1()}.mp4'
    f=open(file_name,'w+b')
    f.write(data)
    logging.info(f.name)
    vid_cap=cv2.VideoCapture(f.name)
    f.close()
    #os.remove(file_name)
    return vid_cap,file_name


#use dcc Store to triger
@app.callback(
    Output('submit-button','disable'),
    Input('submit-button', 'n_clicks'),
)
def submit_trigger(n_clicks):
    if n_clicks==0:
        return 'first',False
    return 'run', True





#https://stackoverflow.com/questions/74919882/read-video-from-base64-in-python
#TODO: add excution monitor to allow canceling and process bar
#For simplicity, only one process is allowed to enter this
@app.callback(
                Output('exec-result', 'children'),
              State('upload-img', 'filename'),
              State('upload-img', 'contents'),
              State('upload-vid', 'filename'),
              State('upload-vid', 'contents'))
def exec_back_end(start_flag, img_names, img_contents, vid_names, vid_contents):
    default_message="Please upload images and videos"
    if start_flag=='first':
        return default_message
    #print(f'vids:{vid_names} imgs:{img_names}')
    logging.info(f"exec back end: {start_flag}")
    det = Detector(['person', 'suitcase'])
    #TODO: refactor loadIDFeats to take imgs as faces
    if img_contents is None:
        return "Please upload image of the person who lost belongings"
    cv2_img=[str_to_cv2(img_bytes) for img_bytes in img_contents]
    name_list, known_embedding = det.loadIDFeats(img_names,cv2_img)
    result_list=[]
    if vid_names is None:
        return "Please upload video footages"
    for index,vid_name in enumerate(vid_names):
        cap,tmp_name=str_to_video(vid_contents[index])
        logging.info(f"VID name: {vid_name} Content {vid_contents[index][:100]}")
        #set visualization=True in exec_one_video to visualize
        exec_result=exec_one_video(cap,det,known_embedding)
        result_list.append(exec_result)
        cap.release()
        logging.info(f"Deleting temp file {tmp_name}")
        os.remove(tmp_name)
    logging.info(f"Result List: {result_list}")
    return str(result_list), "stop"



#https://stackoverflow.com/questions/73167161/error-when-trying-to-capture-some-frames-from-a-video-using-open-cv-on-windows

if __name__ == '__main__':
    app.run_server(debug=True)