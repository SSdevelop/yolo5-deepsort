## Starting the Flask Backend: 
Run the following in `yolov5-deepsort`:`python web-ui/app.py`
Alternatively, you can also start a local opencv demo (make sure the paths to images and videos are correct):
`python web-ui/backend.py`
When the target person is located, its bounding box turn green:
![target.png](examples%2Ftarget.png)
At the moment when the person losts his or her belonging, "Lost" will
pop up on the bounding box of the suitcase (Currently only suitcase is supported).
![lost.PNG](examples%2Flost.PNG)The suitcase is "lost" when pixel distance between 
the center of the person and the suitcase exceed a threshold. However, this might not be the
perfect criteria and we might need some better distance estimation algorithms.

## video capture
\demo.py\main

## frame of lost 
\tracker.py\plot_bboxes

## where to add face id
\tracker.py\plot_bboxes\image

## where to change suitcase to other items
\tracker.py\demo.py -> item_to_detect