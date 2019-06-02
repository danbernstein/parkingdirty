# parking dirty workflow for model deployment

import numpy
import requests
from pprint import pprint
import time
import cv2

# flask app set up

# load pre-trained TensorFlow model on a server using TF Serving and Docker - https://medium.com/@pierrepaci/deploy-tensorflow-object-detection-model-in-less-than-5-minutes-604e6bb0bb04

! MODEL_URL = 'http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz'
! git clone https://gist.github.com/adabb12c69cced99d3864d601d033001.git  object-detect
! cd object-detect
! docker built -t object-detect --build-arg model_url=MODEL_URL .
! docker run -p 8080:8080 -p 8081:8081 object-detect


# read data from somewhere - either batch or streaming
image = cv2.imread("object-detect/2016-09-16 150825 cam135.png")  # Change dog.jpg with your image

# pre-process data for input into model
image_np = np.array(image)

payload = {"instances": [image_np.tolist()]}

# send POST requests to the served model
res = requests.post("http://localhost:8080/v1/models/default:predict", json = payload)

# process POST request to classify images as blocked not blocked

classes_int = np.squeeze(res.json()['predictions'][0]['detection_classes'])
boxes = np.squeeze(res.json()['predictions'][0]['detection_boxes'])
scores = np.squeeze(res.json()['predictions'][0]['detection_scores'])

## calculate overlap from boxes and lane polygon
## filter boxes based on class, score, and overlap
## produce output