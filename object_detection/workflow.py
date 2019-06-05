# parking dirty workflow for model deployment

import numpy as np
import requests
from pprint import pprint
import time
import cv2
import pandas as pd
import shapely
import matplotlib.path as mpltPath
import matplotlib.patches as patches
from shapely.geometry import Polygon



# flask app set up

# load pre-trained TensorFlow model on a server using TF Serving and Docker - https://medium.com/@pierrepaci/deploy-tensorflow-object-detection-model-in-less-than-5-minutes-604e6bb0bb04

# need to figure out how to run these commands in python
import subprocess

#!MODEL_URL = 'http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz'
#! git clone https://gist.github.com/adabb12c69cced99d3864d601d033001.git  object-detect
#! cd object-detect
#! docker built -t object-detect --build-arg model_url=MODEL_URL .
#! docker run -p 8080:8080 -p 8081:8081 object-detect


# functions ------------------------------------------------------------------------------
def get_polygon(camera):

  print('getting the polygon for this bike lane')

  d = {'camera': ['cam31', 'cam135','cam68'],
     'polygon': [[(202,144),(213,145),(351,221),(350,240)],
                [(158,278),(126,272),(302,115),(310,116)],
                [(220,140),(241,143),(299,53),(291,52)]]
    }

  df = pd.DataFrame(data=d)

  poly = df.polygon[df.camera == 'cam' + str(camera)].values[0]

  return poly

def process_boxes(box, lane):

  ymin, xmin, ymax, xmax = box

  height, width, channels = image.shape

  center_x = (((xmax * width) - (xmin * width)) / 2) + (xmin * width) # x dimension of image
  center_y = (((ymax * height) - (ymin * height)) / 2) + (ymin * height) # y dimension of image
  points = [(center_x, center_y)]
  # area of the object
  obj_area =  ((xmax * width) - (xmin * width)) * ((ymax * height) - (ymin * height))
  # get the absolute position of the object in the image
  p1 = Polygon([((xmax * width),(ymax * height)), ((xmin * width),(ymax * height)), ((xmin * width),(ymin * height)), ((xmax * width),(ymin * height))])
  # location of the bike lane
  p2 = Polygon(lane)

  # get intersection between object and bike lane
  p3 = p1.intersection(p2)
  # get ratio of overlap to total object area
  overlap = p3.area / obj_area

  return points, overlap # the two values needed to access overlap

def analyze_detection_results(boxes, scores, classes, lane_poly, score_threshold, overlap_threshold, num_objs=0):
    csv_file = 'csvfile.csv'
    f = open(csv_file, 'w')

    boxes = np.squeeze(boxes)
    scores = np.squeeze(scores)
    classes_int = np.squeeze(classes)

    for i in range(boxes.shape[0]):
      if (scores[i] > score_threshold) & (classes_int[i] in {1,3}):
        # need vector of class integers for 'car', 'truck', 'bus', 'motorcycle','person'
        box = tuple(boxes[i].tolist())
        # the box is given as a fraction of the distance in each dimension of the image
        # so we have to multiple it by the image dimensions to get the center of each box, relative to the rest of the image
        points, overlap = process_boxes(box, lane_poly)
        # print(class_name)
        if overlap >= overlap_threshold:
          num_objs += 1

    f.write(#timestamp + ',' +
      str(num_objs) + '\n')

    print(num_objs)

    # return the data table
    return f


# read data from somewhere - either batch or streaming
image = cv2.imread("object-detect/2016-09-16 150825 cam135.png")  # Change dog.jpg with your image

## calculate overlap from boxes and lane polygon
pattern = 135
polygon = get_polygon(pattern)
# pre-process data for input into model
image_np = np.array(image)

payload = {"instances": [image_np.tolist()]}

# send POST requests to the served model
res = requests.post("http://localhost:8080/v1/models/default:predict", json = payload)

# process POST request to classify images as blocked not blocked

classes_int = np.squeeze(res.json()['predictions'][0]['detection_classes'])
boxes = np.squeeze(res.json()['predictions'][0]['detection_boxes'])
scores = np.squeeze(res.json()['predictions'][0]['detection_scores'])

analyze_detection_results(boxes, scores, classes_int, polygon, score_threshold = 0.4, overlap_threshold = 0.4)
