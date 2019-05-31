import datetime, os
import matplotlib.path as mpltPath
import numpy as np

def process_images(detection_graph, path_images_dir, save_directory, threshold, n, lane_poly, category_index):

  csv_file = 'object_detection/output_csv/csvfile.csv'

  f = open(csv_file, 'w')

  print('starting processing at ' + str(datetime.datetime.now()))
  print("lane polygon: " + str(lane_poly))

  with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:

      # configure tf object detection API for boxes, scores, classes, and num of detections
      image_tensor, detection_boxes, detection_scores, detection_classes, num_detections = set_up_detection(sess, detection_graph)

      # the lane polygon is specific to each camera at a particular point in time
      # it could change if the camera's perspective is changed
      # a more robust solution would automatically identify bike lanes
      # lane points identified with: https://www.image-map.net/
      lane = np.array(lane_poly)
      pathbikelane = mpltPath.Path(lane)

      # loop through the object detection algorithm for each image
      if n == 'all':  
        # used this path join in the for loop to get both the 'blocked' and 'notblocked' folders
        for image_path in [os.path.join(path, name) for path, subdirs, files in os.walk(path_images_dir) for name in files]:
  #       for image_path in [os.path.join(path, name) for path, subdirs, files in os.walk(path_images_dir) for name in files:

          timestamp, img_name, img_labels, boxes, scores, classes, num = analyze_image(image_path, path_images_dir, sess, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections)        

          num_cars_in_bikelane_01, num_cars_in_bikelane_015, num_cars_in_bikelane_02, num_cars_in_bikelane_025, num_cars_in_bikelane_03, num_cars_in_bikelane_035, num_cars_in_bikelane_04, num_cars_in_bikelane_045, num_cars_in_bikelane_05, num_cars_in_bike_lane_contains, num_bikes_in_bike_lane = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0        

    # analyzing the detected objects for which are in the bikelane and converting into a tabular format 
    #      writer = Writer(image_path, width, height)

          analyze_boxes(model, category_index, boxes, scores, classes, lane_poly, pathbikelane, f, threshold, timestamp, img_labels, num_cars_in_bikelane_01, num_cars_in_bikelane_015, 
          num_cars_in_bikelane_02, num_cars_in_bikelane_025, num_cars_in_bikelane_03, num_cars_in_bikelane_035, num_cars_in_bikelane_04, num_cars_in_bikelane_045,
          num_cars_in_bikelane_05, num_cars_in_bike_lane_contains, num_bikes_in_bike_lane)
      else:  
        # used this path join in the for loop to get both the 'blocked' and 'notblocked' folders
        for image_path in [os.path.join(path, name) for path, subdirs, files in os.walk(path_images_dir) for name in files[:n]]:
  #       for image_path in [os.path.join(path, name) for path, subdirs, files in os.walk(path_images_dir) for name in files:

          timestamp, img_name, img_labels, boxes, scores, classes, num = analyze_image(image_path, path_images_dir, sess, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections)        

          num_cars_in_bikelane_01, num_cars_in_bikelane_015, num_cars_in_bikelane_02, num_cars_in_bikelane_025, num_cars_in_bikelane_03, num_cars_in_bikelane_035, num_cars_in_bikelane_04, num_cars_in_bikelane_045, num_cars_in_bikelane_05, num_cars_in_bike_lane_contains, num_bikes_in_bike_lane = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0        

    # analyzing the detected objects for which are in the bikelane and converting into a tabular format 
    #      writer = Writer(image_path, width, height)

          analyze_boxes(model, category_index, boxes, scores, classes, lane_poly, pathbikelane, f, threshold, timestamp, img_labels, num_cars_in_bikelane_01, num_cars_in_bikelane_015, 
          num_cars_in_bikelane_02, num_cars_in_bikelane_025, num_cars_in_bikelane_03, num_cars_in_bikelane_035, num_cars_in_bikelane_04, num_cars_in_bikelane_045,
          num_cars_in_bikelane_05, num_cars_in_bike_lane_contains, num_bikes_in_bike_lane)

          #print("Process Time " + str(time.time() - start_time))
          #scipy.misc.imsave('object_detection/output_imgs/' + os.path.split(image_path)[1], image_np) # save csv to a different directory than annotated images

  f.close()
  print('successfully run at ' + str(datetime.datetime.now()))
  return csv_file

import time
import os
from PIL import Image, ImageOps


def analyze_image(image_path, path_images_dir, sess, image_tensor, detection_boxes, detection_scores, detection_classes, num_detections):

  start_time = time.time()
  timestamp = image_path.split(".png")[0]
  img_name = timestamp.split("/")[-1]


  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  try:
      image = Image.open(image_path)
      image_np = load_image_into_numpy_array(image)
  except IOError:
    print("Issue opening "+ image_path)


  width, height = image.size


  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
  image_np_expanded = np.expand_dims(image_np, axis=0)

  # if the image file name contains "not" then assigned 0, otherwise 1, so 1 is blocked, 0 is notblocked
  if os.path.join(path_images_dir + "/" + image_path).find('not') is not -1:
    img_labels = 0
  else:
    img_labels = 1 

  # Actual detection
  (boxes, scores, classes, num) = sess.run(
      [detection_boxes, detection_scores, detection_classes, num_detections],
      feed_dict={image_tensor: image_np_expanded}) 


  scores = np.squeeze(scores)
  boxes = np.squeeze(boxes)

  return timestamp, img_name, img_labels, boxes, scores, classes, num



def analyze_boxes_yolo(model, category_index, boxes, scores, classes, lane_poly, threshold, timestamp, f, img_labels, num_cars_in_bikelane_01, num_cars_in_bikelane_015, 
  num_cars_in_bikelane_02, num_cars_in_bikelane_025, 
  num_cars_in_bikelane_03, num_cars_in_bikelane_035, 
  num_cars_in_bikelane_04, num_cars_in_bikelane_045,
  num_cars_in_bikelane_05, num_cars_in_bike_lane_contains, 
  num_bikes_in_bike_lane):

  boxes = np.squeeze(boxes)
  scores = np.squeeze(scores)
  classes_int = np.squeeze(classes).astype(np.int32)  

  for i in range(boxes.shape[0]):
     if scores[i] > threshold:
        box = tuple(boxes[i].asnumpy().tolist())

    #    print(lane_poly)
        points, overlap = process_polygons(model, box, lane_poly)

        pathbikelane = mpltPath.Path(lane_poly)  

        if classes_int[i] in {3, 8, 6, 4, 1}:
          if overlap >= 0.1:
              num_cars_in_bikelane_01 += 1
          if overlap >= 0.15:
              num_cars_in_bikelane_015 += 1
          if overlap >= 0.2:
              num_cars_in_bikelane_02 += 1
          if overlap >= 0.25:
              num_cars_in_bikelane_025 += 1
          if overlap >= 0.3:
              num_cars_in_bikelane_03 += 1
          if overlap >= 0.35:
              num_cars_in_bikelane_035 += 1
          if overlap >= 0.4:
              num_cars_in_bikelane_04 += 1
          if overlap >= 0.45:
              num_cars_in_bikelane_045 += 1
          if overlap >= 0.5:
              num_cars_in_bikelane_05 += 1    
          if pathbikelane.contains_points(points):
              num_cars_in_bike_lane_contains +=1  

#     if class_name == 'bicycle':
#       if pathbikelane.contains_points(points):
#           num_bikes_in_bike_lane += 1    

 # print(num_cars_in_bikelane_03)

  f.write(timestamp + ',' + 
          str(num_cars_in_bikelane_01) + ',' +
          str(num_cars_in_bikelane_015) + ',' +
          str(num_cars_in_bikelane_02) + ',' +
          str(num_cars_in_bikelane_025) + ',' +
          str(num_cars_in_bikelane_03) + ',' +
          str(num_cars_in_bikelane_035) + ',' +
          str(num_cars_in_bikelane_04) + ',' +
          str(num_cars_in_bikelane_045) + ',' +
          str(num_cars_in_bikelane_05) + ',' + 
          str(num_cars_in_bike_lane_contains) + ',' + 
          str(num_bikes_in_bike_lane) + ',' + 
          str(img_labels) + '\n')

 #  return the data table
  return f


def process_polygons(model, box, lane):

  ymin, xmin, ymax, xmax = box
  # print(box)  
  # the box is given as a fraction of the distance in each dimension of the image
  # so we have to multiple it by the image dimensions to get the center of each box, relative to the rest of the image
  if model == "yolo":
    center_x = (((xmax) - (xmin)) / 2) + (xmin) # x dimension of image
    center_y = (((ymax) - (ymin)) / 2) + (ymin) # y dimension of image

    points = [(center_x, center_y)]

    # area of the object
    obj_area =  ((xmax) - (xmin)) * ((ymax) - (ymin))

    # get the absolute position of the object in the image
    p1 = Polygon([((xmax),(ymax)), ((xmin),(ymax)), ((xmin),(ymin)), ((xmax),(ymin))])

    # location of the bike lane
    p2 = Polygon(np.array(lane) * 1.777) # THIS RETURNS AN ERROR

  else: 
    center_x = (((xmax * 352) - (xmin * 352)) / 2) + (xmin * 352) # x dimension of image
    center_y = (((ymax * 288) - (ymin * 288)) / 2) + (ymin * 288) # y dimension of image
    points = [(center_x, center_y)]

    # area of the object
    obj_area =  ((xmax * 352) - (xmin * 352)) * ((ymax * 288) - (ymin * 288))

    # get the absolute position of the object in the image
    p1 = Polygon([((xmax * 352),(ymax * 288)), ((xmin * 352),(ymax * 288)), ((xmin * 352),(ymin * 288)), ((xmax * 352),(ymin * 288))])

    # location of the bike lane
    p2 = Polygon(lane)
    #print(lane_poly)

  # get intersection between object and bike lane
  p3 = p1.intersection(p2)
  # get ratio of overlap to total object area
  overlap = p3.area / obj_area  
 # print(overlap)

  return points, overlap # the two values needed to access overlap


def calculate_overlap(points, overlap):
  if overlap >= 0.1:
      num_cars_in_bikelane_01 += 1
  if overlap >= 0.15:
      num_cars_in_bikelane_015 += 1
  if overlap >= 0.2:
      num_cars_in_bikelane_02 += 1
  if overlap >= 0.25:
      num_cars_in_bikelane_025 += 1
  if overlap >= 0.3:
      num_cars_in_bikelane_03 += 1
  if overlap >= 0.35:
      num_cars_in_bikelane_035 += 1
  if overlap >= 0.4:
      num_cars_in_bikelane_04 += 1
  if overlap >= 0.45:
      num_cars_in_bikelane_045 += 1
  if overlap >= 0.5:
      num_cars_in_bikelane_05 += 1    
  if pathbikelane.contains_points(points):
      num_cars_in_bike_lane_contains +=1

  return 


#     if class_name == 'bicycle':
#       if pathbikelane.contains_points(points):
#           num_bikes_in_bike_lane += 1    


def process_images_yolo(model, trained_model, path_images_dir, save_directory, threshold, n, lane_poly, category_index):

  csv_file = 'object_detection/output_csv/csvfile.csv'

  f = open(csv_file, 'w')

  print('starting processing')
  print(datetime.datetime.now())

  print("lane polygon: " + str(lane_poly))

  num_cars_in_bikelane_01, num_cars_in_bikelane_015, num_cars_in_bikelane_02, num_cars_in_bikelane_025, num_cars_in_bikelane_03, num_cars_in_bikelane_035, num_cars_in_bikelane_04, num_cars_in_bikelane_045, num_cars_in_bikelane_05, num_cars_in_bike_lane_contains, num_bikes_in_bike_lane = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0        

  lane = np.array(lane_poly)

  pathbikelane = mpltPath.Path(lane)


  # configure tf object detection API for boxes, scores, classes, and num of detections
 # net = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True)

    # loop through the object detection algorithm for each image
  if n == 'all':  
    # used this path join in the for loop to get both the 'blocked' and 'notblocked' folders
    for image_path in [os.path.join(path, name) for path, subdirs, files in os.walk(path_images_dir) for name in files]:

      timestamp, img_name, img_labels, boxes, scores, classes, width_transform, height_transform = analyze_image_yolo(trained_model, image_path, 'object_detection/input_imgs', lane_poly, threshold)

      # the lane polygon is specific to each camera at a particular point in time
      # it could change if the camera's perspective is changed
      # a more robust solution would automatically identify bike lanes
      # lane points identified with: https://www.image-map.net/
      # analyzing the detected objects for which are in the bikelane and converting into a tabular format 

      analyze_boxes_yolo(model, category_index, boxes, scores, classes, lane_po, threshold, timestamp, img_labels,num_cars_in_bikelane_01, num_cars_in_bikelane_015, 
        num_cars_in_bikelane_02, num_cars_in_bikelane_025, 
        num_cars_in_bikelane_03, num_cars_in_bikelane_035, 
        num_cars_in_bikelane_04, num_cars_in_bikelane_045,
        num_cars_in_bikelane_05, num_cars_in_bike_lane_contains, 
        num_bikes_in_bike_lane) 
  else:  
    # used this path join in the for loop to get both the 'blocked' and 'notblocked' folders
    for image_path in [os.path.join(path, name) for path, subdirs, files in os.walk(path_images_dir) for name in files[:n]]:
      #print(image_path)
      timestamp, img_name, img_labels, boxes, scores, classes, width_transform, height_transform = analyze_image_yolo(trained_model, image_path, 'object_detection/input_imgs', lane_poly, threshold)

      # the lane polygon is specific to each camera at a particular point in time
      # it could change if the camera's perspective is changed
      # a more robust solution would automatically identify bike lanes
      # lane points identified with: https://www.image-map.net/

 # analyzing the detected objects for which are in the bikelane and converting into a tabular format 

      analyze_boxes_yolo(model, category_index, boxes, scores, classes, lane_poly, threshold, timestamp, f, img_labels,num_cars_in_bikelane_01, num_cars_in_bikelane_015, 
        num_cars_in_bikelane_02, num_cars_in_bikelane_025, 
        num_cars_in_bikelane_03, num_cars_in_bikelane_035, 
        num_cars_in_bikelane_04, num_cars_in_bikelane_045,
        num_cars_in_bikelane_05, num_cars_in_bike_lane_contains, 
        num_bikes_in_bike_lane) 

  f.close()
  print('successfully run, completed at ' + str(datetime.datetime.now()))
  return csv_file



def analyze_boxes(model, category_index, boxes, scores, classes, lane_poly, pathbikelane, f, threshold, timestamp, img_labels, num_cars_in_bikelane_01, num_cars_in_bikelane_015, 
        num_cars_in_bikelane_02, num_cars_in_bikelane_025, 
        num_cars_in_bikelane_03, num_cars_in_bikelane_035, 
        num_cars_in_bikelane_04, num_cars_in_bikelane_045,
        num_cars_in_bikelane_05, num_cars_in_bike_lane_contains, 
        num_bikes_in_bike_lane):

        for i in range(boxes.shape[0]):
           if scores[i] > threshold:
              box = tuple(boxes[i].tolist())

              classes_int = np.squeeze(classes).astype(np.int32)

              if classes_int[i] in category_index.keys():
                class_name = category_index[classes_int[i]]['name']

              ymin, xmin, ymax, xmax = box

            #  print(lane_poly)
              # the box is given as a fraction of the distance in each dimension of the image
              # so we have to multiple it by the image dimensions to get the center of each box, relative to the rest of the image
              points, overlap = process_polygons(model, box, lane_poly)

              #print(class_name)
              if class_name in {'car', 'truck', 'bus', 'motorcycle','person'}:
                if overlap >= 0.1:
                    num_cars_in_bikelane_01 += 1
                if overlap >= 0.15:
                    num_cars_in_bikelane_015 += 1
                if overlap >= 0.2:
                    num_cars_in_bikelane_02 += 1
                if overlap >= 0.25:
                    num_cars_in_bikelane_025 += 1
                if overlap >= 0.3:
                    num_cars_in_bikelane_03 += 1
                if overlap >= 0.35:
                    num_cars_in_bikelane_035 += 1
                if overlap >= 0.4:
                    num_cars_in_bikelane_04 += 1
                if overlap >= 0.45:
                    num_cars_in_bikelane_045 += 1
                if overlap >= 0.5:
                    num_cars_in_bikelane_05 += 1    
                if pathbikelane.contains_points(points):
                    num_cars_in_bike_lane_contains +=1

              if class_name == 'bicycle':
                if pathbikelane.contains_points(points):
                    num_bikes_in_bike_lane += 1    


        f.write(timestamp + ',' + 
                str(num_cars_in_bikelane_01) + ',' +
                str(num_cars_in_bikelane_015) + ',' +
                str(num_cars_in_bikelane_02) + ',' +
                str(num_cars_in_bikelane_025) + ',' +
                str(num_cars_in_bikelane_03) + ',' +
                str(num_cars_in_bikelane_035) + ',' +
                str(num_cars_in_bikelane_04) + ',' +
                str(num_cars_in_bikelane_045) + ',' +
                str(num_cars_in_bikelane_05) + ',' + 
                str(num_cars_in_bike_lane_contains) + ',' + 
                str(num_bikes_in_bike_lane) + ',' + 
                str(img_labels) + '\n')

    # return the data table
        return f
        
import subprocess

def get_misclassification(file, n):

  command = 'Rscript'
  path2script = 'parkingdirty/object_detection/R/get_misclassification.R'

  args = [file, n]
  cmd = [command, path2script] + args
  x = subprocess.check_output(cmd, universal_newlines=True)

  print(x)

def plot_classification_by_hour(file):

  command = 'Rscript'
  path2script = 'parkingdirty/object_detection/R/mis_classification_by_time.R'

  args = [file]
  cmd = [command, path2script] + args
  x = subprocess.check_output(cmd, universal_newlines=True)

  print(x)
