import tensorflow as tf
import cv2
import numpy as np
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from matplotlib import pyplot as plt
import imageio

def visualize_boxes(image_path, detection_graph, threshold, lane_poly):

  with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
      # configure tf object detection API for boxes, scores, classes, and num of detections

      image_tensor, detection_boxes, detection_scores, detection_classes, num_detections = set_up_detection(sess, detection_graph)
      image_np = cv2.imread(image_path)
     # image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
      IMAGE_SIZE = (12, 8)

#      image_np = load_image_into_numpy_array(image)
 #     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#      image = clahe.apply(image_np)



      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)


        # Actual detection.
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})# Visualization of the results of a detection.

      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          min_score_thresh=threshold,
          use_normalized_coordinates=True,
          line_thickness=2)

      lane = np.array([lane_poly], np.int32)
      overlay = image_np.copy()
      alpha = 0.7
      beta = ( 1.0 - alpha );

      src2 = cv2.fillPoly(image_np, lane, (255, 255, 0))
      frame_out = cv2.addWeighted(overlay, alpha, src2, beta, 0, image_np);


      plt.figure(figsize=IMAGE_SIZE)
      plt.imshow(frame_out)

      imageio.imwrite('object_detection/output_imgs/' + os.path.split(image_path)[1], frame_out) # save csv to a different directory than annotated images



