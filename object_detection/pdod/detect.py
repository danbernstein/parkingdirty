from pdod import analyses, setupMods, visualize, imports

def run_model(model, pattern, threshold, n):
  pattern = pattern
  subset_data(pattern)
  polygon = get_polygon(pattern)

  if model == "yolo":
    print('setting up: ' + model)
    net, category_index = set_up_model_yolo('yolo3_darknet53_voc')

    process_images_yolo(
                 model,
                 net, 
                 'object_detection/input_imgs_subset_cam' + str(pattern), # path to subdirectory of images
                 'object_detection/output_imgs', # where to put output images, if visualization is included
                 threshold,  # threshold for classification
                 n, # number of images to process from each folder
                 polygon,
                 category_index)
  else:
    print('setting up: ' + model)
    detection_graph, label_map, categories, category_index = set_up_model(model)

    ## run the detection and classification processing
    ## args: detection_graph from set_up_model(), the input dir, output dir, threshold for obstacle detection, and number of images to process
    ## get lane polygon from https://www.image-map.net/

    process_images(detection_graph, 
                   'object_detection/input_imgs_subset_cam' + str(pattern), # path to subdirectory of images
                   'object_detection/output_imgs', # where to put output images, if visualization is included
                   threshold,  # threshold for classification
                   n, # number of images to process from each folder
                   polygon,
                   category_index)

    print('done')
    
