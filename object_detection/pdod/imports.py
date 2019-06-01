import requests, zipfile, io
import fnmatch, os
import shutil
import pandas as pd
import numpy as np

def download_data(cam):
    if cam == "single":
        zip_address = 'http://parkingdirty.com/BlockedBikeLaneTrainingSingleCam.zip'
    else:
        zip_address = 'http://parkingdirty.com/BlockedBikeLaneTrainingFull.zip'

    r = requests.get(zip_address)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall('object_detection/input_imgs') # extract images from zip to input_imgs folder

    print('data downloaded successfully')
    

def filter_data(dir_blocked, dir_notblocked, pattern):
    pattern = 'cam' + str(pattern)
    pattern = '*' + pattern + '*'

    blocked = fnmatch.filter(os.listdir(dir_blocked), pattern)
    notblocked = fnmatch.filter(os.listdir(dir_notblocked), pattern)

    files = [blocked, notblocked]

    return files


def subset_data(dir_blocked, dir_notblocked, pattern):
    pattern_path = 'object_detection/input_imgs_subset_cam' + str(pattern)
    if not os.path.exists(pattern_path):
    #  shutil.rmtree('object_detection/input_imgs_subset')
      os.makedirs(pattern_path + '/blocked')
      os.makedirs(pattern_path + '/notblocked')

    print('subsetting the data')

    for f in filter_data(dir_blocked, dir_notblocked, pattern)[0]:
        shutil.copy(dir_blocked + '/' + f, pattern_path + '/blocked')
    for f in filter_data(pattern)[1]:
        shutil.copy(dir_notblocked + '/' + f, pattern_path + '/notblocked')    


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


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  
  return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
    
