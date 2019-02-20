import os
import sys
import tarfile
import zipfile
from collections import defaultdict
from io import StringIO

import cv2
import numpy as np
import pandas as pd
import six.moves.urllib as urllib
import tensorflow as tf
from lxml import etree as et
from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Download and extract model 
def download_model(DOWNLOAD_BASE,MODEL_FILE):
  opener = urllib.request.URLopener()
  opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
  tar_file = tarfile.open(MODEL_FILE)
  for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
      tar_file.extract(file, os.getcwd())

# Helper code
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def save_xml(prefix,pd_output,category_index):
    annotation=et.Element('annotation')
    img_filename = prefix + '.jpg'
    xml_filename = prefix + '.xml'
    filename = et.SubElement(annotation,'filename')
    size = et.SubElement(annotation,'size')
    width = et.SubElement(size,'width')
    height = et.SubElement(size,'height')
    depth = et.SubElement(size,'depth')
    filename.text = img_filename
    width.text = '800'
    height.text = '600'
    depth.text = '3'

    for row in pd_output.iterrows():
      object1 = et.SubElement(annotation,'object')
      name = et.SubElement(object1,'name')
      pose = et.SubElement(object1,'pose')
      truncated = et.SubElement(object1,'truncated')
      difficult = et.SubElement(object1,'difficult')
      bndbox = et.SubElement(object1,'bndbox')
      xmin = et.SubElement(bndbox,'xmin')
      ymin = et.SubElement(bndbox,'ymin')
      xmax = et.SubElement(bndbox,'xmax')
      ymax = et.SubElement(bndbox,'ymax')
      score = et.SubElement(object1,'score')
      
      clase = int(row[1]['classes'])
      name.text = str(category_index[clase]['name'])
      pose.text = "Unspecified"
      truncated.text = "0"
      difficult.text = "0"
      xmin.text = str(int(row[1]['xmin']*800))
      ymin.text = str(int(row[1]['ymin']*600))
      xmax.text = str(int(row[1]['xmax']*800))
      ymax.text = str(int(row[1]['ymax']*600))
      score.text = str(row[1]['scores'])

    # print(et.tostring(annotation, pretty_print=True).decode('utf-8'))
    with open(xml_filename, "w") as text_file:
        text_file.write(et.tostring(annotation, pretty_print=True).decode('utf-8')) 

def main():

  # Initiate argument parser
  import argparse
  parser = argparse.ArgumentParser(
      description="Save video detections to output folder")
  parser.add_argument("-i",
                      "--inputfile",
                      help="Path to video file",
                      type=str)
  # parser.add_argument("-o",
  #                     "--outputfolder",
  #                     help="output_data if blank", type=str)
  args = parser.parse_args()

  inputfile = "../auxiliar/data/cut2.mp4"
  if(args.inputfile is None):
    print("Inputfile argument is blank")
    print("Using: {}".format(inputfile))
  else:
    inputfile = args.inputfile  

  # TODO ALL THIS PART MUST CHANGE TO DOWNLOAD MODEL AND TO INCLUDE ARGUMENT
  # Model initial configuration
  MODEL_NAME = '../auxiliar/ssd_mobilenet_v1_coco_11_06_2017'
  MODEL_FILE = MODEL_NAME + '.tar.gz'
  DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
  # Path to frozen detection graph. This is the actual model that is used for the object detection.
  PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
  # List of the strings that is used to add correct label for each box.
  PATH_TO_LABELS = os.path.join('../auxiliar/data', 'mscoco_label_map.pbtxt')
  NUM_CLASSES = 90

  #PERFORM DOWNLOAD, UNCOMMENT TO USE
  # download_model(DOWNLOAD_BASE,MODEL_FILE)

  # ## Load a (frozen) Tensorflow model into memory.
  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')

  # ## Loading label map
  # Label maps map indices to category names, so that when our convolution network predicts `5`, 
  # we know that this corresponds to `airplane`.  Here we use internal utility functions, 
  # but anything that returns a dictionary mapping integers to appropriate string labels would be fine
  label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
  categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
  category_index = label_map_util.create_category_index(categories)

  pd_output = pd.DataFrame()
  cap = cv2.VideoCapture(inputfile)
  #eliminar todos los archivos de la carpeta de salida
  os.system("mkdir output_data")
  os.system("rm -R output_data/*")
  counter = 0
  frame_count =100
  with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
      while True:
        ret, image_np = cap.read()
        if ret==0:
              print("End of video")
              break
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)

        # store results in pandas dataframe pd_output
        a = pd.DataFrame(np.squeeze(classes),columns=['classes']) #ymin, xmin, ymax, xmax
        # b = pd.DataFrame(np.squeeze(boxes),columns=['xmin','ymin','xmax','ymax'])
        b = pd.DataFrame(np.squeeze(boxes),columns=['ymin','xmin','ymax','xmax'])
        c = pd.DataFrame(np.squeeze(scores),columns=['scores'])
        pd_aux = pd.concat([a,b,c],axis=1)
        pd_aux = pd_aux[(pd_aux.scores>0.5) & (pd_aux.scores<0.95)]
        pd_aux = pd_aux[pd_aux.classes==37]
        frame_count += 1
        if pd_aux.shape[0]>0 and frame_count>=5: #stores max every 5 frames
              # Guardar info
              # pd_output = pd.concat([pd_output,pd_aux])
              frame_count=0
              counter +=1
              prefix = 'output_data/data_' + str(counter).zfill(3) 
              img_filename = prefix + '.jpg'
              save_xml(prefix,pd_aux,category_index)
              cv2.imwrite(img_filename,cv2.resize(image_np, (800,600)))
        
        cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
        if cv2.waitKey(25) & 0xFF == ord('q'):
          cv2.destroyAllWindows()
          break
  
  # split into train and test values
  split_ratio = 0.9
  os.system("mkdir output_data/images")
  os.system("mkdir output_data/annotations")
  os.system("mkdir output_data/images/train")
  os.system("mkdir output_data/images/test")

  # get number of points
  # DIR = 'detect_n_label/training_demo'
  # nfiles = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])/2
  nfiles = counter
  data_id = np.arange(1,nfiles+1)
  np.random.shuffle(data_id)
  n_train = int(round(split_ratio*nfiles))
  training_id, test_id = data_id[:n_train],data_id[n_train:]
  
  for i in training_id:
    prefix = 'output_data/data_' + str(i).zfill(3)
    xml_file = prefix + '.xml'
    jpg_file = prefix + '.jpg'
    os.system("mv {} output_data/images/train".format(xml_file))
    os.system("mv {} output_data/images/train".format(jpg_file))

  for i in test_id:
    prefix = 'output_data/data_' + str(i).zfill(3)
    xml_file = prefix + '.xml'
    jpg_file = prefix + '.jpg'
    os.system("mv {} output_data/images/test".format(xml_file))
    os.system("mv {} output_data/images/test".format(jpg_file))

  os.system("cd output_data/; zip -r data.zip annotations images")

if __name__ == '__main__':
    main()

# pd.DataFrame(np.squeeze(boxes),columns=['xmin','ymin','xmax','ymax'])
