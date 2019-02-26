import os
import sys
import zipfile
from collections import defaultdict
from io import StringIO

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from lxml import etree as et
from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# Helper code

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def save_xml(prefix, pd_output, category_index,outputshape,cropedshape):
    annotation = et.Element('annotation')
    img_filename = prefix + '.jpg'
    xml_filename = prefix + '.xml'
    filename = et.SubElement(annotation, 'filename')
    size = et.SubElement(annotation, 'size')
    width = et.SubElement(size, 'width')
    height = et.SubElement(size, 'height')
    depth = et.SubElement(size, 'depth')
    filename.text = img_filename
    width.text = str(cropedshape[0])
    height.text = str(cropedshape[1])
    depth.text = '3'

    for row in pd_output.iterrows():
        object1 = et.SubElement(annotation, 'object')
        name = et.SubElement(object1, 'name')
        pose = et.SubElement(object1, 'pose')
        truncated = et.SubElement(object1, 'truncated')
        difficult = et.SubElement(object1, 'difficult')
        bndbox = et.SubElement(object1, 'bndbox')
        xmin = et.SubElement(bndbox, 'xmin')
        ymin = et.SubElement(bndbox, 'ymin')
        xmax = et.SubElement(bndbox, 'xmax')
        ymax = et.SubElement(bndbox, 'ymax')
        score = et.SubElement(object1, 'score')

        clase = int(row[1]['classes'])
        name.text = str(category_index[clase]['name'])
        pose.text = "Unspecified"
        truncated.text = "0"
        difficult.text = "0"
        xmin.text = str(int(row[1]['xmin']*outputshape[0]))
        ymin.text = str(int(row[1]['ymin']*outputshape[1]))
        xmax.text = str(int(row[1]['xmax']*outputshape[0]))
        ymax.text = str(int(row[1]['ymax']*outputshape[1]))
        score.text = str(row[1]['scores'])

    # print(et.tostring(annotation, pretty_print=True).decode('utf-8'))
    with open('output_data/' + xml_filename, "w") as text_file:
        text_file.write(et.tostring(
            annotation, pretty_print=True).decode('utf-8'))


def main():

    try:
      import google.colab
      IN_COLAB = True
    except:
      IN_COLAB = False
    
    # Default values
    # Model initial configuration
    # MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
    # MODEL_NAME = 'fine_tuned_model_20190221-044234'
    modelpath = '../auxiliar/fine_tuned_model_20190226-'
    # PATH_TO_LABELS = os.path.join('../auxiliar/data', 'mscoco_label_map.pbtxt')
    PATH_TO_LABELS = os.path.join('../auxiliar/data', 'ball_label_map.pbtxt')
    NUM_CLASSES = 90
    class_name = 'sports ball' #the script will search for this object
    inputfile = '../auxiliar/data/cut2.mp4'
    cropedshape = tuple([80, 60])
    outputshape = tuple([640, 480])
    crop = False

    crop_offset = 15

    # Initiate argument parser
    import argparse
    parser = argparse.ArgumentParser(
        description="Save video detections to output folder")
    parser.add_argument("-i",
                        "--inputfile",
                        help="Path to video file ex: -i ../auxiliar/data/cut2.mp4",
                        type=str)
    parser.add_argument("-m",
                        "--modelpath",
                        help="Path to model ex: -m ../auxiliar/ssd_mobilenet_v1_coco_11_06_2017",
                        type=str)
    parser.add_argument("-l",
                        "--labels",
                        help="Path to labels ex: -m ../auxiliar/data/mscoco_label_map.pbtxt",
                        type=str)
    parser.add_argument("-os",
                        "--outputshape",
                        nargs='+',
                        help="Shape of output image ex: -os 480 360", type=int)
    parser.add_argument("-cs",
                        "--cropedshape",
                        nargs='+',
                        help="Shape of input image to the graph ex: -is 80 60", type=int)
    # parser.add_argument("-o",
    #                     "--outputfolder",
    #                     help="output_data if blank", type=str)
    args = parser.parse_args()

    if(args.inputfile is not None):
        inputfile = args.inputfile
    else:
        print("Inputfile argument is blank")
    if(args.modelpath is not None):
        modelpath = args.modelpath
    else:
        print("modelpath argument is blank")
    if(args.labels is not None):
        PATH_TO_LABELS = args.labels
    if(args.outputshape is not None):
        assert(len(args.outputshape) == 2)
        outputshape = tuple(args.outputshape)
    if(args.cropedshape is not None):
        assert(len(cropedshape) == 2)
        cropedshape = tuple(args.cropedshape)
        crop=True
    
    print("using inputfile: {}".format(inputfile))
    print("modelpath: {}".format(modelpath))
    print("labels: {}".format(modelpath))
    print("outputshape: {}x{}".format(outputshape[0], outputshape[1]))
    print("cropedshape: {}x{}".format(cropedshape[0], cropedshape[1]))

    PATH_TO_CKPT = modelpath + '/frozen_inference_graph.pb'

    assert(os.path.isfile(inputfile))
    assert(os.path.isdir(modelpath))
    assert(os.path.isfile(PATH_TO_CKPT))
    assert(os.path.isfile(PATH_TO_LABELS))
    assert(len(outputshape) == 2)
    assert(len(cropedshape) == 2)

    # Path to frozen detection graph. This is the actual model that is used for the object detection.

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
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    for cat in category_index:
      if category_index[cat]['name']==class_name:
          class_id=category_index[cat]['id']
          break
    assert(class_id is not None)

    if crop is False:
        cropedshape=outputshape

    pd_output = pd.DataFrame()
    cap = cv2.VideoCapture(inputfile)
    # eliminar todos los archivos de la carpeta de salida
    os.system("mkdir output_data")
    os.system("rm -R output_data/*")
    counter = 0
    frame_count = 100
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            while True:
                ret, image_np = cap.read()
                if ret == 0:
                    print("End of video")
                    break
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                image_tensor = detection_graph.get_tensor_by_name(
                    'image_tensor:0')
                # Each box represents a part of the image where a particular object was detected.
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                # Each score represent how level of confidence for each of the objects.
                # Score is shown on the result image, together with the class label.
                scores = detection_graph.get_tensor_by_name(
                    'detection_scores:0')
                classes = detection_graph.get_tensor_by_name(
                    'detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name(
                    'num_detections:0')
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.ruqn(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # store results in pandas dataframe pd_output
                a = pd.DataFrame(np.squeeze(classes), columns=[
                                 'classes'])  # ymin, xmin, ymax, xmax
                # b = pd.DataFrame(np.squeeze(boxes),columns=['xmin','ymin','xmax','ymax'])
                b = pd.DataFrame(np.squeeze(boxes), columns=[
                                 'ymin', 'xmin', 'ymax', 'xmax'])
                c = pd.DataFrame(np.squeeze(scores), columns=['scores'])
                pd_aux = pd.concat([a, b, c], axis=1)
                pd_aux = pd_aux[pd_aux.classes == class_id]
                pd_aux = pd_aux[(pd_aux.scores > 0.3) & (pd_aux.scores < 0.95)]
                pd_aux = pd_aux.head(1)  # show only 1 result
                pd_aux = pd_aux.reset_index(drop=True)

                frame_count += 1
                image_np = cv2.resize(image_np, (outputshape[0], outputshape[1]))
                if pd_aux.shape[0] > 0 and frame_count >= 5:  # stores max every 5 frames
                    # Guardar info
                    # pd_output = pd.concat([pd_output,pd_aux])
                    frame_count = 0
                    counter += 1
                    # crop image
                    if crop is True:
                      ymin = int(round(pd_aux.ymin[0]*outputshape[1]))
                      xmin = int(round(pd_aux.xmin[0]*outputshape[0]))
                      y = ymin - crop_offset
                      x = xmin - crop_offset
                      if x < 0:
                        x = xmin
                      if y < 0:
                        y = ymin
                      crop_img = image_np[y:y+cropedshape[1], x:x+cropedshape[0]]  # new image is 80x60
                      # translate object information to new coords system
                      pd_aux.ymax[0] = pd_aux.ymax[0] - pd_aux.ymin[0] + (ymin-y)/outputshape[0]
                      pd_aux.xmax[0] = pd_aux.xmax[0] - pd_aux.xmin[0] + (xmin-x)/outputshape[1]
                      if pd_aux.ymax[0]>cropedshape[1]/outputshape[1]:
                          pd_aux.ymax[0]=cropedshape[1]/outputshape[1]
                      if pd_aux.xmax[0]>cropedshape[0]/outputshape[0]:
                          pd_aux.xmax[0]=cropedshape[0]/outputshape[0]
                      pd_aux.ymin[0] = (ymin-y)/outputshape[0]
                      pd_aux.xmin[0] = (xmin-x)/outputshape[1]
                    else:
                          crop_img = image_np

                    prefix = 'data_' + str(counter).zfill(3)
                    img_filename = 'output_data/' + prefix + '.jpg'
                    if crop_img.shape[:2] == cropedshape[::-1] or crop ==False:
                      save_xml(prefix, pd_aux, category_index,outputshape,cropedshape)
                      cv2.imwrite(img_filename, crop_img)
                    

                # Visualization of the results of a detection.
                if not IN_COLAB:
                  vis_util.visualize_boxes_and_labels_on_image_array(
                      image_np,
                      np.squeeze(boxes),
                      np.squeeze(classes).astype(np.int32),
                      np.squeeze(scores),
                      category_index,
                      min_score_thresh=0.3,
                      use_normalized_coordinates=True,
                      line_thickness=2)

                  cv2.imshow('object detection',
                            cv2.resize(image_np, (outputshape[0], outputshape[1])))
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    if not IN_COLAB:
                      cv2.destroyAllWindows()
                    break

if __name__ == '__main__':
    main()

# pd.DataFrame(np.squeeze(boxes),columns=['xmin','ymin','xmax','ymax'])
