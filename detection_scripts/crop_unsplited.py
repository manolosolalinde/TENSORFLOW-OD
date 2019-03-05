import argparse
import glob
import os
import random
import xml.etree.ElementTree as ET

import cv2
import pandas as pd
from lxml import etree as et


def save_xml(pd_output,xml_filename,outputshape):
    # prefix, pd_output, category_index,outputshape,cropedshape):
    annotation = et.Element('annotation')
    filename = et.SubElement(annotation, 'filename')
    size = et.SubElement(annotation, 'size')
    width = et.SubElement(size, 'width')
    height = et.SubElement(size, 'height')
    depth = et.SubElement(size, 'depth')
    filename.text = pd_output['filename']
    width.text = str(outputshape[0])
    height.text = str(outputshape[1])
    depth.text = '3'

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

    name.text = str(pd_output['class'])
    pose.text = "Unspecified"
    truncated.text = "0"
    difficult.text = "0"
    xmin.text = str(int(pd_output['xmin']*outputshape[0]))
    ymin.text = str(int(pd_output['ymin']*outputshape[1]))
    xmax.text = str(int(pd_output['xmax']*outputshape[0]))
    ymax.text = str(int(pd_output['ymax']*outputshape[1]))

    # print(et.tostring(annotation, pretty_print=True).decode('utf-8'))
    with open(xml_filename, "w") as text_file:
        text_file.write(et.tostring(
            annotation, pretty_print=True).decode('utf-8'))


def main(inputdir = './unsplited_data',
    outputpath = './cropped_unsplited_data',
    class_name = 'sports ball',
    cropedshape = tuple([80, 80]),
    outputshape = tuple([300, 300]),
    resize = False):

    try:
      import google.colab
      IN_COLAB = True
    except:
      IN_COLAB = False
    import argparse
    parser = argparse.ArgumentParser(
        description="Crop images using xml bounding boxes")
    parser.add_argument("-i",
                        "--inputdir",
                        help="Path to input dir ex: -i ./unsplited_data (must contain images dir)",
                        type=str)
    parser.add_argument("-o",
                        "--outputpath",
                        help="Path to output dir ex: -o ./cropped_unsplited_data",
                        type=str)
    parser.add_argument("-os",
                        "--outputshape",
                        nargs='+',
                        help="Shape of resized output image in pixels ex: -os 300 300", type=int)
    parser.add_argument("-cs",
                        "--cropedshape",
                        nargs='+',
                        help="Shape of croping area ex: -is 80 80", type=int)
    args = parser.parse_args()

    if(args.inputdir is not None):
        inputdir = args.inputdir
    else:
        print("Inputfile argument is blank")
    if(args.outputpath is not None):
        outputpath = args.outputpath
    else:
        print("modelpath argument is blank")
    if(args.outputshape is not None):
        assert(len(args.outputshape) == 2)
        outputshape = tuple(args.outputshape)
        resize = True
    else:
        resize = False
        outputshape=cropedshape
    if(args.cropedshape is not None):
        assert(len(cropedshape) == 2)
        cropedshape = tuple(args.cropedshape)
    
    print("using inputdir: {}".format(inputdir))
    print("outputpath: {}".format(outputpath))
    print("outputshape: {}x{}".format(outputshape[0], outputshape[1]))
    print("cropedshape: {}x{}".format(cropedshape[0], cropedshape[1]))

    assert(os.path.isdir(inputdir))
    assert(os.path.isdir(inputdir + '/images'))
    assert(len(outputshape) == 2)
    assert(len(cropedshape) == 2)

    # setup output path
    os.system('mkdir {}'.format(outputpath))
    os.system('rm -r {}/*.*'.format(outputpath))

    xml_list = []
    for xml_file in glob.glob(inputdir + '/images/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                    int(root.find('size')[0].text),
                    int(root.find('size')[1].text),
                    member[0].text,
                    int(member[4][0].text),
                    int(member[4][1].text),
                    int(member[4][2].text),
                    int(member[4][3].text)
                    )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height',
                'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)

    # Get the values from csv file
    images_list = xml_df['filename'].values
    class_list = xml_df['class'].values
    x_min_list = xml_df['xmin'].values
    x_max_list = xml_df['xmax'].values
    y_min_list = xml_df['ymin'].values
    y_max_list = xml_df['ymax'].values

    for i in range(0,len(images_list)):
        pd_output = xml_df.iloc[i]  # show only 1 result
        filename = inputdir + '/images/' + images_list[i]
        class_name_aux = images_list[i]
        xmin = x_min_list[i]
        xmax = x_max_list[i]
        ymin = y_min_list[i]
        ymax = y_max_list[i]
        img = cv2.imread(filename)
        xavg = (xmin+xmax)/2
        yavg = (ymin+ymax)/2
        max_x_offset = (cropedshape[0]-(xmax-xmin)) # [max_x_offset+(detection_width)]
        max_y_offset = (cropedshape[1]-(ymax-ymin)) # [max_y_offset+(detection_height)]
        x_offset = max_x_offset*random.random()
        y_offset = max_y_offset*random.random()
        if max_y_offset<0 or max_x_offset<0:
            print("Error with {}. detection is larger than cropshape.".format(filename))
            continue

        xmin_crop = int(round(xmin-x_offset))
        xmax_crop = xmin_crop + cropedshape[0]
        ymin_crop = int(round(ymin-y_offset))
        ymax_crop = ymin_crop + cropedshape[1]

        #check limits
        if xmin_crop<0:
            xmax_crop-=xmin_crop
            xmin_crop=0
        if ymin_crop<0:
            ymax_crop-=ymin_crop
            ymin_crop=0
        if ymax_crop>img.shape[0]:
            ymin_crop-=ymax_crop-img.shape[0]
            ymax_crop=img.shape[0]
        if xmax_crop>img.shape[1]:
            xmin_crop-=xmax_crop-img.shape[1]
            xmax_crop=img.shape[1]

        #new detection box
        newbox_xmin = xmin-xmin_crop
        newbox_ymin = ymin-ymin_crop
        newbox_xmax = xmax-xmin_crop
        newbox_ymax = ymax-ymin_crop

        pd_output['xmin']=newbox_xmin/cropedshape[0]
        pd_output['ymin']=newbox_ymin/cropedshape[1]
        pd_output['xmax']=newbox_xmax/cropedshape[0]
        pd_output['ymax']=newbox_ymax/cropedshape[1]

        #write xml file
        prefix, file_extension = os.path.splitext(images_list[i])
        xml_filename = outputpath + '/' + prefix + '.xml'
        img_filename = outputpath + '/' + prefix + '.jpg'
        save_xml(pd_output,xml_filename,outputshape)

        # crop image
        crop_img = img[ymin_crop:ymin_crop+cropedshape[1], xmin_crop:xmin_crop+cropedshape[0]]  # new image is 80x80
        if resize is True:
            crop_img = cv2.resize(crop_img,outputshape)
        cv2.imwrite(img_filename, crop_img)
        





if __name__ == '__main__':
    main()
