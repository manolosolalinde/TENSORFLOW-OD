import os
import glob
import pandas as pd
import argparse
import xml.etree.ElementTree as ET

#visualization imports
# import sys
# sys.path.append('/mnt/0F22134B0F22134B/GITHUB/Tensorflow Object Detection/models/research/slim')

#configuracion inicial
initial_path = os.getcwd()
assert(os.path.isdir(initial_path+'/output_data'))
os.chdir(initial_path+'/output_data')
os.system('mkdir visualization')

#def get xml_df
path = '.' #current folder

xml_list = []
for xml_file in glob.glob(path + '/*.xml'):
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
x_min_list = xml_df['xmin'].values
x_max_list = xml_df['xmax'].values
y_min_list = xml_df['ymin'].values
y_max_list = xml_df['ymax'].values

import matplotlib.pyplot as plt
from matplotlib import patches, patheffects
import cv2
import numpy as np

def draw_outline(o, lw):
    o.set_path_effects([patheffects.Stroke(
        linewidth=lw, foreground='black'), patheffects.Normal()])
    
def draw_rect(ax, b):
    patch = ax.add_patch(patches.Rectangle(b[:2], *b[-2:], fill=False, edgecolor='red', lw=0.5))
    draw_outline(patch, 0.5)

def draw_text(ax, xy, txt, sz=14):
    text = ax.text(*xy, txt,
        verticalalignment='top', color='red', fontsize=sz, weight='bold')
    draw_outline(text, 1)

# From bounding box (bb) to height weight VOC
# (bb) ymin, xmin, ymax, xmax -> (hw-VOC) xmin, ymax, height, width
def bb_hw(a): return np.array([a[1],a[0],a[3]-a[1]+1,a[2]-a[0]+1])

def show_img(im, figsize=None, ax=None):
    if not ax: fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return ax

def open_image(fn):
    """ Opens an image using OpenCV given the file path.
    Arguments:
        fn: the file path of the image
    Returns:
        The image in RGB format as numpy array of floats normalized to range between 0.0 - 1.0
    """
    flags = cv2.IMREAD_UNCHANGED+cv2.IMREAD_ANYDEPTH+cv2.IMREAD_ANYCOLOR
    if not os.path.exists(fn):
        raise OSError('No such file or directory: {}'.format(fn))
    elif os.path.isdir(fn):
        raise OSError('Is a directory: {}'.format(fn))
    else:
        #res = np.array(Image.open(fn), dtype=np.float32)/255
        #if len(res.shape)==2: res = np.repeat(res[...,None],3,2)
        #return res
        try:
            im = cv2.imread(str(fn), flags).astype(np.float32)/255
            if im is None: raise OSError(f'File not recognized by opencv: {fn}')
            return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise OSError('Error handling image at: {}'.format(fn)) from e

sample_n = 5
for i in range(0,len(images_list)-1):
    sample_n = i+1
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    image_path = os.path.join(path,images_list[sample_n-1])
    ymin, xmin, ymax, xmax = y_min_list[sample_n-1], x_min_list[sample_n-1], y_max_list[sample_n-1], x_max_list[sample_n-1]
    im = open_image(image_path)
    ax.set_title(image_path)
    b = bb_hw((ymin, xmin, ymax, xmax))
    draw_rect(ax, b)
    draw_text(ax, [xmin-30, ymin-30], 'ball')
    

    fig.add_axes(ax)
    ax.imshow(im)

    # plt.setp(plt.gca(), frame_on=False)
    plt.savefig('visualization/'+images_list[sample_n-1],format='jpg')
    # plt.show()









