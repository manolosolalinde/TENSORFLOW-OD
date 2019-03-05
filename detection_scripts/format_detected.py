import os
import numpy as np
import sys
'''
requires an output_data folder with images .jpg and corresponding .xml files
'''
# sys.path.insert(0, '/mnt/3E0CAB3B0CAAECD9/NMS/PROGRAMACION\ NMS/TENSORFLOW-OD/training_demo/scripts/tf_preprocessing')
sys.path.append("/mnt/3E0CAB3B0CAAECD9/NMS/PROGRAMACION_NMS/TENSORFLOW-OD/training_demo/scripts/preprocessing/")
import split_data

# impath = 'output_data'
impath = 'cropped_unsplited_data'
zipfilename = 'cropped_data.zip'

initial_path = os.getcwd()
assert(os.path.isdir(impath))
# os.chdir(os.path.join(initial_path,impath))
imdir = os.path.join(impath,'images')
os.system("mkdir "+imdir)
assert(os.path.isdir(imdir))

anndir = os.path.join(impath,'annotations')
os.system("mkdir "+anndir)
assert(os.path.isdir(imdir))

os.system("mv {}/*.jpg ".format(impath)+imdir)
os.system("mv {}/*.xml ".format(impath)+imdir)

os.chdir(imdir)
split_data.split_data()
os.chdir(initial_path)

os.system("cd {}; zip -r {} annotations images".format(impath,zipfilename))

