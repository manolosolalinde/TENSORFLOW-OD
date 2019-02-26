import os
import numpy as np
import sys
'''
requires an output_data folder with images .jpg and corresponding .xml files
'''
# sys.path.insert(0, '/mnt/3E0CAB3B0CAAECD9/NMS/PROGRAMACION\ NMS/TENSORFLOW-OD/training_demo/scripts/tf_preprocessing')
sys.path.append("/mnt/3E0CAB3B0CAAECD9/NMS/PROGRAMACION_NMS/TENSORFLOW-OD/training_demo/scripts/preprocessing/")
import split_data

initial_path = os.getcwd()
assert(os.path.isdir(initial_path+'/output_data'))
os.chdir(initial_path+'/output_data')
os.system("mkdir images")
os.system("mkdir annotations")
os.system("mv *.jpg images/")
os.system("mv *.xml images/")
os.chdir(initial_path+'/output_data/images')

split_data.split_data()

os.chdir(initial_path)

os.system("cd output_data/; zip -r data_g3.zip annotations images")

