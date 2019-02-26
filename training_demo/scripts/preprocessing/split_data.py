import os
import numpy as np

# here output_data is where the files are. 1 .xml and corresponding 1 .jpg
def main():
    split_data()

def split_data():
    split_ratio = 0.7
    os.system("mkdir train")
    os.system("mkdir eval")

    list_files = []
    for x in os.listdir():
        if len(x.split('.'))==2: #if its a file
            if x.split('.')[1]=='jpg':
                list_files.append(x.split('.')[0])
        
    im_files = np.array(list_files)
    n_train = int(round(len(im_files)*split_ratio))
    np.random.shuffle(im_files)
    training_im, eval_im = im_files[:n_train],im_files[n_train:]

    for prefix in training_im:
        xml_file = prefix + '.xml'
        jpg_file = prefix + '.jpg'
        os.system("mv {} train/".format(xml_file))
        os.system("mv {} train/".format(jpg_file))
    for prefix in eval_im:
        xml_file = prefix + '.xml'
        jpg_file = prefix + '.jpg'
        os.system("mv {} eval/".format(xml_file))
        os.system("mv {} eval/".format(jpg_file))

if __name__ == '__main__':
    main()
    