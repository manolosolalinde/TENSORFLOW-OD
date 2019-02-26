import tarfile
import six.moves.urllib as urllib
import os

# Download and extract model 
def download_model(DOWNLOAD_BASE,MODEL_FILE,MODEL_DIR):
  opener = urllib.request.URLopener()
  opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
  tar_file = tarfile.open(MODEL_DIR + MODEL_FILE)
  for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
      tar_file.extract(file, MODEL_DIR) #os.getcwd()

def main():
            
    # TODO ALL THIS PART MUST CHANGE TO DOWNLOAD MODEL AND TO INCLUDE ARGUMENT
    # Model initial configuration
    MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
    MODEL_DIR = '../auxiliar/'
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

    download_model(DOWNLOAD_BASE,MODEL_FILE,MODEL_DIR)

if __name__ == '__main__':
    main()
