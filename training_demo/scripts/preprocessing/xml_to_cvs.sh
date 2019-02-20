IMAGES="detect_n_label/example_data/datalab_dog_dataset/images"
ANNOTATIONS="detect_n_label/example_data/datalab_dog_dataset/annotations"

# Create train data:
python3 detect_n_label/xml_to_csv.py -i $IMAGES/train -o $ANNOTATIONS/train_labels.csv

# Create test data:
python3 detect_n_label/xml_to_csv.py -i $IMAGES/test -o $ANNOTATIONS/test_labels.csv
