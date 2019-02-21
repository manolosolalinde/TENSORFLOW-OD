IMAGES="detect_n_label/example_data/datalab_dog_dataset/images"
ANNOTATIONS="detect_n_label/example_data/datalab_dog_dataset/annotations"

# Nota: label debe coincidir con los nombre de imagen, ej dog_001.jpg y con la class en el documento csv

# Create train data:
python3 detect_n_label/generate_tfrecord.py \
    --label=dog \
    --csv_input=$ANNOTATIONS/train_labels.csv \
    --img_path=$IMAGES/train \
    --output_path=$ANNOTATIONS/train.record

# Create eval data:
python3 detect_n_label/generate_tfrecord.py \
    --label=dog --csv_input=$ANNOTATIONS/eval_labels.csv \
    --img_path=$IMAGES/eval \
    --output_path=$ANNOTATIONS/eval.record
