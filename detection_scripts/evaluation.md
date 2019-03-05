
```bash
legacy_dir='/mnt/0F22134B0F22134B/GITHUB/Tensorflow_Object_Detection/models/research/object_detection/legacy'
python3 $legacy_dir/eval.py --help
python3 $legacy_dir/eval.py \
        --logtostderr \
        --pipeline_config_path=/mnt/3E0CAB3B0CAAECD9/NMS/PROGRAMACION_NMS/TENSORFLOW-OD/config_files/ssd_mobilenet_v1_coco.config \
        --checkpoint_dir=/mnt/3E0CAB3B0CAAECD9/NMS/PROGRAMACION_NMS/TENSORFLOW-OD/trained_models/fine_tuned_model_20190301-214708 \
        --eval_dir=/mnt/3E0CAB3B0CAAECD9/NMS/PROGRAMACION_NMS/TENSORFLOW-OD/trained_models/fine_tuned_model_20190301-214708/eval
```
