#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"

cd $PROJECT_DIR

DEVICE_ID='0'
NPZ_ROOT=data/test_data/remote_sense_thermal/eval_Optical-Infrared
NPZ_LIST_PATH=data/test_data/remote_sense_thermal/eval_Optical-Infrared/val_list.txt
OUTPUT_PATH=results/remote_sense_thermal

# ELoFTR pretrained:
CUDA_VISIBLE_DEVICES=$DEVICE_ID python tools/evaluate_datasets.py configs/models/eloftr_model.py --ckpt_path weights/matchanything_eloftr.ckpt --method matchanything_eloftr@-@ransac_affine --imgresize 832 --npe --npz_root $NPZ_ROOT --npz_list_path $NPZ_LIST_PATH --output_path $OUTPUT_PATH

# ROMA pretrained:
CUDA_VISIBLE_DEVICES=$DEVICE_ID python tools/evaluate_datasets.py  configs/models/roma_model.py --ckpt_path weights/matchanything_roma.ckpt --method matchanything_roma@-@ransac_affine --imgresize 832 --npe --npz_root $NPZ_ROOT --npz_list_path $NPZ_LIST_PATH --output_path $OUTPUT_PATH