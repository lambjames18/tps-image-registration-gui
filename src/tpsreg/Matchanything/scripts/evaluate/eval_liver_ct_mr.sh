#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"

echo "$SCRIPTPATH"
echo "$PROJECT_DIR"

cd $PROJECT_DIR

DEVICE_ID='0'
NPZ_ROOT=data/test_data/Liver_CT-MR/eval_indexs
NPZ_LIST_PATH=data/test_data/Liver_CT-MR/eval_indexs/val_list.txt
OUTPUT_PATH=results/Liver_CT-MR
DATASET_NAME=Liver_CT-MR

# ELoFTR pretrained:
# CUDA_VISIBLE_DEVICES=$DEVICE_ID python tools/evaluate_datasets.py configs/models/eloftr_model.py --ckpt_path weights/matchanything_eloftr.ckpt --method matchanything_eloftr@-@ransac_affine --imgresize 832 --npe --npz_root $NPZ_ROOT --npz_list_path $NPZ_LIST_PATH --output_path $OUTPUT_PATH --dataset_name $DATASET_NAME

# ROMA pretrained:
CUDA_VISIBLE_DEVICES=$DEVICE_ID python tools/evaluate_datasets_argparse.py  configs/models/roma_model.py --ckpt_path weights/matchanything_roma.ckpt --method matchanything_roma@-@ransac_affine --imgresize 832 --npe --npz_root $NPZ_ROOT --npz_list_path $NPZ_LIST_PATH --output_path $OUTPUT_PATH --dataset_name $DATASET_NAME #--plot_align #--plot_matches