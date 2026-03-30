# MatchAnything: Universal Cross-Modality Image Matching with Large-Scale Pre-Training
### [Project Page](https://zju3dv.github.io/MatchAnything) | [Paper](??)

> MatchAnything: Universal Cross-Modality Image Matching with Large-Scale Pre-Training\
> [Xingyi He](https://hxy-123.github.io/),
[Hao Yu](https://ritianyu.github.io/),
[Sida Peng](https://pengsida.net),
[Dongli Tan](https://github.com/Cuistiano),
[Zehong Shen](https://zehongs.github.io),
[Xiaowei Zhou](https://xzhou.me/),
[Hujun Bao](http://www.cad.zju.edu.cn/home/bao/)<sup>†</sup>\
> Arxiv 2025

<p align="center">
    <img src=docs/teaser_demo.gif alt="animated" />
</p>

## TODO List
- [x] Pre-trained models and inference code
- [x] Huggingface demo
- [ ] Data generation and training code
- [ ] Finetune code to further train on your own data
- [ ] Incorporate more synthetic modalities and image generation methods

## Quick Start

### [<img src="https://s2.loli.net/2024/09/15/aw3rElfQAsOkNCn.png" width="20"> HuggingFace demo for MatchAnything](https://huggingface.co/spaces/LittleFrog/MatchAnything)

## Setup
Create the python environment by:
```
conda env create -f environment.yaml
conda activate env
```
We have tested our code on the device with CUDA 11.7.

Download pretrained weights from [here](https://drive.google.com/file/d/12L3g9-w8rR9K2L4rYaGaDJ7NqX1D713d/view?usp=sharing) and place it under repo directory. Then unzip it by running the following command:
```
unzip weights.zip
rm -rf weights.zip
```

## Test:
We evaluate the models pretrained by our framework using a single network weight on all cross-modality matching and registration tasks.

### Data Preparing
Download the `test_data` directory from [here](https://drive.google.com/drive/folders/1jpxIOcgnQfl9IEPPifdXQ7S7xuj9K4j7?usp=sharing) and plase it under `repo_directory/data`. Then, unzip all datasets by:
```shell
cd repo_directiry/data/test_data

for file in *.zip; do
    unzip "$file" && rm "$file"
done
```

The data structure should looks like:
```
repo_directiry/data/test_data
    - Liver_CT-MR
    - havard_medical_matching
    - remote_sense_thermal
    - MTV_cross_modal_data
    - thermal_visible_ground
    - visible_sar_dataset
    - visible_vectorized_map
```

### Evaluation
```shell
# For Tomography datasets:
sh scripts/evaluate/eval_liver_ct_mr.sh
sh scripts/evaluate/eval_harvard_brain.sh



# For visible-thermal datasets:
sh scripts/evaluate/eval_thermal_remote_sense.sh
sh scripts/evaluate/eval_thermal_mtv.sh
sh scripts/evaluate/eval_thermal_ground.sh

# For visible-sar dataset:
sh scripts/evaluate/eval_visible_sar.sh

# For visible-vectorized map dataset:
sh scripts/evaluate/eval_visible_vectorized_map.sh
```

# Citation

If you find this code useful for your research, please use the following BibTeX entry.

```
@inproceedings{he2025matchanything,
title={MatchAnything: Universal Cross-Modality Image Matching with Large-Scale Pre-Training},
author={He, Xingyi and Yu, Hao and Peng, Sida and Tan, Dongli and Shen, Zehong and Bao, Hujun and Zhou, Xiaowei},
booktitle={Arxiv},
year={2025}
}
```

# Acknowledgement
We thank the authors of
[ELoFTR](https://github.com/zju3dv/EfficientLoFTR),
[ROMA](https://github.com/Parskatt/RoMa) for their great works, without which our project/code would not be possible.