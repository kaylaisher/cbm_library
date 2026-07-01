# Concept Bottleneck Models benchmark Library


## *A unified training/evaluation toolkit for LF-CBM and VLG-CBM with ANEC metrics*

 - Built with vllm pre-generated concept sets
 - Configs for different models training
 - Shared final layer training class
 - incroporate ANEC tool


## Table of Contents
- [Setup](#setup)
- [Training](#training)
  - [Train LF-CBM](#train-lf-cbm)
  - [Train VLG-CBM](#train-vlg-cbm)
- [Evaluate with ANEC tool](#evaluate-with-anec-tool)
- [Sources](#sources)
- [Acknowledgement](#acknowledgement)


## Setup

### 1) Clone
```bash
git clone https://github.com/kaylaisher/cbm_library.git
cd cbm_library
```

### 2) Install requirements
```bash
bash requirements.sh
```


## training

### Train LF-CBM
```bash
cd ..
python -m cbm_library.scripts.lf_cbm_train <dataset> \
  --save_dir saved_models \
```


### Train VLG-CBM

```bash
cd ..
python -m cbm_library.scripts.vlg_cbm_train <dataset_name> \
  --annotation_dir /kayla/Annotations \
  --save_dir saved_models
```


## Evaluate with ANEC tool

```bash
cd evaluation/ANEC-evaluator
pip install -e .
get_anec --load_path <path_to_your_data_folder> --output_dir <path_to_save_results>
```


## Sources
 - Label-Free CBM (ICLR 2023) — official code: https://github.com/Trustworthy-ML-Lab/Label-free-CBM
 - VLG-CBM (NeurIPS 2024) — official code & docs: https://github.com/Trustworthy-ML-Lab/VLG-CBM
 - ANEC-evaluator — standalone tool to compute ANEC: https://github.com/windymount/ANEC-evaluator
#### We also build on GLM-SAGA for sparse final-layer training and related tooling.



## Acknowledgement
#### This is a project of International Summer Research Program at UCSD. Developed by PIN-CI HUANG, National Central University. Under the supervision of Prof. Lily Weng and Ph.D Ge Yan. 


