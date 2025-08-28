# Concept Bottleneck Models benchmark Library
## *A unified training/evaluation toolkit for LF-CBM and VLG-CBM with ANEC metrics*

 - Built with vllm pre-generated concept sets
 - Configs for different models training
 - Shared final layer training class
 - incroporate ANEC tool

---

## Table of Contents
- [Setup](#setup)
- [Training](#training)
  - [Train LF-CBM](#train-lf-cbm)
  - [Train VLG-CBM](#train-vlg-cbm)
- [Evaluate (ANEC)](#evaluate-anec)
- [Results (template)](#results-template)
- [Sources / Acknowledgments](#sources--acknowledgments)
- [Citations](#citations)
- [License](#license)

---

## Setup

### 1) Clone
```bash
git clone https://github.com/kaylaisher/cbm_library.git
cd cbm_library
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

### 4) Evaluate with ANEC tool

```bash
cd evaluation/ANEC-evaluator
pip install -e .
get_anec --load_path <path_to_your_data_folder> --output_dir <path_to_save_results>
```

##Sources / Acknowledgments
 - Label-Free CBM (ICLR 2023) — official code: https://github.com/Trustworthy-ML-Lab/Label-free-CBM
 - VLG-CBM (NeurIPS 2024) — official code & docs: https://github.com/Trustworthy-ML-Lab/VLG-CBM
 - ANEC-evaluator — standalone tool to compute ANEC: https://github.com/windymount/ANEC-evaluator
####We also build on GLM-SAGA for sparse final-layer training and related tooling.

##Citations

####If you use this library, please also cite the foundational works:
@inproceedings{oikarinen2023labelfree,
  title     = {Label-free Concept Bottleneck Models},
  author    = {Oikarinen, Tuomas and Das, Subhro and Nguyen, Lam M and Weng, Tsui-Wei},
  booktitle = {International Conference on Learning Representations},
  year      = {2023}
}

@inproceedings{srivastava2024vlg,
  title   = {VLG-CBM: Training Concept Bottleneck Models with Vision-Language Guidance},
  author  = {Srivastava, Divyansh and Yan, Ge and Weng, Tsui-Wei},
  booktitle = {NeurIPS},
  year    = {2024}
}


License

This project is distributed under the license in this repository. Please also respect the licenses of the upstream projects linked above.

