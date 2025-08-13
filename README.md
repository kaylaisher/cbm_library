# CBM Library 
*A unified library for training Concept Bottleneck Models (CBMs) with multiple state-of-the-art methods.*

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-orange.svg)]()
[![Status](https://img.shields.io/badge/status-WIP-yellow.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

>  **Work in Progress**: This library is under active development. Interfaces and features may change without notice.

---

##  Features

- **Unified Framework** – Train different CBM methods with a consistent API  
- **Multiple Methods** – Label-Free CBM, VLG-CBM, LaBo, LM4CV, and CB-LLM  
- **Concept Integration** – Works with our [concept generation module](https://github.com/kaylaisher/CBM-benchmark-project-concept-generation-module)  
- **Unified Training** – Consistent final layer training across all methods  
- **Modular Design** – Easy to extend and customize

---

##  Installation

```bash
git clone https://github.com/kaylaisher/cbm_library.git
cd cbm_library

# (Recommended) create a venv/conda env, then:
pip install -e .
```

## How to use
Use our concept generation module:

```bash
cd concept
python run.py
```

## Roadmap

 - Finalize unified training interfaces
 -  Add full examples for LaBo, LM4CV, CB-LLM
 -  Benchmarks across standard datasets
 -  Documentation website

## Acknowledgements

This project was developed as part of the **UC San Diego International Summer Research Program (ISRP)**  
in **[WengLab]([https://wenglab.org/](https://lilywenglab.github.io/))** under the guidance of **Prof. Tsui-Wei Weng** and **Ph.D. Ge Yan**.

## Citation
If you use this library, please cite:

```bibtex
@inproceedings{oikarinenlabel,
  title={Label-free Concept Bottleneck Models},
  author={Oikarinen, Tuomas and Das, Subhro and Nguyen, Lam M and Weng, Tsui-Wei},
  booktitle={International Conference on Learning Representations},
  year={2023}
}

@inproceedings{srivastava2024vlg,
  title={VLG-CBM: Training Concept Bottleneck Models with Vision-Language Guidance},
  author={Srivastava, Divyansh and Yan, Ge and Weng, Tsui-Wei},
  booktitle={Advances in Neural Information Processing Systems},
  year={2024}
}

@inproceedings{yang2023language,
  title={Language in a bottle: Language model guided concept bottlenecks for interpretable image classification},
  author={Yang, Yue and Panagopoulou, Artemis and Zhou, Shenghao and Jin, Daniel and Callison-Burch, Chris and Yatskar, Mark},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}

@inproceedings{yan2023learning,
  title={Learning Concise and Descriptive Attributes for Visual Recognition},
  author={Yan, An and Wang, Yu and Zhong, Yiwu and Dong, Chengyu and He, Zexue and Lu, Yujie and Wang, William Yang and Shang, Jingbo and McAuley, Julian},
  booktitle={IEEE/CVF International Conference on Computer Vision},
  year={2023}
}
