# FOI2P

Official implementation of **FOI2P**.

## Introduction

FOI2P is developed based on [CoFiI2P](https://github.com/WHU-USI3DV/CoFiI2P).  
The environment setup and dataset preparation follow the CoFiI2P project.  
Please refer to the original CoFiI2P repository for detailed dependency installation and dataset organization.

## Environment Setup

This project is built upon CoFiI2P.  
Please follow the environment deployment instructions in the CoFiI2P repository:

- CoFiI2P project: https://github.com/WHU-USI3DV/CoFiI2P

The datasets used in this project are also the same as those used in CoFiI2P.  
Please prepare the datasets according to the instructions provided in the CoFiI2P repository.

## Dataset

FOI2P uses the same dataset settings as CoFiI2P.  
Please download and organize the dataset following the official instructions of CoFiI2P.

## Training

To train the model, run:

```bash
python train.py

## Test

To test the model, run:

```bash
python eval_all.py

## Acknowledgements

This project is developed based on the excellent open-source codebases of CoFiI2P and CAST.
We sincerely thank the authors for sharing their valuable work.

@ARTICLE{10685082,
  author={Kang, Shuhao and Liao, Youqi and Li, Jianping and Liang, Fuxun and Li, Yuhao and Zou, Xianghong and Li, Fangning and Chen, Xieyuanli and Dong, Zhen and Yang, Bisheng},
  journal={IEEE Robotics and Automation Letters}, 
  title={CoFiI2P: Coarse-to-Fine Correspondences-Based Image to Point Cloud Registration}, 
  year={2024},
  volume={9},
  number={11},
  pages={10264-10271},
  keywords={Point cloud compression;Feature extraction;Transformers;Cameras;Image resolution;Image edge detection;Detectors;Coarse-to-fine correspondences;image-to-point (I2P) cloud registration;transformer network},
  doi={10.1109/LRA.2024.3466068}
}

@article{cast,
  title={A consistency-aware spot-guided transformer for versatile and hierarchical point cloud registration},
  author={Huang, Renlang and Tang, Yufan and Chen, Jiming and Li, Liang},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={70230--70258},
  year={2024}
}
