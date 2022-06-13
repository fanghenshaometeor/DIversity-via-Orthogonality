# DIversity-via-Orthogonality (DIO)

PyTorch implementation of the paper [*Towards Robust Neural Networks Via Orthogonal Diversity*](https://arxiv.org/abs/2010.12190).

If our work is helpful for your research, please consider citing:

```
@article{fang2020towards,
  title={Towards Robust Neural Networks via Orthogonal Diversity},
  author={Fang, Kun and Tao, Qinghua and Wu, Yingwen and Li, Tao and Cai, Jia and Cai, Feipeng and Huang, Xiaolin and Yang, Jie},
  journal={arXiv preprint arXiv:2010.12190},
  year={2020}
}
```

## Table of Content
  - [1. Introduction](#1introduction)
  - [2. Requisite](#2requisite)
  - [3. Training and attack](#3training-and-attack)
  - [4. DIO and other defenses](#4dio-and-other-defenses)

## 1. Introduction


## 2. Requisite

A brief description for the files is listed below:
- `train.sh\py` training scripts
- `attack_dio.sh\py` attack scripts 
- `adapt_attack*` adaptive attack scripts
- `model/dio_*.py` DIO model definitions

Dependencies mainly include:
- Python (miniconda)
- PyTorch
- [AdverTorch](https://github.com/BorealisAI/advertorch)
- [AutoAttack](https://github.com/fra31/auto-attack)

For more specific dependency, please refer to the environment.yml.

## 3. Training and attack

### Training

```
sh train.sh
```

Detailed training settings (model, data set and whether to perform adversarial training) and hyper-parameters ($\alpha,\beta,\tau$ and $L$) have been specified in the `train.sh` script.

### Attack

```
sh attack_dio.sh
```

### Adaptive attack

```
sh adapt_attack.sh
```

## 4. DIO and other defenses

DIO is a **model-augmented** adversarial defense and could cooperate with other **data-augmented** defenses together to even boost the adversarial robustness.

In this work, several representative data-augmented defenses are considered:
- PGD-based adversarial training (AT)
- TRADES: [ICML'19 paper](http://proceedings.mlr.press/v97/zhang19p/zhang19p.pdf), [codes](https://github.com/yaodongyu/TRADES/)
- AWP: [NeurIPS'20 paper](https://proceedings.neurips.cc/paper/2020/file/1ef91c212e30e14bf125e9374262401f-Paper.pdf), [codes](https://github.com/csdongxian/AWP)
- LBGAT: [CVPR'21 paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Cui_Learnable_Boundary_Guided_Adversarial_Training_ICCV_2021_paper.pdf), [codes](https://github.com/dvlab-research/LBGAT)
- GAIRAT: [ICLR'20 paper](https://arxiv.org/pdf/2010.01736.pdf), [codes](https://github.com/zjfheart/Geometry-aware-Instance-reweighted-Adversarial-Training)

We reproduce these defenses based on their source codes and equip DIO with these carefully-designed data augmentation techniques. The training and attack codes are also provided in the corresponding folders in this repo:
- [DIO+TRADES](./DIO+TRADES/)
- [DIO+AWP](./DIO+AWP/)
- [DIO+LBGAT](./DIO+LBGAT/)
- [DIO+GAIRAT](./DIO+GAIRAT/)

Run
```
sh train_*_dio.sh
```
and 
```
sh attack_*_dio.sh
```
in these folders to train and attack the corresponding equipped DIO models respectively.

A complete list of the chosen hyper-parameters for different models could be found in the Table 4 in the appendix of the paper.

## 

If u have problems about the codes or paper, u could contact me (fanghenshao@sjtu.edu.cn) or raise issues here.

If u find the codes useful, welcome to fork and star this repo and cite our paper! :)

Lots of thanks from the REAL DIO!!!

![avatar](./pics/REAL-DIO.png)

<!-- # Dependencies
- python 3.6 (miniconda)
- PyTorch 1.5.0

# File Descriptions

- `train.sh,.py` training scripts for OMP model
- `train_ablation.sh,py` ablation training scripts for OMP model
- `test.sh,.py` test scripts for OMP model
- `white_attack_1,2,3.sh,.py` white-box attack scripts for OMP model
- `black_attack.sh,.py` black-box attack scripts for OMP model

# Usage

We provide trained model files in the `./save/` directory. Users could directly check the performance of these models.

## training

To reproduce the training, users can run the `train.sh` shell scripts directly on the command line.
```
sh train.sh
```

## test

To test the performance of each path in an OMP model, users can run the `test.sh` shell scripts directly on the command line.
```
sh test.sh
```

## attack

To evaluate the robustness of OMP model, users can run the attack scripts directly on the command line. Detailed descriptions of every attack script are listed as follows.
- **white_attack_1** performs white-box FGSM and PGD attacks on **EACH** path in an OMP model. In this setting, each path in the OMP model is viewed as a single network, and we evaluate the robustness of these individual networks.
- **white_attack_2** performs white-box FGSM and PGD attacks on the **SELECTED** path in an OMP model. The resulting adversarial examples are then reclassified by **OTHER** paths in the OMP model. In this setting, we evaluate the transferability of the adversarial examples among different paths in an OMP model. 
- **white_attack_3** performs white-box FGSM and PGD attacks on **ALL** the paths in an OMP model. The resulting adversarial examples are then reclassified by **OTHER** paths in the OMP model. In this setting, we evaluate the robustness of each path by simultaneously attacking all the paths.
- **black_attack** performs white-box FGSM and PGD attacks on vanilla-trained networks. The resulting adversarial examples are then reclassified by each path in an OMP model. In this setting, we evaluate the robustness of OMP model against black-box attacks.

## 

If u have problems about the codes or paper, u could contact me (fanghenshao@sjtu.edu.cn) or raise issues in GitHub.

If u find the codes useful, welcome to fork and star this repo and cite our paper! :)

```
@ARTICLE{2020arXiv201012190F,
       author = {{Fang}, Kun and {Wu}, Yingwen and {Li}, Tao and {Huang}, Xiaolin and
         {Yang}, Jie},
        title = "{Learn Robust Features via Orthogonal Multi-Path}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Computer Vision and Pattern Recognition, Computer Science - Machine Learning},
         year = 2020,
        month = oct,
          eid = {arXiv:2010.12190},
        pages = {arXiv:2010.12190},
archivePrefix = {arXiv},
       eprint = {2010.12190},
 primaryClass = {cs.CV},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020arXiv201012190F},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
``` -->