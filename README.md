# DIversity-via-Orthogonality (DIO)

PyTorch implementation of the paper *Towards Robust Neural Networks Via Orthogonal Diversity* accepted by Pattern Recognition. [journal](https://doi.org/10.1016/j.patcog.2024.110281), [arxiv](https://arxiv.org/abs/2010.12190).

If our work is helpful for your research, please consider citing:

```
@article{fang2024towards,
  title={Towards robust neural networks via orthogonal diversity},
  author={Fang, Kun and Tao, Qinghua and Wu, Yingwen and Li, Tao and Cai, Jia and Cai, Feipeng and Huang, Xiaolin and Yang, Jie},
  journal={Pattern Recognition},
  pages={110281},
  year={2024},
  publisher={Elsevier}
}
```

## Table of Content
  - [1. DIO in a few words](#1-dio-in-a-few-words)
  - [2. Requisite](#2-requisite)
  - [3. Training and attack](#3-training-and-attack)
  - [4. DIO and other defenses](#4-dio-and-other-defenses)

## 1. DIO in a few words


## 2. Requisite

Dependencies mainly include:
- Python (miniconda)
- PyTorch
- [AdverTorch](https://github.com/BorealisAI/advertorch)
- [AutoAttack](https://github.com/fra31/auto-attack)

For more specific dependency, please refer to the [environment.yml](./environment.yml).

## 3. Training and attack

A brief description for the files is listed below:
- `train.sh\py` training scripts
- `attack_dio.sh\py` attack scripts 
- `adapt_attack*` adaptive attack scripts
- `model/dio_*.py` DIO model definitions

### Training

```
sh train.sh
```

Detailed training settings (model, data set and whether to perform adversarial training) and hyper-parameters ($\alpha,\beta,\tau$ and $L$) have been specified in the `train.sh` script.

A complete list of the chosen hyper-parameters for different models could be found in the Table 4 in the appendix of the paper.

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

## 

If u have problems about the code or paper, u could contact me (fanghenshao@sjtu.edu.cn) or raise issues here.

If u find the code useful, welcome to fork and ⭐ this repo and cite our paper! :)

Lots of thanks from REAL DIO!!!

![avatar](./pics/REAL-DIO.png)
