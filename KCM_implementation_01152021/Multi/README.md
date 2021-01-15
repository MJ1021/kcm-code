## Introduction
This supplementary file contains the revised implementation used for the results for a multiclass classification on CIFAR-10 in "Kernel-convoluted Deep Neural Networks with Data Augmentation" accepted to AAAI2021. As we mentioned in the manuscript and the Supplementary material, the code for experiments on CIFAR-10 and CIFAR-100 has its roots in the official code of the Mixup method [1]. Before you implement, please download the folder named 'models' and the python code named 'utils.py' through [1] and save it in this directory. 

## References for the implementation
[1] Zhang, Hongyi, et al. "mixup: Beyond empirical risk minimization." arXiv preprint arXiv:1710.09412 (2017). URL: https://github.com/facebookresearch/mixup-cifar10

## Dependency
argparse                      1.1 
csv                           1.0 
numpy                         1.16.1
torch                         1.0.1
torchvision                   0.2.2
sklearn                       0.20.2

## Directory tree
├── models *
├── README.md 
├── train_rev.py 
├── utils.py *
└── train_kcm.py

* can be downloaded from [1].

## Note
'train_rev.py' is the revised version of train.py from [1]. This python file is used for implementing ERM and MIXUP. 

## Commands to run experiments for a multiclass classification on CIFAR-10 (for training and evaluation)

To run experiments for the best configuration in Table 5 in the Supplementary material, please insert the following command at the directory where train_kcm.py locates:
i) KCM(h=0.01,N=1)
$ CUDA_VISIBLE_DEVICES=0 python3 train_kcm.py --lr 0.1 --seed 20170922 --decay 1e-4 --model ResNet34 --N_kcm 1 --h_kcm 0.01 --alpha 0.0


ii) KCM(h=0.01,N=5)+MIXUP(alpha=1)
$ CUDA_VISIBLE_DEVICES=0 python3 train_kcm.py --lr 0.1 --seed 20170922 --decay 1e-4 --model ResNet34 --N_kcm 5 --h_kcm 0.01

To run experiments for ERM/MIXUP in Table 1 in the manuscript, please insert following command at the directory where train_rev.py locates:
i) ERM 
$ CUDA_VISIBLE_DEVICES=0 python3 train_rev.py --lr 0.1 --seed 20170922 --decay 1e-4 --model ResNet34 --alpha 0.0 

ii) MIXUP(alpha=1)
$ CUDA_VISIBLE_DEVICES=0 python3 train_rev.py --lr 0.1 --seed 20170922 --decay 1e-4 --model ResNet34
