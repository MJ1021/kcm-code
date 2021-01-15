## Introduction
This supplementary file contains the revised implementation used for the results for a binary classification in "Kernel-convoluted Deep Neural Networks with Data Augmentation" accepted to AAAI2021.

## References for the implementation
[1] Miyato, Takeru, et al. "Virtual adversarial training: a regularization method for supervised and semi-supervised learning." IEEE transactions on pattern analysis and machine intelligence 41.8 (2018): 1979-1993. URL: https://github.com/takerum/vat_tf

[2] Zhang, Hongyi, et al. "mixup: Beyond empirical risk minimization." arXiv preprint arXiv:1710.09412 (2017). URL: https://github.com/facebookresearch/mixup-cifar10

## Dependencies
argparse                      1.1 
csv                           1.0 
numpy                         1.16.1
torch                         1.0.1
torchvision                   0.2.2
sklearn                       0.20.2

## Commands to run experiments for a binary classification (for training and evaluation)

To run experiments for the two-moon dataset for the best configuration in Table 3 (KCM(h=0.1,N=1000)) in the Supplementary material, please insert the following command at the directory where main.py locates:
$ CUDA_VISIBLE_DEVICES=0 python3 main.py --dataset twomoon --optimizer adam --base_lr 1e-2 --alpha_list 0.0 --h_list 0.1 --batch_size 64 --patience 500 --max_epoch 500 --M 100 --sampling_proportion_list 1.0 --surrogate_loss hinge


To run experiments for CIFAR-10 for the best configuration in Table 4 in the Supplementary material, please insert the following command at the directory where main.py locates:
i) KCM(h=0.025,N=5)+MIXUP(alpha=0.2)
$ CUDA_VISIBLE_DEVICES=0 python3 main.py --dataset cifar10 --optimizer adam --base_lr 1e-3 --alpha_list 0.2 --h_list 0.025 --batch_size 32 --patience 20 --max_epoch 200 --M 10 --sampling_proportion_list 1.0 --ratio_train_to_val 4.0 --N 5 --N_val 5 --N_test 5 --surrogate_loss hinge 

ii) KCM(h=0.05,N=50)+MIXUP(alpha=0.1)
$ CUDA_VISIBLE_DEVICES=0 python3 main.py --dataset cifar10 --optimizer adam --base_lr 1e-3 --alpha_list 0.1 --h_list 0.05 --batch_size 32 --patience 20 --max_epoch 200 --M 10 --sampling_proportion_list 1.0 --ratio_train_to_val 4.0 --N 50 --N_val 50 --N_test 50 --surrogate_loss hinge 


## Explanations about argments
dataset: the name of dataset. Default: twomoon
optimizer: the name of optimizer. Default: adam
base_lr: the value of initial learning rate. Default: 1e-2
weight_decay: the value of weight decay. Default: 0.0
alpha_list: the list of levels of alpha.
h_list: the list of levels of bandwidth.
batch_size: the size of mini-batch. Default: 64
patience: early stop patience. Default: 20
max_epoch: the number of epochs. Default: 300
M: the number of replications. If the value of seed_num_list is specified, this argument does not have any role. Default: 100
N: the number of samples to approximate kernel-convoluted models. Default: 1000
N_val: the number of samples to approximate kernel-convoluted models in validation set. Default: 1000 (for the Two-moon dataset, default is the same as N)
N_test: the number of samples to approximate kernel-convoluted models in test set. Default: 1000 (for the Two-moon dataset, default is the same as N)
sampling_proportion_list: the list of levels of sampling proportions.
ratio_train_to_val: the ratio of the size of train set to validation set. Default: 2.0
seed_num_list: the list of seed numbers. If the value of seed_num_list is not specified, seed numbers are 0, 1, ..., and M-1.
surrogate_loss: the type of loss function (hinge or logistic). Default: hinge
