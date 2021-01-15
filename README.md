### Introduction

Data augmentation (DA) has been widely used to alleviate overfitting issues and robustness to adversarial examples in deep learning. 
In particular, trained deep models using the  Mixup-generated samples (Mixup; Zhang et al. 2018) have demonstrated superb performances in supervised classification.
In this work, we focus on the role of DA. In this view, the Mixup method encourages the models to satisfy the linearity constraint implicitly, which presents the models' smoothness. In this thesis, we build models that explicitly bring desirable constraints of smoothness. 
We propose kernel-convoluted models (KCM) where the smoothness constraint is explicitly imposed by locally averaging all shifted original functions with a kernel function.
Besides, we extend it to incorporate Mixup into KCM. For more details, we refer to our paper (https://arxiv.org/abs/2012.02521v2). 

This repository contains the experiments used for the results in the paper.


### Others

We categorize implementations into a type of problem: binary classification and multi-class classification. 
Experimental details, including requirements, are summarized in each folder. 


### License

Apache License 2.0.
