import torch, torchvision, matplotlib, random, sys
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset 
from sklearn.datasets import make_circles, make_moons
from sklearn.model_selection import train_test_split
from PIL import Image
import model

def mixup_data(inputs, targets, alpha, use_cuda):    
    
    B = inputs.size()[0]
    if (alpha == 0.0):
        lam = np.zeros(B, dtype = np.float32)
    else:
        lam = np.random.beta(alpha, alpha, size=B).astype(np.float32)
        lam = np.minimum(lam, 1.0-lam)
    
    input_dim_length = len(inputs.size())
    if input_dim_length >= 2:
        shape_tiled_inputs = []; transpose_order = [len(inputs.size())-1]
        for l in range(1, input_dim_length):
            shape_tiled_inputs.append(inputs.size()[l])
            transpose_order.append(l-1)
        shape_tiled_inputs.append(1)
        shape_tiled_inputs[input_dim_length-1] = 1

        lam_inputs = np.tile(lam, shape_tiled_inputs).transpose(transpose_order)
    else:
        lam_inputs = lam
    lam_targets = lam
    lam_inputs, lam_targets = torch.from_numpy(lam_inputs), torch.from_numpy(lam_targets)
    
    if np.sum(lam) != 0:
        index = torch.randperm(B)
    else:
        index = torch.tensor(np.arange(B))
    if use_cuda:
        lam_inputs, lam_targets = lam_inputs.cuda(), lam_targets.cuda()
        index = index.cuda()
    
    mixed_inputs = (1.0-lam_inputs) * inputs + lam_inputs * inputs[index]
    mixed_targets = (1.0-lam_targets) * targets + lam_targets * targets[index]
    
    return mixed_inputs, mixed_targets

class PrepareData_twomoon(Dataset):

    def __init__(self, X, y, use_cuda=True):
        if not torch.is_tensor(X):
            self.X = torch.from_numpy(X)
            if use_cuda:
                self.X = self.X.type(torch.cuda.FloatTensor)
        if not torch.is_tensor(y):
            self.y = torch.from_numpy(y)
            if use_cuda:
                self.y = self.y.type(torch.cuda.FloatTensor)
            
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class PrepareData_cifar10(Dataset):  
    def __init__(self, X, y, transform, use_cuda, seed_num): 
        self.X = X
        self.y = y                                                                     
        self.transform = transform
        self.use_cuda = use_cuda
        self.seed_num = seed_num 
        
        random.seed(self.seed_num)
        
    def __len__(self): 
        return len(self.X)

    def __getitem__(self, idx): 
        img, target = self.X[idx], self.y[idx]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img) 
        target = torch.from_numpy(np.array(target, np.float32))
        return img, target

class LoadAndSplitData():
    def __init__(self, dataset, n_samples, noise, class0, class1, sampling_proportion, ratio_train_to_val, ratio_train_to_test, seed_num):
        self.dataset = dataset
        self.n_samples = n_samples
        self.noise = noise
        self.class0 = class0
        self.class1 = class1
        self.sampling_proportion = sampling_proportion
        self.ratio_train_to_val = ratio_train_to_val
        self.ratio_train_to_test = ratio_train_to_test
        self.seed_num = seed_num

    def load_and_split_data(self):
        x, y = {}, {}
        if self.dataset == 'twomoon':
            x['train'], y['train'], x['val'], y['val'], x['test'], y['test'] = self.load_and_split_twomoon()
        elif self.dataset == 'cifar10':
            x['train'], y['train'], x['val'], y['val'], x['test'], y['test'] = self.load_and_split_cifar10()
        return x['train'], y['train'], x['val'], y['val'], x['test'], y['test']
    
    def load_and_split_twomoon(self):
        if self.sampling_proportion != 1:
            sys.exit("In twomoon dataset, sampling is not required. Please set sampling_proportion be 1.0")
        x, y = {}, {}
        x['whole'], y['whole'] = make_moons(self.n_samples, noise=self.noise) # numpy.ndarray
        y['whole'] = y['whole']*2-1 
        x['whole'] = (x['whole'] - x['whole'].min(axis=0))/(x['whole'].max(axis=0)-x['whole'].min(axis=0))
        
        # Split the whole dataset into train, val, and test.
        ratio_train_to_others = 1.0/(1.0/self.ratio_train_to_val+1.0/self.ratio_train_to_test)
        x['train'], x['val'], y['train'], y['val'] = train_test_split(x['whole'], y['whole'],
                                                                      test_size=1.0/(1.0+ratio_train_to_others),
                                                                      random_state=self.seed_num)
        ratio_val_to_test = self.ratio_train_to_test/self.ratio_train_to_val
        x['val'], x['test'], y['val'], y['test'] = train_test_split(x['val'], y['val'],
                                                                    test_size=1.0/(1.0+ratio_val_to_test),
                                                                    random_state=self.seed_num)
        
        return x['train'], y['train'], x['val'], y['val'], x['test'], y['test']
    
    def load_and_split_cifar10(self):        
        cifar10_trainset = torchvision.datasets.CIFAR10('./data', train=True, download=True)
        cifar10_testset = torchvision.datasets.CIFAR10('./data', train=False, download=True)
        
        classnames = np.array(['airplane', 'automobile', 'bird', 'cat', 'deer',
                               'dog', 'frog', 'horse', 'ship', 'truck'])
        class0_number = np.where(classnames == self.class0)[0][0]
        class1_number = np.where(classnames == self.class1)[0][0]
        
        x, y = {}, {}        
        train_multiclass_inputs = cifar10_trainset.data 
        train_multiclass_targets = np.asarray(cifar10_trainset.targets) 
        
        x['train'], y['train'] = self.multiclass_to_binaryclass(train_multiclass_inputs, train_multiclass_targets, class0_number, class1_number)       
        # Undersample with the given sampling_proportion
        if self.sampling_proportion < 1:
            _, x['train'], _, y['train'] = train_test_split(x['train'], y['train'], test_size = self.sampling_proportion, random_state=self.seed_num)
        
        # Split the train dataset into train and val.
        x['train'], x['val'], y['train'], y['val'] = train_test_split(x['train'], y['train'], test_size=1.0/(1.0+self.ratio_train_to_val), random_state=self.seed_num) 
        
        test_multiclass_inputs = cifar10_testset.data
        test_multiclass_targets = np.asarray(cifar10_testset.targets)
        x['test'], y['test'] = self.multiclass_to_binaryclass(test_multiclass_inputs, test_multiclass_targets,class0_number, class1_number)
        
        return x['train'], y['train'], x['val'], y['val'], x['test'], y['test']
    
    def multiclass_to_binaryclass(self, multiclass_inputs, multiclass_targets, class0_number, class1_number):
        class0_idx, class1_idx = np.where(multiclass_targets==class0_number)[0], np.where(multiclass_targets==class1_number)[0]
        binaryclass_inputs = np.concatenate((multiclass_inputs[class0_idx], multiclass_inputs[class1_idx]), axis = 0)
        binaryclass_targets = np.concatenate((-np.ones(len(class0_idx)), np.ones(len(class1_idx))), axis = 0)

        return binaryclass_inputs, binaryclass_targets
    
class BuildModel():
    
    def __init__(self, dataset, hidden_dims, seed_num):
        self.dataset = dataset
        self.hidden_dims = hidden_dims
        self.seed_num = seed_num
    
    def build_model(self):
        if self.dataset == 'twomoon':
            net = self.build_twomoon_classifier()
        elif self.dataset == 'cifar10':
            net = self.build_cifar10_classifier()
        return net
   
    def build_twomoon_classifier(self):
        return model.twomoon_classifier(self.hidden_dims, self.seed_num)
    
    def build_cifar10_classifier(self):
        return model.cifar10_classifier(self.seed_num)
        
