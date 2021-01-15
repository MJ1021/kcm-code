import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class twomoon_classifier(nn.Module):
    def __init__(self, hidden_dims, seed_num):
        super(twomoon_classifier, self).__init__()
        
        torch.manual_seed(seed_num)
        np.random.seed(seed_num)
        
        self.dataset = 'twomoon'
        self.is_imagedata = False
        
        self.L = len(hidden_dims)
        self.layers = []
        self.layers.append(nn.Linear(2, hidden_dims[0]))
        self.layers.append(nn.ReLU())
        for l in range(1, self.L):
            self.layers.append(nn.Linear(hidden_dims[l-1], hidden_dims[l]))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_dims[-1], 1))
        self.layers = nn.ModuleList(self.layers)
    
    def forward(self, x):
        x = x.view(-1, x.size()[1])
        o = []; o.append(x)
        for layer_idx, layer_name in enumerate(self.layers[:-1]):
            o.append(layer_name(o[-1]))
        o.append(self.layers[-1](o[-1]))
        output = o[-1]
        return output

class cifar10_classifier(nn.Module):
    def __init__(self, seed_num):
        super(cifar10_classifier, self).__init__()
        
        torch.manual_seed(seed_num)
        np.random.seed(seed_num)
        
        self.dataset = 'cifar10'
        self.is_imagedata = True
        
        self.relu = nn.ReLU()
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.conv3_1 = nn.Conv2d(64, 64, kernel_size=3, padding=0)
        self.bn3_1 = nn.BatchNorm2d(64)
        self.conv3_2 = nn.Conv2d(64, 128, kernel_size=1, padding=0)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.linear = nn.Linear(128, 1)
        
    def forward(self, x):
        o1_1 = self.relu(self.bn1_1(self.conv1_1(x)))
        o1_2 = self.relu(self.bn1_2(self.conv1_2(o1_1)))     
        o1_2 = F.max_pool2d(o1_2, kernel_size = 2, stride = 2)
        o2_1 = self.relu(self.bn2_1(self.conv2_1(o1_2)))
        o2_2 = self.relu(self.bn2_2(self.conv2_2(o2_1)))       
        o2_2 = F.max_pool2d(o2_2, kernel_size = 2, stride = 2)
        o3_1 = self.relu(self.bn3_1(self.conv3_1(o2_2)))
        o3_2 = self.relu(self.bn3_2(self.conv3_2(o3_1)))       
        o3_2 = F.avg_pool2d(o3_2, 6)
        o3_2 = o3_2.view(-1, 128)
        output = self.linear(o3_2)

        return output

