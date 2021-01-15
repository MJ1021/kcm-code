import torch
import torch.nn as nn
import numpy as np
import utils

class Trainer():
   
    def __init__(self, net, criterion, surrogate_loss, optimizer_config, use_cuda, device):
        super(Trainer, self).__init__()
        self.net = net
        self.criterion = criterion
        self.surrogate_loss = surrogate_loss
        self.use_cuda = use_cuda
        self.device = device
        self.is_imagedata = self.net.__dict__['is_imagedata']
        self.dataset = self.net.__dict__['dataset']
        
        if optimizer_config['name'] == 'adam':
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=optimizer_config['base_lr'],
                                             weight_decay=optimizer_config['weight_decay'])
        elif optimizer_config['name'] == 'sgd':
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr=optimizer_config['base_lr'], momentum=0.9,
                                             weight_decay=optimizer_config['weight_decay'])
  
    def self_ensemble(self, inputs, h, N):
          
        if self.is_imagedata:
            B, C, H, W = inputs.size()
            shifting_variable = np.random.normal(0.0, h, size=(B, N, C, H, W)) 
            shifting_variable = torch.from_numpy(shifting_variable).type(torch.FloatTensor) 
            inputs_tiled = np.tile(inputs.data.cpu().numpy(), (N, 1, 1, 1, 1)).transpose((1, 0, 2, 3, 4))
            inputs_tiled = torch.from_numpy(inputs_tiled).type(torch.FloatTensor) 

            input_shifted = inputs_tiled-shifting_variable 
            if self.use_cuda:
                input_shifted = input_shifted.cuda()
            outputs = self.net(input_shifted.view(B*N, C, H, W)) 
        else:
            B, d = inputs.size()
            mean = np.zeros(d,)
            cov = (h**2)*np.identity(d) #
            shifting_variable = np.random.multivariate_normal(mean, cov, N) 
            shifting_variable_tiled = np.tile(shifting_variable, (B, 1, 1))
            shifting_variable_tiled = torch.from_numpy(shifting_variable_tiled)

            inputs_tiled = np.tile(inputs.data.cpu().numpy(), (N, 1, 1)).transpose((1, 0, 2))
            inputs_tiled = torch.from_numpy(inputs_tiled)
            
            if self.use_cuda:
                shifting_variable_tiled = shifting_variable_tiled.type(torch.cuda.FloatTensor)
                inputs_tiled = inputs_tiled.type(torch.cuda.FloatTensor)
            input_shifted = torch.clamp(inputs_tiled-shifting_variable_tiled, -1.0, 1.0)

            outputs = self.net(input_shifted.view(B*N, d))  
        outputs = torch.mean(outputs.view(B, N), dim = 1)
        
        return outputs

    def train(self, dataloader, alpha, h, N):
        self.net.train() 
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            self.optimizer.zero_grad()
            if self.use_cuda:     
                inputs, targets = inputs.cuda(), targets.cuda() 
            # mix-up
            if alpha>0:
                inputs, targets = utils.mixup_data(inputs, targets, alpha, self.use_cuda)
            
            if h == 0.0:
                outputs = self.net(inputs)
            else:
                outputs = self.self_ensemble(inputs, h, N)           
                
            if self.surrogate_loss == 'hinge':
                dummy_input = torch.zeros(targets.shape, dtype=torch.float, device=self.device) 
                loss = self.criterion(outputs.squeeze(), dummy_input.squeeze(), targets)
            else:
                loss = self.criterion(outputs.squeeze(), targets)        
            
                      
            loss.backward()
            self.optimizer.step()

            train_loss += targets.size(0)*loss.data.cpu().numpy()
            y_hat_class = np.where(outputs.data.cpu().numpy()<0, -1, 1)
            
            correct += np.sum(targets.data.cpu().numpy()==y_hat_class.squeeze())
            total += targets.size(0) 
        del batch_idx, inputs, targets, outputs
        
        return (train_loss/total, 100.*correct/total)

    def evaluation(self, dataloader, h, N_eval):
        
        self.net.eval()
        eval_loss = 0
        correct = 0
        total = 0
        with torch.no_grad(): 
            for batch_idx, (inputs, targets) in enumerate(dataloader): 
                if self.use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                    
                if h == 0.0:
                    outputs = self.net(inputs)
                else:
                    outputs = self.self_ensemble(inputs, h, N_eval)

                if self.surrogate_loss == 'hinge':
                    dummy_input = torch.zeros(targets.shape, dtype=torch.float, device=self.device)
                    loss = self.criterion(outputs.squeeze(), dummy_input.squeeze(), targets)
                else:
                    loss = self.criterion(outputs.squeeze(), targets)
                
                eval_loss += targets.size(0)*loss.data.cpu().numpy()
                y_hat_class = np.where(outputs.data.cpu().numpy()<0, -1, 1)
                correct += np.sum(targets.data.cpu().numpy().reshape(-1)==y_hat_class.reshape(-1))
                total += targets.size(0)
            del batch_idx, inputs, targets, outputs
                
            return (eval_loss/total, 100.*correct/total)
