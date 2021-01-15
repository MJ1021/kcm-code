import argparse, csv, itertools, matplotlib, os, sklearn, time, torch, torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
import model, trainer, utils

def print_and_write(file_writer, print_statement):
    print(print_statement)
    file_writer.write(print_statement+'\n')
    return None

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='twomoon', type=str, help='the name of dataset')
parser.add_argument('--optimizer', default='adam', type=str, help='the name of optimizer')
parser.add_argument('--base_lr', default = 1e-2, type=float, help='the value of initial learning rate')
parser.add_argument('--lr_schedule', default= None, type=str, help='whether user adjust the learning rate: put any string if you want to use the learning rate schduler')
parser.add_argument('--weight_decay', default = 0.0, type=float, help='the value of weight decay')
parser.add_argument('--alpha_list', nargs='+', type=float, help='the list of levels of alpha')
parser.add_argument('--h_list', nargs='+', type=float, help='the list of levels of bandwidth')
parser.add_argument('--batch_size', default = 64, type=int, help='the size of mini-batch')
parser.add_argument('--patience', default = 20, type=int, help='early stop patience')
parser.add_argument('--max_epoch', default = 300, type=int, help='the number of epochs')
parser.add_argument('--M', default = 100, type=int, help='the number of replications')
parser.add_argument('--N', default = 1000, type=int, help='the number of samples to approximate self-ensembled models in training set')
parser.add_argument('--N_val', default = 1000, type=int, help='the number of samples to approximate self-ensembled models in validation set')
parser.add_argument('--N_test', default = 1000, type=int, help='the number of samples to approximate self-ensembled models in test set')
parser.add_argument('--sampling_proportion_list', nargs='+', type=float, help='the list of levels of sampling proportions')
parser.add_argument('--ratio_train_to_val', default= 2.0, type=float, help='the ratio of the size of train set to validation set')
parser.add_argument('--seed_num_list', nargs='+', type=int, help='the list of seed numbers. If the value of M is specified, seed numbers are 0, 1, ..., and M-1.')
parser.add_argument('--surrogate_loss', default = 'hinge', type=str, help='the type of loss function (hinge or logistic)')
opt = parser.parse_args()
print(opt)


if opt.seed_num_list == None:
    opt.seed_num_list = np.arange(opt.M)
    
if opt.dataset=='twomoon':
    class0, class1 = None, None
    hidden_dims = [16, 16]
    model_path = './results/trained_models/twomoon/'
    log_path = './results/logs/twomoon/' 
    noise = 0.15
    n_samples = 500
    print_period = 5
    ratio_train_to_test = 2.0
    transform_train = None 
    transform_eval = None
    opt.N_val = opt.N
    opt.N_test = opt.N
elif opt.dataset=='cifar10':
    class0, class1 = 'cat', 'dog'
    hidden_dims = None
    model_path = './results/trained_models/cifar10/'
    log_path = './results/logs/cifar10/'
    noise = None
    n_samples = None
    print_period = 5
    ratio_train_to_test = None
    transform_train = torchvision.transforms.Compose([torchvision.transforms.RandomCrop(32, padding =4),
                                                      torchvision.transforms.RandomHorizontalFlip(),
                                                      torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Lambda(lambda x: 2.0*x-1.0), 
                                                     ])
    transform_eval = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Lambda(lambda x: 2.0*x-1.0),
                                                    ])
    dl_num_workers = 0
else:
    raise ValueError('Invalid dataset name')

# specify gpu environment
use_cuda = torch.cuda.is_available() # True/False
torch.backends.cudnn.deterministic = True 
torch.backends.cudnn.benchmark = False 
device = torch.device("cuda" if use_cuda else "cpu")

# generate list and dictionary containing information about alpha, h, and sampling proportion
config_list, config_dict = [], {}
for alpha, h, sampling_proportion in list(itertools.product(opt.alpha_list, opt.h_list, opt.sampling_proportion_list)):
    config = 'alpha=%.4f-h=%.4f-sampling_proportion=%.2f' % (alpha, h, sampling_proportion)
    config_list.append(config)
    config_dict[config] ={}
    config_dict[config]['alpha'] = alpha
    config_dict[config]['h'] = h
    config_dict[config]['sampling_proportion'] = sampling_proportion
del alpha, h, sampling_proportion

# generate directories containing logs and models of current experiment.
now = datetime.now()
date_time = now.strftime("%m.%d.%Y-%H:%M:%S")
current_log_path = log_path+'Experiments-'+date_time
if not os.path.exists(current_log_path):
    os.mkdir(current_log_path)
current_model_path = model_path+'Experiments-'+date_time
if not os.path.exists(current_model_path):
    os.mkdir(current_model_path)
cmd_log = open((current_log_path+'/cmd_log.txt'), 'w')

# start training procedure
loss, accuracy, net, early_stop_epoch, sample_size = {}, {}, {}, {}, {}
best_model_loss, best_model_accuracy = {}, {}

# open log file (csv)
log_name_summary = 'log_final'
with open((current_log_path+'/'+log_name_summary+'.csv'), 'w') as logfile_final:
    logfile_final = csv.writer(logfile_final, delimiter=',')
    logfile_final.writerow(['seed_num','config','test accuracy','cumulated time'])
    
for seed_num in opt.seed_num_list:
    print_and_write(cmd_log, '='*108)
    
    seed_config = 'seed_num=%d' % seed_num
    print_and_write(cmd_log, seed_config)
    
    torch.manual_seed(seed_num) 
    np.random.seed(seed_num) 
        
    net[seed_config] = {}
    loss[seed_config], accuracy[seed_config], early_stop_epoch[seed_config], sample_size[seed_config] = {}, {}, {}, {}
    best_model_loss[seed_config], best_model_accuracy[seed_config] = {}, {}
    
    
    # train models for every configurations with current seed number
    for config in config_list:
        print_and_write(cmd_log, '-'*108)
            
        
        
        torch.manual_seed(seed_num)
        np.random.seed(seed_num)
        
        loss[seed_config][config], accuracy[seed_config][config] = {}, {}
        for splitname in ['train', 'val', 'test']:
            loss[seed_config][config][splitname] = []
            accuracy[seed_config][config][splitname] = []
        del splitname
        best_model_loss[seed_config][config], best_model_accuracy[seed_config][config] = {}, {}
        
        cumulated_time = 0
        t_start_training = time.time()
        print_and_write(cmd_log, config+'-'+seed_config)
        
        # load and split data
        loadandsplitdata = utils.LoadAndSplitData(opt.dataset, n_samples, noise, class0, class1,
                                                  config_dict[config]['sampling_proportion'],
                                                  opt.ratio_train_to_val, ratio_train_to_test, seed_num)
        X, Y = {}, {}
        X['train'], Y['train'], X['val'], Y['val'], X['test'], Y['test'] = loadandsplitdata.load_and_split_data() 
        
        # define dataloader
        dataloader, sample_size[seed_config][config] = {}, {}
        if opt.dataset == 'twomoon':
            for splitname in ['train', 'val', 'test']:
                if splitname == 'train':
                    dataloader[splitname] = DataLoader(utils.PrepareData_twomoon(X=X[splitname], y=Y[splitname]), batch_size=opt.batch_size, shuffle=True)
                else:
                    dataloader[splitname] = DataLoader(utils.PrepareData_twomoon(X=X[splitname], y=Y[splitname]), batch_size=opt.batch_size, shuffle=True)
                sample_size[seed_config][config][splitname] = np.shape(X[splitname])[0]         
            del splitname
        if opt.dataset == 'cifar10':
            for splitname in ['train', 'val', 'test']:
                if splitname == 'train':
                    dataloader[splitname] = DataLoader(utils.PrepareData_cifar10(X=X[splitname], y=Y[splitname], transform=transform_train, use_cuda=use_cuda, seed_num=seed_num), batch_size=opt.batch_size, shuffle=True, num_workers=dl_num_workers)
                else:
                    dataloader[splitname] = DataLoader(utils.PrepareData_cifar10(X=X[splitname], y=Y[splitname], transform=transform_eval, use_cuda=use_cuda, seed_num=seed_num), batch_size=opt.batch_size, shuffle=True, num_workers=dl_num_workers)
                sample_size[seed_config][config][splitname] = np.shape(X[splitname])[0]         
            del splitname
        
        # build model
        buildmodel = utils.BuildModel(opt.dataset, hidden_dims, seed_num)
        net[seed_config][config] = buildmodel.build_model()
       
        if use_cuda:
            net[seed_config][config].cuda()
    
        if opt.surrogate_loss == 'hinge':
            model_loss = nn.MarginRankingLoss(margin=1.0)
        else:
            model_loss = nn.SoftMarginLoss()
        
        # build trainer
        optimizer_config = {'name': opt.optimizer,
                            'base_lr': opt.base_lr,
                            'weight_decay': opt.weight_decay}
        net_trainer = trainer.Trainer(net[seed_config][config], model_loss, opt.surrogate_loss, optimizer_config, use_cuda, device)
        
        # open log file (csv)
        entire_config = config+'-'+seed_config
        log_name = 'log_' + entire_config
        with open((current_log_path+'/'+log_name+'.csv'), 'w') as logfile:
            logfile = csv.writer(logfile, delimiter=',')
            logfile.writerow(['epoch', 'train loss', 'val loss', 'test loss', 'train accuracy', 'val accuracy', 'test accuracy', 'cumulated time'])
   
        # train models for specified configuration and seed number
        for epoch in range(opt.max_epoch):
            t_start_epoch = time.time()
                 
            # learning rate scheduler
            if (opt.lr_schedule is not None) & (opt.dataset =='cifar10'):
                for param_group in net_trainer.__dict__['optimizer'].param_groups:
                    if epoch >= 30:
                        param_group['lr'] = optimizer_config['base_lr']/10.0
                    if epoch >= 60:
                        param_group['lr'] = optimizer_config['base_lr']/100.0   
                    if epoch >= 120:
                        param_group['lr'] = optimizer_config['base_lr']/1000.0
            
            # update parameters and store losses and accuracies            
            for splitname in ['train', 'val', 'test']:
                if splitname == 'train':
                    current_loss, current_accuracy = net_trainer.train(dataloader[splitname],
                                                                       config_dict[config]['alpha'],
                                                                       config_dict[config]['h'], opt.N)
                elif splitname == 'val':
                    current_loss, current_accuracy = net_trainer.evaluation(dataloader[splitname],
                                                                            config_dict[config]['h'], opt.N_val)
                else:
                    current_loss, current_accuracy = net_trainer.evaluation(dataloader[splitname],
                                                                            config_dict[config]['h'], opt.N_test)                                              
                loss[seed_config][config][splitname].append(current_loss.item())
                accuracy[seed_config][config][splitname].append(current_accuracy.item())
            del splitname
            
            
            # save the best model for validation loss
            if (epoch == 0) or (best_val_loss > loss[seed_config][config]['val'][epoch]):
                current_patience = 0
                best_val_loss = loss[seed_config][config]['val'][epoch]
                torch.save(net[seed_config][config].state_dict(), current_model_path+'/'+entire_config+'.pth') 
                early_stop_epoch[seed_config][config] = epoch
                for splitname in ['train', 'val', 'test']:
                    best_model_loss[seed_config][config][splitname] = loss[seed_config][config][splitname][epoch]
                    best_model_accuracy[seed_config][config][splitname] = accuracy[seed_config][config][splitname][epoch]
                del splitname
            else:
                current_patience += 1
            
            # print and save various numerics including losses and accuracies to monitor learning process.
            t_end_epoch = time.time()
            cumulated_time += t_end_epoch - t_start_epoch
            if epoch % print_period ==0:
                print_state = ("[Epoch %d] loss (train/val/test): %.5f/%.5f/%.5f, accuracy (train/val/test): %.3f/%.3f/%.3f "
                               % (epoch, loss[seed_config][config]['train'][epoch], loss[seed_config][config]['val'][epoch],
                                  loss[seed_config][config]['test'][epoch], accuracy[seed_config][config]['train'][epoch],
                                  accuracy[seed_config][config]['val'][epoch], accuracy[seed_config][config]['test'][epoch]))
                print_and_write(cmd_log, print_state)
                print_and_write(cmd_log, 'Cumulated time: %.4f (sec)' % cumulated_time)
            
            with open((current_log_path+'/'+log_name+'.csv'), 'a') as logfile:
                logwriter = csv.writer(logfile, delimiter=',')
                logwriter.writerow([epoch, loss[seed_config][config]['train'][epoch], loss[seed_config][config]['val'][epoch], loss[seed_config][config]['test'][epoch], accuracy[seed_config][config]['train'][epoch], accuracy[seed_config][config]['val'][epoch], accuracy[seed_config][config]['test'][epoch], cumulated_time])
            
            # early stop
            if current_patience == opt.patience:
                best_model_epoch = early_stop_epoch[seed_config][config]
                for splitname in ['train', 'val', 'test']:
                    best_model_loss[seed_config][config][splitname] = loss[seed_config][config][splitname][best_model_epoch]
                    best_model_accuracy[seed_config][config][splitname] = accuracy[seed_config][config][splitname][best_model_epoch]
                del splitname
                for param_group in net_trainer.__dict__['optimizer'].param_groups:
                    print_and_write(cmd_log, 'final_lr_main: %.6f' % param_group['lr'])
                break                
       
            if epoch == (opt.max_epoch-1):
                for param_group in net_trainer.__dict__['optimizer'].param_groups:
                    print_and_write(cmd_log, 'final_lr_main: %.6f' % param_group['lr'])
    
        del epoch, net[seed_config][config] # end for epoch
        
        t_end_training = time.time()
        print_and_write(cmd_log, 'Test loss: %s' % str(best_model_loss[seed_config][config]['test']))
        print_and_write(cmd_log, 'Test accuracy: %s' % str(best_model_accuracy[seed_config][config]['test']))
        print_and_write(cmd_log, 'Total training time: %.4f (sec)' % (t_end_training - t_start_training))
        
        with open((current_log_path+'/'+log_name_summary+'.csv'), 'a') as logfile_final:
            logwriter_final = csv.writer(logfile_final, delimiter=',')
            logwriter_final.writerow([seed_num, config, best_model_accuracy[seed_config][config]['test'], cumulated_time])

    del config
del seed_num

# Summary experiments
print_and_write(cmd_log, '='*108)
print_and_write(cmd_log, '[Summary statistics]')
for config in config_list:
    print_and_write(cmd_log, '-'*108)
    print_and_write(cmd_log, config)
    print_and_write(cmd_log, 'sample size (train/val/test): %d/%d/%d' % (sample_size[seed_config][config]['train'],
                                                     sample_size[seed_config][config]['val'],
                                                     sample_size[seed_config][config]['test']))
    test_loss, test_accuracy = [], []
    for seed_num in opt.seed_num_list:
        seed_config = 'seed_num=%d' % seed_num
        test_loss.append(best_model_loss[seed_config][config]['test'])
        test_accuracy.append(best_model_accuracy[seed_config][config]['test'])
    print_and_write(cmd_log, 'Test loss (mean with standard error): %.3f (%.3f)' % 
                    (np.mean(test_loss), np.std(test_loss)/np.sqrt(len(opt.seed_num_list))))
    print_and_write(cmd_log, 'Test accuracy (mean with standard error): %.3f (%.3f)' %
                    (np.mean(test_accuracy), np.std(test_accuracy)/np.sqrt(len(opt.seed_num_list))))

print_and_write(cmd_log, '='*108)
print_and_write(cmd_log, '[Common configurations]')
print_and_write(cmd_log, str(opt))
print_and_write(cmd_log, 'class0: %s' % str(class0))
print_and_write(cmd_log, 'class1: %s' % str(class1))
print_and_write(cmd_log, 'hidden_dims: %s' % str(hidden_dims))
print_and_write(cmd_log, 'max_epoch: %s' % str(opt.max_epoch))
print_and_write(cmd_log, 'model_path: %s' % str(model_path))
print_and_write(cmd_log, 'log_path: %s' % str(log_path))
print_and_write(cmd_log, 'noise: %s' % str(noise))
print_and_write(cmd_log, 'n_samples: %s' % str(n_samples))
print_and_write(cmd_log, 'print_period: %s' % str(print_period))
print_and_write(cmd_log, 'ratio_train_to_test: %s' % str(ratio_train_to_test))
print_and_write(cmd_log, 'ratio_train_to_val: %s' % str(opt.ratio_train_to_val))
print_and_write(cmd_log, 'surrogate loss: %s' % str(opt.surrogate_loss))

cmd_log.close()
