# train.py
#
# Runs the original 50-50 train/test split without StratifiedKFolds or train/val splits; testing is done on test data
# To be used to train the final model
# Can be run with multitask or RandAug enabled

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
import shutil
import time
from config import num_classes, model_name, model_path, lr_milestones, lr_decay_rate, input_size, \
    root, end_epoch, save_interval, init_lr, batch_size, CUDA_VISIBLE_DEVICES, weight_decay, \
    proposalN, set, channels, patience_counter, patience, multitask, rand_aug, N, M
from utils.train_model import train, train_multitask
from utils.read_dataset import read_dataset
from utils.auto_load_resume import auto_load_resume
from networks.model import MainNet, MainNetMultitask

import os, sys, random
import numpy as np
from sklearn.model_selection import StratifiedKFold

os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES

# Set seeds
seed = 42
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

def main():
    _, trainloader, _, testloader = read_dataset(input_size, batch_size, root, set)
    # image will be resize to the input_size
    # batch size means the number of images the nn process before updating the weight and biases 
    # root is the root to the dataset
    # set is the dataset name (change in config)

    # Load checkpoint
    save_path = os.path.join(model_path, model_name)
    if os.path.exists(save_path):
        load_model_from_path = save_path
        if not os.path.exists(load_model_from_path):
            os.makedirs(load_model_from_path)

        # Create model
        if (multitask):
            model = MainNetMultitask(proposalN=proposalN, num_classes=num_classes, channels=channels)
        else:
            model = MainNet(proposalN=proposalN, num_classes=num_classes, channels=channels)
        
        start_epoch, lr, patience_counter = auto_load_resume(model, load_model_from_path, status='train')
        print(f'Patience counter starting from: {patience_counter}')
        assert start_epoch < end_epoch, 'maximum number of epochs reached'
        assert patience_counter <= patience
    else:
        os.makedirs(save_path)
        start_epoch = 0
        lr = init_lr
        patience_counter = 0
    
    if (multitask):
        print(f'Multitask: {multitask}')
        print(f'Random Augmentation: {rand_aug}')
        if (rand_aug):
            print(f'N:{N} M: {M}')

        # Create model
        model = MainNetMultitask(proposalN=proposalN, num_classes=num_classes, channels=channels)

        criterion = nn.CrossEntropyLoss()

        # Define optimizers
        parameters = model.parameters()
        optimizer = torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)

        model = model.cuda()

        scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_decay_rate)

        train_multitask(
            model=model,
            trainloader=trainloader,
            testloader=testloader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            save_path=save_path,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            patience_counter=patience_counter,
            save_interval=save_interval
        )

    else:
        print(f'Multitask: {multitask}')
        print(f'Random Augmentation: {rand_aug}')
        if (rand_aug):
            print(f'N:{N} M: {M}')

        # Create model
        model = MainNet(proposalN=proposalN, num_classes=num_classes, channels=channels)

        criterion = nn.CrossEntropyLoss()

        # Define optimizers
        parameters = model.parameters()
        optimizer = torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)

        model = model.cuda()

        scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_decay_rate)

        train(
            model=model,
            trainloader=trainloader,
            testloader=testloader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            save_path=save_path,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            patience_counter=patience_counter,
            save_interval=save_interval
        )

if __name__ == '__main__':
    main()