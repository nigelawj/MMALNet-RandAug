#coding=utf-8
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
import shutil
import time
from config import num_classes, model_name, model_path, lr_milestones, lr_decay_rate, input_size, \
    root, end_epoch, save_interval, init_lr, batch_size, CUDA_VISIBLE_DEVICES, weight_decay, \
    proposalN, set, channels, num_folds, start_from_fold, patience_counter, patience
from utils.train_model import train
from utils.read_dataset import read_dataset
from utils.auto_load_resume import auto_load_resume
from networks.model import MainNet

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
    #加载数据
    trainset, _, testset, _ = read_dataset(input_size, batch_size, root, set)
    #image will be resize to the input_size
    #batch size means the number of images the nn process before updating the weight and biases 
    #root is the root to the dataset
    #set is the dataset name (change in config)


    #定义模型
    model = MainNet(proposalN=proposalN, num_classes=num_classes, channels=channels)
    

    #设置训练参数
    criterion = nn.CrossEntropyLoss()

    parameters = model.parameters()

    # Load checkpoint from a fold number
    save_path = os.path.join(model_path, model_name)
    if os.path.exists(save_path):
        load_model_from_path = os.path.join(save_path, f'fold_{start_from_fold}')
        if not os.path.exists(load_model_from_path):
            os.makedirs(load_model_from_path)
        start_epoch, lr, patience_counter = auto_load_resume(model, load_model_from_path, status='train')
        assert start_epoch < end_epoch, 'end of fold reached, please increment start_from_fold'
        assert start_from_fold < num_folds
        assert patience_counter <= patience
    else:
        os.makedirs(save_path)
        start_epoch = 0
        lr = init_lr
        patience_counter = 0

    # define optimizers
    optimizer = torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)

    model = model.cuda()  # 部署在GPU

    scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_decay_rate)

    # 保存config参数信息
    time_str = time.strftime("%Y%m%d-%H%M%S")
    shutil.copy('./config.py', os.path.join(save_path, "{}config.py".format(time_str)))

    print('\nSplitting trainset into train and val sets (80:20)\nNote testset is loaded but unused, and will not be used unless test.py is run.')
    # NOTE: split train into train/val set; but for consistency of code we'll leave the variable names as 'test' instead of 'val'
    # split train set into train/val 80:20
    X_train = []
    y_train = []
    for i, j in trainset.train_img_label:
        X_train.append(i)
        y_train.append(j)
    # convert lists into numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    skf = StratifiedKFold(n_splits=num_folds, random_state=seed, shuffle=True)

    for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
        print(f'\n=============== Fold {fold+1} ==================')
        if (fold+1 < start_from_fold):
            print('Skipping this fold...\n')
            continue
        # Prepare save_path for the fold
        save_path_fold = os.path.join(save_path, str(f'fold_{fold+1}'))

        # Split trainloader into train and val loaders
        X_train_fold = X_train[train_index]
        X_val_fold = X_train[val_index]
        y_train_fold = y_train[train_index]
        y_val_fold = y_train[val_index]

        # Zip back the X and y values
        train_img_label_fold = list(zip(X_train_fold, y_train_fold))
        val_img_label_fold = list(zip(X_val_fold, y_val_fold))

        # Hijack the original trainset with the X and y for the particular fold
        trainset_fold = trainset
        trainset_fold.train_img_label = train_img_label_fold
        valset_fold = testset
        valset_fold.test_img_label = val_img_label_fold # variable name kept as test_img_label for code consistency

        print(f'Size of trainset: {len(trainset_fold)}')
        print(f'Size of valset: {len(valset_fold)}')
        
        # Recreate DataLoaders with train and val sets
        trainloader_fold = torch.utils.data.DataLoader(trainset_fold, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=False)
        valloader_fold = torch.utils.data.DataLoader(valset_fold, batch_size=batch_size, shuffle=False, num_workers=8, drop_last=False)

        # 开始训练
        train(
            model=model,
            trainloader=trainloader_fold,
            testloader=valloader_fold,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            save_path=save_path_fold,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            patience_counter=patience_counter,
            save_interval=save_interval
        )
        start_epoch = 0 # refresh start_epoch for next fold

        print(f'\n=============== End of fold {fold+1} ==================\n')


if __name__ == '__main__':
    main()