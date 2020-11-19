#coding=utf-8
import torch
import torch.nn as nn
import sys
from tqdm import tqdm
from config import input_size, root, proposalN, channels, multitask
from utils.read_dataset import read_dataset
from utils.auto_load_resume import auto_load_resume
from networks.model import MainNet, MainNetMultitask

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def main():
    CUDA = torch.cuda.is_available()
    DEVICE = torch.device("cuda" if CUDA else "cpu")

    set = 'CompCars' # ensure dataset is set properly
    
    if set == 'CUB':
        root = './datasets/CUB_200_2011'  # dataset path
        # model path
        pth_path = "./models/cub_epoch144.pth"
        num_classes = 200
    elif set == 'CompCars':
        root = './datasets/CompCars'  # dataset path
        # model path
        pth_path = "./models/car_model_randaug_epoch45.pth" # remember to change as per saved model's name
        num_classes = 431
        if (multitask):
            num_classes = (431, 75)
            assert isinstance(num_classes, tuple), "Multitask mode is enabled but num_classes is an integer; please pass in a tuple for multiple predictions."
            assert num_classes[0] == 431, "Please put the number of car models (431) as the first element of the tuple"
    elif set == 'Aircraft':
        root = './datasets/FGVC-aircraft'  # dataset path
        # model path
        pth_path = "./models/air_epoch146.pth"
        num_classes = 100

    batch_size = 10

    # load test dataset
    _, _, _, testloader = read_dataset(input_size, batch_size, root, set)

    # create model
    if (multitask):
        model = MainNetMultitask(proposalN=proposalN, num_classes=num_classes, channels=channels)
    else:
        model = MainNet(proposalN=proposalN, num_classes=num_classes, channels=channels)

    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    # Load checkpoint model
    if os.path.exists(pth_path):
        epoch = auto_load_resume(model, pth_path, status='test')
    else:
        sys.exit('No saved model checkpoint detected.')

    print('Testing on saved model...')
    if (multitask):
        print('MULTITASK TEST...')
        object_correct_1 = 0
        object_correct_2 = 0
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(tqdm(testloader)):
                if set == 'CUB':
                    x, y, boxes, _ = data
                else:
                    x, y = data
                x = x.to(DEVICE)

                y_1, y_2 = y[0], y[1]

                y_1 = y_1.to(DEVICE)
                y_2 = y_2.to(DEVICE)

                local_logits_1, local_logits_2, local_imgs = model(x, epoch, i, 'test', DEVICE)[-3:]
                # local
                pred_1 = local_logits_1.max(1, keepdim=True)[1]
                pred_2 = local_logits_2.max(1, keepdim=True)[1]
                object_correct_1 += pred_1.eq(y_1.view_as(pred_1)).sum().item()
                object_correct_2 += pred_2.eq(y_2.view_as(pred_2)).sum().item()

            print('\nObject branch accuracy for task 1: {}/{} ({:.2f}%)\n'.format(object_correct_1, len(testloader.dataset), 100. * object_correct_1 / len(testloader.dataset)))
            print('\nObject branch accuracy for task 2: {}/{} ({:.2f}%)\n'.format(object_correct_2, len(testloader.dataset), 100. * object_correct_2 / len(testloader.dataset)))
    else:
        object_correct = 0
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(tqdm(testloader)):
                if set == 'CUB':
                    x, y, boxes, _ = data
                else:
                    x, y = data
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                local_logits, local_imgs = model(x, epoch, i, 'test', DEVICE)[-2:]
                # local
                pred = local_logits.max(1, keepdim=True)[1]
                object_correct += pred.eq(y.view_as(pred)).sum().item()

            print('\nObject branch accuracy: {}/{} ({:.2f}%)\n'.format(
                    object_correct, len(testloader.dataset), 100. * object_correct / len(testloader.dataset)))

if __name__ == "__main__":
    main()