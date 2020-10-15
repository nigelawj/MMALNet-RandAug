import torch
from tqdm import tqdm
import os
from tensorboardX import SummaryWriter
import numpy as np
from config import coordinates_cat, proposalN, set, vis_num
from utils.cal_iou import calculate_iou
from utils.vis import image_with_boxes

def eval(model, testloader, criterion, status, save_path, epoch):
    model.eval()
    print(f'\nEvaluating on {status} set')
    if status == 'train':
        print('NOTE: evaluation on train set in this context simply means to run eval mode on the training set to obtain more performance metrics which were not obtained during the training; this is by design of the authors of this model.')

    raw_loss_sum = 0
    local_loss_sum = 0
    windowscls_loss_sum = 0
    total_loss_sum = 0
    iou_corrects = 0
    raw_correct = 0
    local_correct = 0

    with torch.no_grad():
        for i, data in enumerate(tqdm(testloader)):
            if set == 'CUB':
                images, labels, boxes, scale = data
            else:
                images, labels = data
            images = images.cuda()
            labels = labels.long().cuda()

            proposalN_windows_score,proposalN_windows_logits, indices, \
            window_scores, coordinates, raw_logits, local_logits, local_imgs = model(images, epoch, i, status)

            raw_loss = criterion(raw_logits, labels)
            local_loss = criterion(local_logits, labels)
            windowscls_loss = criterion(proposalN_windows_logits,
                                        labels.unsqueeze(1).repeat(1, proposalN).view(-1))

            total_loss = raw_loss + local_loss + windowscls_loss

            raw_loss_sum += raw_loss.item()
            local_loss_sum += local_loss.item()
            windowscls_loss_sum += windowscls_loss.item()

            total_loss_sum += total_loss.item()

            if set == 'CUB':
                # computer resized coordinates of boxes
                boxes_coor = boxes.float()
                resized_boxes = torch.cat([(boxes_coor[:,0] * scale[:, 0]).unsqueeze(1) ,(boxes_coor[:,1] * scale[:, 1]).unsqueeze(1),
                                           (boxes_coor[:,2] * scale[:, 0]).unsqueeze(1), (boxes_coor[:,3] * scale[:, 1]).unsqueeze(1)], dim=1)
                resized_coor = torch.cat([resized_boxes[:,0].unsqueeze(1) ,resized_boxes[:,1].unsqueeze(1),
                                           (resized_boxes[:,0] + resized_boxes[:,2]).unsqueeze(1), (resized_boxes[:,1]+resized_boxes[:,3]).unsqueeze(1)], dim=1).round().int()


                iou = calculate_iou(coordinates.cpu().numpy(), resized_coor.numpy())
                iou_corrects += np.sum(iou >= 0.5)

            # correct num
            # raw
            pred = raw_logits.max(1, keepdim=True)[1]
            raw_correct += pred.eq(labels.view_as(pred)).sum().item()
            # local
            pred = local_logits.max(1, keepdim=True)[1]
            local_correct += pred.eq(labels.view_as(pred)).sum().item()

            # raw branch tensorboard
            if i == 0:
                if set == 'CUB':
                    box_coor = resized_coor[:vis_num].numpy()
                    pred_coor = coordinates[:vis_num].cpu().numpy()
                    with SummaryWriter(log_dir=os.path.join(save_path, 'log'), comment=status + 'raw') as writer:
                        cat_imgs = []
                        for j, coor in enumerate(box_coor):
                            img = image_with_boxes(images[j], [coor])
                            img = image_with_boxes(img, [pred_coor[j]], color=(0, 255, 0))
                            cat_imgs.append(img)
                        cat_imgs = np.concatenate(cat_imgs, axis=1)
                        writer.add_images(status + '/' + 'raw image with boxes', cat_imgs, epoch, dataformats='HWC')

            # object branch tensorboard
            if i == 0:
                indices_ndarray = indices[:vis_num,:proposalN].cpu().numpy()
                with SummaryWriter(log_dir=os.path.join(save_path, 'log'), comment=status + 'object') as writer:
                    cat_imgs = []
                    for j, indice_ndarray in enumerate(indices_ndarray):
                        img = image_with_boxes(local_imgs[j], coordinates_cat[indice_ndarray])
                        cat_imgs.append(img)
                    cat_imgs = np.concatenate(cat_imgs, axis=1)
                    writer.add_images(status + '/' + 'object image with windows', cat_imgs, epoch, dataformats='HWC')

            # if status == 'train':
            #     if i >= 2 :
            #         break

    raw_loss_avg = raw_loss_sum / (i+1)
    local_loss_avg = local_loss_sum / (i+1)
    windowscls_loss_avg = windowscls_loss_sum / (i+1)
    total_loss_avg = total_loss_sum / (i+1)

    raw_accuracy = raw_correct / len(testloader.dataset)
    local_accuracy = local_correct / len(testloader.dataset)


    return raw_loss_avg, windowscls_loss_avg, total_loss_avg, raw_accuracy, local_accuracy, \
           local_loss_avg

def eval_multitask(model, testloader, criterion, status, save_path, epoch):
    model.eval()
    print(f'\nEvaluating on {status} set')
    if status == 'train':
        print('NOTE: evaluation on train set in this context simply means to run eval mode on the training set to obtain more performance metrics which were not obtained during the training; this is by design of the authors of this model.')

    raw_loss_sum_1 = 0
    raw_loss_sum_2 = 0
    local_loss_sum_1 = 0
    local_loss_sum_2 = 0
    windowscls_loss_sum_1 = 0
    windowscls_loss_sum_2 = 0
    total_loss_sum = 0
    iou_corrects = 0
    raw_correct_1 = 0
    raw_correct_2 = 0
    local_correct_1 = 0
    local_correct_2 = 0

    with torch.no_grad():
        for i, data in enumerate(tqdm(testloader)):
            if set == 'CUB':
                images, labels, boxes, scale = data
            else:
                images, labels = data

            labels_1, labels_2 = torch.split(labels, 1, dim=1)
            labels_1 = torch.flatten(labels_1)
            labels_2 = torch.flatten(labels_2)

            images = images.cuda()
            labels_1 = labels_1.long().cuda()
            labels_2 = labels_2.long().cuda()

            proposalN_windows_score, proposalN_windows_logits_1, proposalN_windows_logits_2, indices, \
            window_scores, coordinates, raw_logits_1, raw_logits_2, local_logits_1, local_logits_2, local_imgs = model(images, epoch, i, status)

            raw_loss_1 = criterion(raw_logits_1, labels_1)
            raw_loss_2 = criterion(raw_logits_2, labels_2)
            local_loss_1 = criterion(local_logits_1, labels_1)
            local_loss_2 = criterion(local_logits_2, labels_2)
            windowscls_loss_1 = criterion(proposalN_windows_logits_1, labels_1.unsqueeze(1).repeat(1, proposalN).view(-1))
            windowscls_loss_2 = criterion(proposalN_windows_logits_2, labels_2.unsqueeze(1).repeat(1, proposalN).view(-1))

            total_loss = raw_loss_1 + raw_loss_2 + local_loss_1 + local_loss_2 + windowscls_loss_1 + windowscls_loss_2

            raw_loss_sum_1 += raw_loss_1.item()
            raw_loss_sum_2 += raw_loss_2.item()
            local_loss_sum_1 += local_loss_1.item()
            local_loss_sum_2 += local_loss_2.item()
            windowscls_loss_sum_1 += windowscls_loss_1.item()
            windowscls_loss_sum_2 += windowscls_loss_2.item()

            total_loss_sum += total_loss.item()

            if set == 'CUB':
                # computer resized coordinates of boxes
                boxes_coor = boxes.float()
                resized_boxes = torch.cat([(boxes_coor[:,0] * scale[:, 0]).unsqueeze(1) ,(boxes_coor[:,1] * scale[:, 1]).unsqueeze(1),
                                           (boxes_coor[:,2] * scale[:, 0]).unsqueeze(1), (boxes_coor[:,3] * scale[:, 1]).unsqueeze(1)], dim=1)
                resized_coor = torch.cat([resized_boxes[:,0].unsqueeze(1) ,resized_boxes[:,1].unsqueeze(1),
                                           (resized_boxes[:,0] + resized_boxes[:,2]).unsqueeze(1), (resized_boxes[:,1]+resized_boxes[:,3]).unsqueeze(1)], dim=1).round().int()


                iou = calculate_iou(coordinates.cpu().numpy(), resized_coor.numpy())
                iou_corrects += np.sum(iou >= 0.5)

            # correct num
            # raw
            pred_1 = raw_logits_1.max(1, keepdim=True)[1]
            pred_2 = raw_logits_2.max(1, keepdim=True)[1]
            raw_correct_1 += pred_1.eq(labels_1.view_as(pred_1)).sum().item()
            raw_correct_2 += pred_2.eq(labels_2.view_as(pred_2)).sum().item()
            # local
            pred_1 = local_logits_1.max(1, keepdim=True)[1]
            pred_2 = local_logits_2.max(1, keepdim=True)[1]
            local_correct_1 += pred_1.eq(labels_1.view_as(pred_1)).sum().item()
            local_correct_2 += pred_2.eq(labels_2.view_as(pred_2)).sum().item()

            # raw branch tensorboard
            if i == 0:
                if set == 'CUB':
                    box_coor = resized_coor[:vis_num].numpy()
                    pred_coor = coordinates[:vis_num].cpu().numpy()
                    with SummaryWriter(log_dir=os.path.join(save_path, 'log'), comment=status + 'raw') as writer:
                        cat_imgs = []
                        for j, coor in enumerate(box_coor):
                            img = image_with_boxes(images[j], [coor])
                            img = image_with_boxes(img, [pred_coor[j]], color=(0, 255, 0))
                            cat_imgs.append(img)
                        cat_imgs = np.concatenate(cat_imgs, axis=1)
                        writer.add_images(status + '/' + 'raw image with boxes', cat_imgs, epoch, dataformats='HWC')

            # object branch tensorboard
            if i == 0:
                indices_ndarray = indices[:vis_num,:proposalN].cpu().numpy()
                with SummaryWriter(log_dir=os.path.join(save_path, 'log'), comment=status + 'object') as writer:
                    cat_imgs = []
                    for j, indice_ndarray in enumerate(indices_ndarray):
                        img = image_with_boxes(local_imgs[j], coordinates_cat[indice_ndarray])
                        cat_imgs.append(img)
                    cat_imgs = np.concatenate(cat_imgs, axis=1)
                    writer.add_images(status + '/' + 'object image with windows', cat_imgs, epoch, dataformats='HWC')

            # if status == 'train':
            #     if i >= 2 :
            #         break

    raw_loss_avg_1 = raw_loss_sum_1 / (i+1)
    raw_loss_avg_2 = raw_loss_sum_2 / (i+1)
    local_loss_avg_1 = local_loss_sum_1 / (i+1)
    local_loss_avg_2 = local_loss_sum_2 / (i+1)
    windowscls_loss_avg_1 = windowscls_loss_sum_1 / (i+1)
    windowscls_loss_avg_2 = windowscls_loss_sum_2 / (i+1)
    total_loss_avg = total_loss_sum / (i+1)

    raw_accuracy_1 = raw_correct_2 / len(testloader.dataset)
    raw_accuracy_2 = raw_correct_2 / len(testloader.dataset)
    local_accuracy_1 = local_correct_1 / len(testloader.dataset)
    local_accuracy_2 = local_correct_2 / len(testloader.dataset)


    return raw_loss_avg_1, raw_loss_avg_2, windowscls_loss_avg_1, windowscls_loss_avg_2, total_loss_avg, raw_accuracy_1, raw_accuracy_2, local_accuracy_1, local_accuracy_2, local_loss_avg_1, local_loss_avg_2