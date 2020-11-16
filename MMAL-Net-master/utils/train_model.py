import os
import glob
import torch
from tqdm import tqdm
from tensorboardX import SummaryWriter
from config import max_checkpoint_num, proposalN, eval_trainset, set, patience, multitask
from utils.eval_model import eval, eval_multitask

def train(model,
          trainloader,
          testloader,
          criterion,
          optimizer,
          scheduler,
          save_path,
          start_epoch,
          end_epoch,
          patience_counter,
          save_interval):

    best_acc_so_far = 0 # local_accuracy

    for epoch in range(1, end_epoch + 1):
        if (epoch <= start_epoch): # to recreate the shuffling of data as per before training stopped
            continue
        model.train()

        print(f'Multitask: {multitask}')
        print(f'Epoch: {epoch}, patience counter: {patience_counter}')

        lr = next(iter(optimizer.param_groups))['lr']

        for i, data in enumerate(tqdm(trainloader)):
            if set == 'CUB':
                images, labels, _, _ = data
            else:
                images, labels = data
            images, labels = images.cuda(), labels.long().cuda()

            optimizer.zero_grad()

            proposalN_windows_score, proposalN_windows_logits, indices, \
            window_scores, _, raw_logits, local_logits, _ = model(images, epoch, i, 'train')

            raw_loss = criterion(raw_logits, labels)
            local_loss = criterion(local_logits, labels)
            windowscls_loss = criterion(proposalN_windows_logits,
                               labels.unsqueeze(1).repeat(1, proposalN).view(-1))

            if epoch < 2:
                total_loss = raw_loss
            else:
                total_loss = raw_loss + local_loss + windowscls_loss

            total_loss.backward()

            optimizer.step()

        scheduler.step()

        # code to obtain training accuracy after every epoch; eval mode is set and the losses are printed out
        if eval_trainset:
            raw_loss_avg, windowscls_loss_avg, total_loss_avg, raw_accuracy, local_accuracy, local_loss_avg\
                = eval(model, trainloader, criterion, 'train', save_path, epoch)

            print(
                'Train set: raw accuracy: {:.2f}%, local accuracy: {:.2f}%'.format(
                    100. * raw_accuracy, 100. * local_accuracy))

            # tensorboard
            with SummaryWriter(log_dir=os.path.join(save_path, 'log'), comment='train') as writer:
                writer.add_scalar('Train/learning rate', lr, epoch)
                writer.add_scalar('Train/raw_accuracy', raw_accuracy, epoch)
                writer.add_scalar('Train/local_accuracy', local_accuracy, epoch)
                writer.add_scalar('Train/raw_loss_avg', raw_loss_avg, epoch)
                writer.add_scalar('Train/local_loss_avg', local_loss_avg, epoch)
                writer.add_scalar('Train/windowscls_loss_avg', windowscls_loss_avg, epoch)
                writer.add_scalar('Train/total_loss_avg', total_loss_avg, epoch)

        # eval testset
        raw_loss_avg, windowscls_loss_avg, total_loss_avg, raw_accuracy, local_accuracy, \
        local_loss_avg\
            = eval(model, testloader, criterion, 'test', save_path, epoch)

        print(
            'Test set: raw accuracy: {:.2f}%, local accuracy: {:.2f}%'.format(
                100. * raw_accuracy, 100. * local_accuracy))

        # tensorboard
        with SummaryWriter(log_dir=os.path.join(save_path, 'log'), comment='test') as writer:
            writer.add_scalar('Test/raw_accuracy', raw_accuracy, epoch)
            writer.add_scalar('Test/local_accuracy', local_accuracy, epoch)
            writer.add_scalar('Test/raw_loss_avg', raw_loss_avg, epoch)
            writer.add_scalar('Test/local_loss_avg', local_loss_avg, epoch)
            writer.add_scalar('Test/windowscls_loss_avg', windowscls_loss_avg, epoch)
            writer.add_scalar('Test/total_loss_avg', total_loss_avg, epoch)

        if (patience == 0):
            pass
        else:
            # save checkpoint only if its the best model so far
            if (local_accuracy > best_acc_so_far):
                # if acc is the best so far, update
                best_acc_so_far = local_accuracy # validation acc; training acc variable is overwritten

                # save checkpoint model
                print('Saving checkpoint')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'learning_rate': lr,
                    'patience_counter': patience_counter
                }, os.path.join(save_path, 'epoch' + str(epoch) + '.pth'))
                
                patience_counter = 0

            else:
                patience_counter += 1

            print(f'Patience counter: {patience_counter}')
            
            if (patience_counter > patience):
                print(f'Early Stopping at epoch {epoch}; hit patience {patience_counter}\n')
                break

        # Limit the number of checkpoints to less than or equal to max_checkpoint_num,
        # and delete the redundant ones
        checkpoint_list = [os.path.basename(path) for path in glob.glob(os.path.join(save_path, '*.pth'))]
        if len(checkpoint_list) == max_checkpoint_num + 1:
            idx_list = [int(name.replace('epoch', '').replace('.pth', '')) for name in checkpoint_list]
            min_idx = min(idx_list)
            os.remove(os.path.join(save_path, 'epoch' + str(min_idx) + '.pth'))

def train_multitask(model,
          trainloader,
          testloader,
          criterion,
          optimizer,
          scheduler,
          save_path,
          start_epoch,
          end_epoch,
          patience_counter,
          save_interval):

    best_acc_so_far = 0 # local_accuracy_1

    for epoch in range(1, end_epoch + 1):
        if (epoch <= start_epoch): # to recreate the shuffling of data as per before training stopped
            continue
        model.train()

        print(f'Multitask: {multitask}')
        print(f'Epoch: {epoch}, patience counter: {patience_counter}')

        lr = next(iter(optimizer.param_groups))['lr']

        for i, data in enumerate(tqdm(trainloader)):
            if set == 'CUB':
                images, labels, _, _ = data
            else:
                images, labels = data

            labels_1, labels_2 = torch.split(labels, 1, dim=1)
            labels_1 = torch.flatten(labels_1)
            labels_2 = torch.flatten(labels_2)

            images, labels_1, labels_2 = images.cuda(), labels_1.long().cuda(), labels_2.long().cuda()

            optimizer.zero_grad()

            proposalN_windows_score, proposalN_windows_logits_1, proposalN_windows_logits_2, indices, \
            window_scores, _, raw_logits_1, raw_logits_2, local_logits_1, local_logits_2, _ = model(images, epoch, i, 'train')

            raw_loss_1 = criterion(raw_logits_1, labels_1)
            raw_loss_2 = criterion(raw_logits_2, labels_2)
            local_loss_1 = criterion(local_logits_1, labels_1)
            local_loss_2 = criterion(local_logits_2, labels_2)
            windowscls_loss_1 = criterion(proposalN_windows_logits_1, labels_1.unsqueeze(1).repeat(1, proposalN).view(-1))
            windowscls_loss_2 = criterion(proposalN_windows_logits_2, labels_2.unsqueeze(1).repeat(1, proposalN).view(-1))

            if epoch < 2:
                total_loss = raw_loss_1 + raw_loss_2
            else:
                total_loss = raw_loss_1 + raw_loss_2 + local_loss_1 + local_loss_2 + windowscls_loss_1 + windowscls_loss_2

            total_loss.backward()

            optimizer.step()

        scheduler.step()

        # code to obtain training accuracy after every epoch; eval mode is set and the losses are printed out
        if eval_trainset:
            raw_loss_avg_1, raw_loss_avg_2, windowscls_loss_avg_1, windowscls_loss_avg_2, total_loss_avg, raw_accuracy_1, raw_accuracy_2, local_accuracy_1, local_accuracy_2, local_loss_avg_1, local_loss_avg_2 = eval_multitask(model, trainloader, criterion, 'train', save_path, epoch)

            print('Train set: raw accuracy_1: {:.2f}%, local accuracy_1: {:.2f}%'.format(100. * raw_accuracy_1, 100. * local_accuracy_1))
            print('Train set: raw accuracy_2: {:.2f}%, local accuracy_2: {:.2f}%'.format(100. * raw_accuracy_2, 100. * local_accuracy_2))

            # tensorboard
            with SummaryWriter(log_dir=os.path.join(save_path, 'log'), comment='train') as writer:
                writer.add_scalar('Train/learning rate', lr, epoch)
                writer.add_scalar('Train/raw_accuracy_1', raw_accuracy_1, epoch)
                writer.add_scalar('Train/raw_accuracy_2', raw_accuracy_2, epoch)
                writer.add_scalar('Train/local_accuracy_1', local_accuracy_1, epoch)
                writer.add_scalar('Train/local_accuracy_2', local_accuracy_2, epoch)
                writer.add_scalar('Train/raw_loss_avg_1', raw_loss_avg_1, epoch)
                writer.add_scalar('Train/raw_loss_avg_2', raw_loss_avg_2, epoch)
                writer.add_scalar('Train/local_loss_avg_1', local_loss_avg_1, epoch)
                writer.add_scalar('Train/local_loss_avg_2', local_loss_avg_2, epoch)
                writer.add_scalar('Train/windowscls_loss_avg_1', windowscls_loss_avg_1, epoch)
                writer.add_scalar('Train/windowscls_loss_avg_2', windowscls_loss_avg_2, epoch)
                writer.add_scalar('Train/total_loss_avg', total_loss_avg, epoch)

        # eval testset
        raw_loss_avg_1, raw_loss_avg_2, windowscls_loss_avg_1, windowscls_loss_avg_2, total_loss_avg, raw_accuracy_1, raw_accuracy_2, local_accuracy_1, local_accuracy_2, local_loss_avg_1, local_loss_avg_2 = eval_multitask(model, testloader, criterion, 'test', save_path, epoch)

        print('Test set: raw accuracy_1: {:.2f}%, local accuracy_1: {:.2f}%'.format(100. * raw_accuracy_1, 100. * local_accuracy_1))
        print('Test set: raw accuracy_2: {:.2f}%, local accuracy_2: {:.2f}%'.format(100. * raw_accuracy_2, 100. * local_accuracy_2))

        # tensorboard
        with SummaryWriter(log_dir=os.path.join(save_path, 'log'), comment='test') as writer:
            writer.add_scalar('Test/raw_accuracy_1', raw_accuracy_1, epoch)
            writer.add_scalar('Test/raw_accuracy_2', raw_accuracy_2, epoch)
            writer.add_scalar('Test/local_accuracy_1', local_accuracy_1, epoch)
            writer.add_scalar('Test/local_accuracy_2', local_accuracy_2, epoch)
            writer.add_scalar('Test/raw_loss_avg_1', raw_loss_avg_1, epoch)
            writer.add_scalar('Test/raw_loss_avg_2', raw_loss_avg_2, epoch)
            writer.add_scalar('Test/local_loss_avg_1', local_loss_avg_1, epoch)
            writer.add_scalar('Test/local_loss_avg_2', local_loss_avg_2, epoch)
            writer.add_scalar('Test/windowscls_loss_avg_1', windowscls_loss_avg_1, epoch)
            writer.add_scalar('Test/windowscls_loss_avg_2', windowscls_loss_avg_2, epoch)
            writer.add_scalar('Test/total_loss_avg', total_loss_avg, epoch)

        if (patience == 0):
            pass
        else:
            # save checkpoint only if its the best model so far
            if (local_accuracy_1 > best_acc_so_far):
                # if acc is the best so far, update
                best_acc_so_far = local_accuracy_1 # validation acc; training acc variable is overwritten

                # save checkpoint model
                print('Saving checkpoint')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'learning_rate': lr,
                    'patience_counter': patience_counter
                }, os.path.join(save_path, 'epoch' + str(epoch) + '.pth'))
                patience_counter = 0

            else:
                patience_counter += 1

            print(f'Patience counter: {patience_counter}')
            
            if (patience_counter > patience):
                print(f'Early Stopping at epoch {epoch}.\n')
                break

        # Limit the number of checkpoints to less than or equal to max_checkpoint_num,
        # and delete the redundant ones
        checkpoint_list = [os.path.basename(path) for path in glob.glob(os.path.join(save_path, '*.pth'))]
        if len(checkpoint_list) == max_checkpoint_num + 1:
            idx_list = [int(name.replace('epoch', '').replace('.pth', '')) for name in checkpoint_list]
            min_idx = min(idx_list)
            os.remove(os.path.join(save_path, 'epoch' + str(min_idx) + '.pth'))