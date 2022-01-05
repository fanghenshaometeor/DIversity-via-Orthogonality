"""
Created on Mon Feb 24 2020

@author: fanghenshao

"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

import os
import ast
import copy
import time
import random
import argparse
import numpy as np

from utils import setup_seed
from utils import get_datasets, get_model
from utils import AverageMeter, accuracy


from advertorch.attacks import LinfPGDAttack
from advertorch.context import ctx_noparamgrad_and_eval

# ======== fix data type ========
torch.set_default_tensor_type(torch.FloatTensor)

# ======== options ==============
parser = argparse.ArgumentParser(description='Training Enhanced OMP')
# -------- file param. --------------
parser.add_argument('--data_dir',type=str,default='/media/Disk1/KunFang/data/CIFAR10/',help='file path for data')
parser.add_argument('--model_dir',type=str,default='./save/',help='file path for saving model')
parser.add_argument('--logs_dir',type=str,default='./runs/',help='log path')
parser.add_argument('--dataset',type=str,default='CIFAR10',help='data set name')
parser.add_argument('--arch',type=str,default='vgg16',help='model architecture')
# -------- training param. ----------
parser.add_argument('--batch_size',type=int,default=256,help='batch size for training (default: 256)')    
parser.add_argument('--epochs',type=int,default=200,help='number of epochs to train (default: 200)')
parser.add_argument('--save_freq',type=int,default=20,help='model save frequency (default: 20 epoch)')
# -------- enable adversarial training --------
parser.add_argument('--adv_train',type=ast.literal_eval,dest='adv_train',help='enable the adversarial training')
# -------- hyper parameters -------
parser.add_argument('--alpha',type=float,default=0.1,help='coefficient of the orthogonality regularization term')
parser.add_argument('--beta',type=float,default=0.1,help='coefficient of the margin regularization term')
parser.add_argument('--num_heads',type=int,default=10,help='number of orthogonal paths')
parser.add_argument('--tau',type=float,default=0.2,help='upper bound of the margin')
parser.add_argument('--tau_adv',type=float,default=0.2,help='upper bound of the margin for adversarial training')
# -------- enable adversarial training --------
parser.add_argument('--adv_train',type=ast.literal_eval,dest='adv_train',help='enable the adversarial training')
parser.add_argument('--train_eps', default=8., type=float, help='epsilon of attack during training')
parser.add_argument('--train_step', default=10, type=int, help='itertion number of attack during training')
parser.add_argument('--train_gamma', default=2., type=float, help='step size of attack during training')
parser.add_argument('--test_eps', default=8., type=float, help='epsilon of attack during testing')
parser.add_argument('--test_step', default=20, type=int, help='itertion number of attack during testing')
parser.add_argument('--test_gamma', default=2., type=float, help='step size of attack during testing')
args = parser.parse_args()

# ======== initialize log writer
if args.adv_train == True:
    writer = SummaryWriter(os.path.join(args.logs_dir, args.dataset, args.arch+'-adv', \
        'p-'+str(args.num_heads)+'-a-'+str(args.alpha)+'-b-'+str(args.beta)+ \
        '-tau-'+str(args.tau)+'-taua-'+str(args.tau_adv)+'/'))
    # --------
    if not os.path.exists(os.path.join(args.model_dir,args.dataset,args.arch+'-adv')):
        os.makedirs(os.path.join(args.model_dir,args.dataset,args.arch+'-adv'))
    # --------
    model_name = 'p-'+str(args.num_heads) \
        +'-a-'+str(args.alpha)+'-b-'+str(args.beta)+ \
        '-tau-'+str(args.tau)+'-taua-'+str(args.tau_adv)+'.pth'
    # --------
    args.save_path = os.path.join(args.model_dir,args.dataset,args.arch+'-adv',model_name)
    # writer_dir = args.model+'-p-'+str(args.num_heads)+ \
    # '-a-'+str(args.alpha)+'-b-'+str(args.beta)+ \
    # '-tau-'+str(args.tau)+'-taua-'+str(args.tau_adv)+'-adv'
else:
    writer = SummaryWriter(os.path.join(args.logs_dir, args.dataset, args.arch, \
        'p-'+str(args.num_heads)+'-a-'+str(args.alpha)+'-b-'+str(args.beta)+ \
        '-tau-'+str(args.tau)+'/'))
    # --------
    if not os.path.exists(os.path.join(args.model_dir,args.dataset,args.arch)):
        os.makedirs(os.path.join(args.model_dir,args.dataset,args.arch))
    # --------
    model_name = 'p-'+str(args.num_heads) \
        +'-a-'+str(args.alpha)+'-b-'+str(args.beta)+ '-tau-'+str(args.tau)+'.pth'
    # --------
    args.save_path = os.path.join(args.model_dir,args.dataset,args.arch,model_name)
    # writer_dir = args.model+'-p-'+str(args.num_heads)+ \
    # '-a-'+str(args.alpha)+'-b-'+str(args.beta)+ \
    # '-tau-'+str(args.tau)
# writer = SummaryWriter(os.path.join(args.logs_dir, args.dataset, writer_dir+'/'))

# -------- main function
def main():
    
    # ======== fix random seed ========
    setup_seed(666)

    # ======== get data set =============
    trainloader, testloader = get_datasets(args)
    print('-------- DATA INFOMATION --------')
    print('---- dataset: '+args.dataset)

    # ======== initialize net
    backbone, head = get_model(args)
    backbone, head = backbone.cuda(), head.cuda()
    # if args.adv_train:
    #     model_name = args.model+'-p-'+str(args.num_heads) \
    #     +'-a-'+str(args.alpha)+'-b-'+str(args.beta)+ \
    #     '-tau-'+str(args.tau)+'-taua-'+str(args.tau_adv)+'-adv.pth'
    #     args.model_path = os.path.join(args.model_dir, args.dataset, model_name)
    # else:
    #     model_name = args.model+'-p-'+str(args.num_heads) \
    #     +'-a-'+str(args.alpha)+'-b-'+str(args.beta)+ \
    #     '-tau-'+str(args.tau)+'.pth'
    #     args.model_path = os.path.join(args.model_dir, args.dataset, model_name)
    print('-------- MODEL INFORMATION --------')
    print('---- architecture: '+args.arch)
    print('---- adv.   train: '+str(args.adv_train))
    print('---- saved path: '+args.save_path)

    # ======== set criterions & optimizers
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD([{'params':backbone.parameters()},{'params':head.parameters()}], lr=0.05, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [60,120,160], gamma=0.1)

    # ======== 
    args.train_eps /= 255.
    args.train_gamma /= 255.
    if args.adv_train:
        adversary = LinfPGDAttack(net, loss_fn=criterion, eps=args.train_eps, nb_iter=args.train_step, eps_iter=args.train_gamma, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)
    else:
        adversary = None


    print('-------- START TRAINING --------')
    for epoch in range(args.epochs):

        start = time.time()
        # -------- train
        train_epoch(backbone, head, trainloader, optimizer, criterion, epoch, adversary)

        # -------- validation
        if epoch % 20 == 0 or epoch == (args.epochs-1):
            print('evaluating...')
            corr_tr, corr_te = val(backbone, head, trainloader, testloader)
            acc_tr, acc_te = corr_tr/train_num, corr_te/test_num
            print('evaluation end.')
            

        scheduler.step()

        duration = time.time() - start


        # -------- print info. & save model
        if epoch % 20 == 0 or epoch == (args.epochs-1):
            print('     train acc. on each path: ', acc_tr)
            print('     test  acc. on each path: ', acc_te)
            checkpoint = {'state_dict_backbone': backbone.state_dict(), 'state_dict_head': head.state_dict()}
            torch.save(checkpoint, args.model_path)

        print('Epoch %d/%d in training costs %fs:' % (epoch, args.epochs, duration))
        print('Current training model: ', args.model_path)
    
    print('-------- TRAINING finished.')

# ======== train  model ========
def train_epoch(backbone, head, trainloader, optim, criterion, epoch):
    
    backbone.train()
    head.train()

    batch_time = AverageMeter()
    losses_ce, losses_ortho, losses_margin = AverageMeter(), AverageMeter(), AverageMeter()

    end = time.time()
    for batch_idx, (b_data, b_label) in enumerate(trainloader):
        
        # -------- move to gpu
        b_data, b_label = b_data.cuda(), b_label.cuda()

        # -------- forward and compute loss
        total_loss = .0

        # -------- compute CROSS ENTROPY loss
        loss_ce = .0
        all_logits = head(backbone(b_data), 'all')
        for idx in range(args.num_heads):
            logits = all_logits[idx]
            loss = criterion(logits, b_label)
            loss_ce += 1/args.num_heads * loss                      # sum the weighted loss for backward propagation
            avg_loss_ce[idx] = avg_loss_ce[idx] + loss.item()       # save the loss value
    
        # -------- compute the ORTHOGONALITY constraint
        loss_ortho = .0
        if args.num_heads > 1:
            loss_ortho = head._orthogonal_costr()
            # avg_loss_ortho = avg_loss_ortho + reduce_tensor(loss_ortho.data).item()
            avg_loss_ortho = avg_loss_ortho + loss_ortho.item()
        else:
            loss_ortho = 0
        
        # -------- find correctly-classified samples-clfs (x, clf-i)
        # -------- compute MARGIN loss
        # -------- 对每条路径遍历
        loss_margin = .0
        for idx in range(args.num_heads):           # 对每条路径
            logits = all_logits[idx]                # 读取当前路径对当前 batch 数据的预测 logits
            _, pred = torch.max(logits.data, 1)     # 获取当前路径对当前 batch 数据的预测结果
            b_corr_idx = (pred == b_label)          # 获取当前路径预测正确的数据的索引

            if b_corr_idx.any() == False:           # 如果当前路径对 batch 数据全部预测错误，即，b_data_idx 全部 false
                continue

            b_logits_correct = logits[b_corr_idx,:]                                                         # 依据索引，读取当前路径预测正确的那些数据的 logits
            b_corr_num = b_logits_correct.size(0)
            b_logits_correct_max, _ = torch.max(b_logits_correct, 1)                                        # 获取预测正确数据的 logits 的最大值，即，groundtruth 所在类的 logits
            b_logits_correct_max = b_logits_correct_max.unsqueeze(1).expand(b_corr_num, args.num_classes)   # 最大值复制

            hyperplane_norm = head._compute_l2_norm_specified(idx)                                              # 获取当前路径的超平面的 l2 norm
            hyperplane_norm = hyperplane_norm.repeat(b_corr_num,1)                                              # 数据复制，方便计算距离

            b_distance = torch.div(torch.abs(b_logits_correct-b_logits_correct_max), hyperplane_norm)           # 计算到最大值的 distance，这其中存在 0 值 
            b_distance = torch.where(b_distance>0, b_distance, torch.tensor(1000.0).cuda())                     # 去除 0 值
            b_margin, _ = torch.min(b_distance, 1)                                                              # 获取 margin

            loss_margin += (args.tau - b_margin).clamp(min=0).mean()                # 计算 margin loss
            avg_loss_margin += loss_margin.item()
        
        total_loss = loss_ce + args.alpha * loss_ortho + args.beta * loss_margin

        # -------- print info. at the last batch
        if batch_idx == (len(trainloader)-1):
            avg_loss_ce = avg_loss_ce / len(trainloader)
            avg_loss_ortho = avg_loss_ortho / len(trainloader)
            avg_loss_margin = avg_loss_margin / len(trainloader)
        
            # -------- record the cross entropy loss of each path
            loss_path_ce_record = {}
            for idx in range(args.num_heads):
                loss_path_ce_record['path-%d'%idx] = avg_loss_ce[idx]
            loss_path_ce_record['avg.'] = avg_loss_ce.mean()
            writer.add_scalars('loss-ce', loss_path_ce_record, epoch)
        
            # -------- record the orthgonality loss
            writer.add_scalar('loss-ortho', avg_loss_ortho, epoch)

            # -------- record the margin loss
            writer.add_scalar('loss-margin', avg_loss_margin, epoch)

            # -------- print in terminal
            print('Epoch %d/%d CLEAN samples:'%(epoch, args.epochs))
            print('     CE     loss of each path: ', avg_loss_ce)
            print('     ORTHO  loss = %f.'%avg_loss_ortho)
            print('     MARGIN loss = %f.'%avg_loss_margin)

        # -------- backprop. & update
        optim.zero_grad()
        total_loss.backward()
        optim.step()

        # -------- training with adversarial examples
        if args.adv_train and epoch >= args.adv_delay:
            backbone.eval()
            head.eval()
            perturbed_data = pgd_attack(backbone, head, b_data, b_label, eps=0.031, alpha=0.01, iters=7)
            backbone.train()
            head.train()

            loss_ce = .0
            all_logits = head(backbone(perturbed_data), 'all')
            for idx in range(args.num_heads):
                logits = all_logits[idx]
                loss = criterion(logits, b_label)
                loss_ce += 1/args.num_heads * loss         # sum the weighted loss for backward propagation 
                avg_loss_ce_adv[idx] = avg_loss_ce_adv[idx] + loss.item()       # save the loss value 
            
            # ------- compute the orthogonal constraint
            loss_ortho = .0
            if args.num_heads > 1:
                loss_ortho = head._orthogonal_costr()
                # avg_loss_ortho_adv = avg_loss_ortho_adv + reduce_tensor(loss_ortho.data).item()
                avg_loss_ortho_adv = avg_loss_ortho_adv + loss_ortho.item()
            else:
                loss_ortho = 0

            # -------- find correctly-classified samples-clfs (x, clf-i)
            # -------- compute distance/margin
            # -------- 对每条路径遍历
            loss_margin = .0
            for idx in range(args.num_heads):           # 对每条路径
                logits = all_logits[idx]                # 读取当前路径对当前 batch 数据的预测 logits
                _, pred = torch.max(logits.data, 1)     # 获取当前路径对当前 batch 数据的预测结果
                b_corr_idx = (pred == b_label)          # 获取当前路径预测正确的那些数据的索引

                if b_corr_idx.any() == False:           # 如果当前路径对 batch 数据全部预测错误，即，b_data_idx 全部 false
                    continue

                b_logits_correct = logits[b_corr_idx,:] # 依据索引，读取当前路径预测正确的那些数据的 logits
                b_corr_num = b_logits_correct.size(0)

                b_logits_correct_max, _ = torch.max(b_logits_correct, 1)                                        # 获取预测正确数据的 logits 的最大值
                b_logits_correct_max = b_logits_correct_max.unsqueeze(1).expand(b_corr_num, args.num_classes)   # 最大值复制

                hyperplane_norm = head._compute_l2_norm_specified(idx)                                              # 获取当前路径的超平面的 l2 norm
                hyperplane_norm = hyperplane_norm.repeat(b_corr_num,1)                                              # 数据复制，方便计算距离

                b_distance = torch.div(torch.abs(b_logits_correct-b_logits_correct_max), hyperplane_norm)           # 计算 distance
                b_distance = torch.where(b_distance>0, b_distance, torch.tensor(1000.0).cuda())                     # 去除 0 值
                b_margin, _ = torch.min(b_distance, 1)                                                              # 获取 margin

                loss_margin += (args.tau_adv - b_margin).clamp(min=0).mean()                # 计算 margin loss
                avg_loss_margin_adv += loss_margin.item()
            
            total_loss = loss_ce + args.alpha * loss_ortho + args.beta * loss_margin          
            
            if batch_idx == (len(trainloader)-1):
                avg_loss_ce_adv = avg_loss_ce_adv / len(trainloader)
                avg_loss_ortho_adv = avg_loss_ortho_adv / len(trainloader)
                avg_loss_margin_adv = avg_loss_margin_adv / len(trainloader)
            
                # -------- record the cross entropy loss of each path
                loss_path_ce_record = {}
                for idx in range(args.num_heads):
                    loss_path_ce_record['path-%d'%idx] = avg_loss_ce_adv[idx]
                loss_path_ce_record['avg.'] = avg_loss_ce_adv.mean()
                writer.add_scalars('loss-ce-adv', loss_path_ce_record, epoch)
            
                # -------- record the orthgonality loss
                writer.add_scalar('loss-ortho-adv', avg_loss_ortho_adv, epoch)

                # -------- record the margin loss
                writer.add_scalar('loss-margin-adv', avg_loss_margin_adv, epoch)

                # -------- print in terminal
                print('Epoch %d/%d ADVERSARIAL examples:'%(epoch, args.epochs))
                print('     CE     loss of each path: ', avg_loss_ce_adv)
                print('     ORTHO  loss = %f.'%avg_loss_ortho_adv)
                print('     MARGIN loss = %f.'%avg_loss_margin_adv)

            optim.zero_grad()
            total_loss.backward()
            optim.step()

    return

# ======== evaluate model ========
def val(backbone, head, trainloader, testloader):
    
    backbone.eval()
    head.eval()

    correct_train, correct_test = np.zeros(args.num_heads), np.zeros(args.num_heads)
    
    with torch.no_grad():
        
        # -------- compute the accs. of train, test set
        for test in testloader:
            images, labels = test
            images, labels = images.cuda(), labels.cuda()

            # ------- forward 
            all_logits = head(backbone(images), 'all')
            for idx in range(args.num_heads):
                logits = all_logits[idx]
                logits = logits.detach()
                _, pred = torch.max(logits.data, 1)
                correct_test[idx] += (pred == labels).sum().item()
            
        
        for train in trainloader:
            images, labels = train
            images, labels = images.cuda(), labels.cuda()

            all_logits = head(backbone(images), 'all')
            for idx in range(args.num_heads):
                logits = all_logits[idx]
                logits = logits.detach()
                _, pred = torch.max(logits.data, 1)
                correct_train[idx] += (pred == labels).sum().item()
        
    return correct_train, correct_test


# ======== startpoint
if __name__ == '__main__':
    main()