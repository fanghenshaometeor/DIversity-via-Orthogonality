"""
Created on Mon Feb 24 2020

@author: fanghenshao

"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

import os
import ast
import copy
import time
import random
import argparse
import numpy as np
from itertools import chain

from utils import setup_seed
from attackers import pgd_attack

# ======== fix data type ========
torch.set_default_tensor_type(torch.FloatTensor)

# ======== fix seed =============
setup_seed(666)

# ======== options ==============
parser = argparse.ArgumentParser(description='Training Enhanced OMP')
# -------- file param. --------------
parser.add_argument('--data_dir',type=str,default='/media/Disk1/KunFang/data/CIFAR10/',help='file path for data')
parser.add_argument('--model_dir',type=str,default='./save/',help='file path for saving model')
parser.add_argument('--logs_dir',type=str,default='./runs/',help='log path')
parser.add_argument('--dataset',type=str,default='CIFAR10',help='data set name')
parser.add_argument('--model',type=str,default='vgg16',help='model name')
# -------- training param. ----------
parser.add_argument('--batch_size',type=int,default=512,help='batch size for training (default: 256)')    
parser.add_argument('--epochs',type=int,default=200,help='number of epochs to train (default: 200)')
# -------- enable adversarial training --------
parser.add_argument('--adv_train',type=ast.literal_eval,dest='adv_train',help='enable the adversarial training')
parser.add_argument('--adv_delay',type=int,default=10,help='epochs delay for adversarial training')
# -------- hyper parameters -------
parser.add_argument('--alpha',type=float,default=0.1,help='coefficient of the orthogonality regularization term')
parser.add_argument('--beta',type=float,default=0.1,help='coefficient of the margin regularization term')
parser.add_argument('--num_heads',type=int,default=10,help='number of orthogonal paths')
parser.add_argument('--num_classes',type=int,default=10,help='number of classes')
parser.add_argument('--tau',type=float,default=0.2,help='upper bound of the margin')
parser.add_argument('--tau_adv',type=float,default=0.2,help='upper bound of the margin for adversarial training')
args = parser.parse_args()

# ======== initialize log writer
if args.adv_train == True:
    writer_dir = args.model+'-p-'+str(args.num_heads)+ \
    '-a-'+str(args.alpha)+'-b-'+str(args.beta)+ \
    '-tau-'+str(args.tau)+'-taua-'+str(args.tau_adv)+'-adv'
else:
    writer_dir = args.model+'-p-'+str(args.num_heads)+ \
    '-a-'+str(args.alpha)+'-b-'+str(args.beta)+ \
    '-tau-'+str(args.tau)
writer = SummaryWriter(os.path.join(args.logs_dir, args.dataset, writer_dir+'/'))

# -------- main function
def main():
    
    # ======== data set preprocess =============
    if args.dataset == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor()
        ])
        trainset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
    elif args.dataset == 'CIFAR100':
        args.num_classes = 100
        args.batch_size = 512
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor()
        ])
        trainset = datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=transform_train)
        testset = datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transform_test)
    elif args.dataset == 'SVHN':
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        trainset = datasets.SVHN(root=args.data_dir, split='train', download=True, 
                            transform=transform)
        testset = datasets.SVHN(root=args.data_dir, split='test', download=True, 
                            transform=transform)
    elif args.dataset == 'STL10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(96, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            ])
        trainset = datasets.STL10(root=args.data_dir, split='train', transform=transform_train, download=True)
        testset = datasets.STL10(root=args.data_dir, split='test', transform=transform_test, download=True)
    elif args.dataset == 'FMNIST':
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor()
        ])
        trainset = datasets.FashionMNIST(root=args.data_dir, train=True, download=True, transform=transform_train)
        testset = datasets.FashionMNIST(root=args.data_dir, train=False, download=True, transform=transform_test)
    else:
        assert False, "Unknow dataset : {}".format(args.dataset)
    
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    testloader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    train_num, test_num = len(trainset), len(testset)
    print('-------- DATA INFOMATION --------')
    print('---- dataset: '+args.dataset)
    print('---- #train : %d'%train_num)
    print('---- #test  : %d'%test_num)

    # ======== initialize net
    if args.model == 'vgg11':
        from model.dio_vgg import vgg11_bn
        backbone, head = vgg11_bn(embedding_size=512, num_classes=args.num_classes, num_classifiers=args.num_heads)
    elif args.model == 'vgg13':
        from model.dio_vgg import vgg13_bn
        backbone, head = vgg13_bn(embedding_size=512, num_classes=args.num_classes, num_classifiers=args.num_heads)
    elif args.model == 'vgg16':
        from model.dio_vgg import vgg16_bn
        backbone, head = vgg16_bn(embedding_size=512, num_classes=args.num_classes, num_classifiers=args.num_heads)
    elif args.model == 'vgg19':
        from model.dio_vgg import vgg19_bn
        backbone, head = vgg19_bn(embedding_size=512, num_classes=args.num_classes, num_classifiers=args.num_heads)
    elif args.model == 'resnet20':
        from model.dio_resnet_v1 import resnet20
        backbone, head = resnet20(num_classes=args.num_classes, num_classifiers=args.num_heads)
    elif args.model == 'resnet32':
        from model.dio_resnet_v1 import resnet32
        backbone, head = resnet32(num_classes=args.num_classes, num_classifiers=args.num_heads)
    elif args.model == 'wrn28x5':
        from model.dio_wideresnet import wideresnet_28_5
        backbone, head = wideresnet_28_5(num_classes=args.num_classes, num_classifiers=args.num_heads)
    elif args.model == 'wrn34x10':
        from model.dio_wideresnet import wideresnet_34_10
        backbone, head = wideresnet_34_10(num_classes=args.num_classes, num_classifiers=args.num_heads)
    elif args.model == 'wrn28x10':
        from model.dio_wideresnet import wideresnet_28_10
        backbone, head = wideresnet_28_10(num_classes=args.num_classes, num_classifiers=args.num_heads)
    else:
        assert False, "Unknown model : {}".format(args.model)
    backbone, head = backbone.cuda(), head.cuda()
    if args.adv_train:
        model_name = args.model+'-p-'+str(args.num_heads) \
        +'-a-'+str(args.alpha)+'-b-'+str(args.beta)+ \
        '-tau-'+str(args.tau)+'-taua-'+str(args.tau_adv)+'-adv.pth'
        args.model_path = os.path.join(args.model_dir, args.dataset, model_name)
    else:
        model_name = args.model+'-p-'+str(args.num_heads) \
        +'-a-'+str(args.alpha)+'-b-'+str(args.beta)+ \
        '-tau-'+str(args.tau)+'.pth'
        args.model_path = os.path.join(args.model_dir, args.dataset, model_name)
    print('-------- MODEL INFORMATION --------')
    print('---- model:      '+args.model)
    print('---- adv. train: '+str(args.adv_train))
    print('---- saved path: '+args.model_path)

    # ======== set criterions & optimizers
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD([{'params':backbone.parameters()},{'params':head.parameters()}], lr=0.05, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [60,120,160], gamma=0.1)

    print('-------- START TRAINING --------')

    for epoch in range(args.epochs):

        start = time.time()
        # -------- train
        train_epoch(backbone, head, trainloader, optimizer, criterion, epoch)

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

    avg_loss_ce = np.zeros(args.num_heads)
    avg_loss_ortho = 0.0
    avg_loss_margin = 0.0

    avg_loss_ce_adv = np.zeros(args.num_heads)
    avg_loss_ortho_adv = 0.0
    avg_loss_margin_adv = 0.0

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
            perturbed_data, _ = pgd_attack(backbone, head, b_data, b_label, eps=0.031, alpha=0.01, iters=7)
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