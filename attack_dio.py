"""
Created on Mon Mar 09 2020

@author: fanghenshao
"""

from __future__ import print_function


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
from torchvision import datasets, transforms
import os
import ast
import copy
import argparse

from utils import setup_seed

import numpy as np

from autoattack import AutoAttack
from attackers import fgsm_attack, pgd_attack, cw_attack

# ======== fix data type ========
torch.set_default_tensor_type(torch.FloatTensor)

# ======== fix seed =============
setup_seed(666)

# ======== options ==============
parser = argparse.ArgumentParser(description='Attack Deep Neural Networks')
# -------- file param. --------------
parser.add_argument('--data_dir',type=str,default='/media/Disk1/KunFang/data/CIFAR10/',help='file path for data')
parser.add_argument('--dataset',type=str,default='CIFAR10',help='data set name')
parser.add_argument('--model',type=str,default='vgg16',help='model architecture name')
parser.add_argument('--model_path',type=str,default='./save/CIFAR10-VGG.pth',help='saved model path')
# -------- training param. ----------
parser.add_argument('--batch_size',type=int,default=512,help='batch size for training (default: 256)')
# -------- save adv. images --------
parser.add_argument('--save_adv_img',type=ast.literal_eval,dest='save_adv_img',help='save adversarial examples')
parser.add_argument('--attack_type',type=str,default='fgsm',help='attack method')
# -------- hyper parameters -------
parser.add_argument('--num_heads',type=int,default=10,help='number of orthogonal paths')
parser.add_argument('--num_classes',type=int,default=10,help='number of classes')
args = parser.parse_args()

# -------- main function
def main():
    
    # ======== data set preprocess =============
    # ======== mean-variance normalization is removed
    if args.dataset == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        trainset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
        testset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)
    elif args.dataset == 'CIFAR100':
        args.batch_size = 128
        args.num_classes = 100
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        trainset = datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=transform)
        testset = datasets.CIFAR100(root=args.data_dir, train=False, download=True, transform=transform)
    elif args.dataset == 'SVHN':
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        trainset = datasets.SVHN(root=args.data_dir, split='train', download=True, 
                            transform=transform)
        testset = datasets.SVHN(root=args.data_dir, split='test', download=True, 
                            transform=transform)
    else:
        assert False, "Unknow dataset : {}".format(args.dataset)
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False)
    testloader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)
    train_num, test_num = len(trainset), len(testset)
    print('-------- DATA INFOMATION --------')
    print('---- dataset: '+args.dataset)
    print('---- #train : %d'%train_num)
    print('---- #test  : %d'%test_num)

    # ======== load network ========
    checkpoint = torch.load(args.model_path, map_location=torch.device("cpu"))
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
    elif args.model == 'wrn28x10':
        from model.dio_wideresnet import wideresnet_28_10
        backbone, head = wideresnet_28_10(num_classes=args.num_classes, num_classifiers=args.num_heads)
    elif args.model == 'wrn34x10':
        from model.dio_wideresnet import wideresnet_34_10
        backbone, head = wideresnet_34_10(num_classes=args.num_classes, num_classifiers=args.num_heads)
    else:
        assert False, "Unknown model : {}".format(args.model)
    backbone, head = backbone.cuda(), head.cuda()
    backbone.load_state_dict(checkpoint['state_dict_backbone'])
    head.load_state_dict(checkpoint['state_dict_head'])
    backbone.eval()
    head.eval()
    print('-------- MODEL INFORMATION --------')
    print('---- model:      '+args.model)
    print('---- saved path: '+args.model_path)

    print('-------- START TESTING --------')
    corr_tr, corr_te = evaluate(backbone, head, trainloader, testloader)
    acc_tr, acc_te = corr_tr / float(train_num), corr_te / float(test_num)
    print("acc. of each path on clean training set: ")
    print(acc_tr)
    print("--------")
    print("acc. of each path on clean test     set: ")
    print(acc_te)
    print('--------')
    print("mean/std. acc. on clean training set:")
    print(np.mean(acc_tr))
    print(np.std(acc_tr))
    print("mean/std. acc. on clean test     set:")
    print(np.mean(acc_te))
    print(np.std(acc_te)) 

    # return
    

    print('-------- START ATTACKING --------')
    attack_param = {}
    if args.attack_type == 'fgsm':
        print('-------- START FGSM ATTACK...')
        fgsm_epsilons = [1/255, 2/255, 3/255, 4/255, 5/255, 6/255, 7/255, 8/255, 9/255, 10/255, 11/255, 12/255]
        # fgsm_epsilons = [1/255, 4/255, 8/255, 12/255]
        avg_acc_fgsm = np.zeros([args.num_heads,len(fgsm_epsilons)])
        for idx in range(args.num_heads):
            print("-------- attacking network-%d..."%idx)
            acc_fgsm = []
            for eps in fgsm_epsilons:
                attack_param['eps'] = eps
                corr_te_fgsm = attack(backbone, head, idx, testloader, attack_param)
                acc_fgsm.append(corr_te_fgsm/float(test_num))

            avg_acc_fgsm[idx,:] = avg_acc_fgsm[idx,:] + np.array(acc_fgsm)
            print("acc. of path-%d under FGSM attack:"%idx)
            print(acc_fgsm)  

        print('--------')
        print("mean acc. under FGSM attack:")
        print(np.mean(avg_acc_fgsm,axis=0))
        print("std. acc. under FGSM attack:")
        print(np.std(avg_acc_fgsm,axis=0))

    elif args.attack_type == 'pgd':
        print('-------- START PGD ATTACK...')
        # pgd_epsilons = [1/255, 2/255, 3/255, 4/255, 5/255, 6/255, 7/255, 8/255, 9/255, 10/255, 11/255, 12/255]
        pgd_epsilons = [1/255, 2/255, 4/255]
        avg_acc_pgd = np.zeros([args.num_heads,len(pgd_epsilons)])
        for idx in range(args.num_heads):
            print("-------- attacking network-%d..."%idx)
            acc_pgd = []
            for eps in pgd_epsilons:
                attack_param['eps'] = eps
                corr_te_pgd = attack(backbone, head, idx, testloader, attack_param)
                acc_pgd.append(corr_te_pgd/float(test_num))

            avg_acc_pgd[idx,:] = avg_acc_pgd[idx,:] + np.array(acc_pgd)
            print("acc. of path-%d under PGD attack:"%idx)
            print(acc_pgd)

        print('--------')
        print("mean acc. under PGD attack:")
        print(np.mean(avg_acc_pgd,axis=0))
        print("std. acc. under PGD attack:")
        print(np.std(avg_acc_pgd,axis=0))

    elif args.attack_type == 'cw':
        print("-------- START CW ATTACK...")
        # attack_param['c'] = 0.1
        attack_param['c'] = 0.01
        attack_param['kappa'] = 0
        attack_param['n_iters'] = 10
        attack_param['lr'] = 0.001
        print('---- attack parameters')
        print(attack_param)

        acc_cw = []
        for idx in range(args.num_heads):
            print("-------- attacking network-%d..."%idx)
            corr_te_cw = attack(backbone, head, idx, testloader, attack_param)
            acc_te_cw = corr_te_cw / float(test_num)
            acc_cw.append(acc_te_cw)

            print("acc. of path-%d under C&W attack = %f"%(idx, acc_te_cw))

        print('--------')
        print("mean acc. under C&W attack:")
        print(np.mean(acc_cw))
        print("std. acc. under C&W attack:")
        print(np.std(acc_cw)) 

    
    elif args.attack_type == 'square':
        print("-------- START AUTO-ATTACK-SQUARE...") 
        attack_param['norm'] = 'L2'
        attack_param['eps'] = 0.5
        attack_param['version'] = 'standard'
        print('---- attack parameters')
        print(attack_param)

        acc_aa = []
        for idx in range(args.num_heads):
            corr_te_aa = attack(backbone, head, idx, testloader, attack_param)
            break
        acc_te_aa = corr_te_aa / float(test_num)
        acc_aa.append(acc_te_aa)
        print('--------')
        print('saved path: '+args.model_path)
        print("mean acc. under SQUARE attack:")
        print(np.mean(acc_aa))
        print("std. acc. under SQUARE attack:")
        print(np.std(acc_aa))   
    
    elif args.attack_type == 'fab':
        print("-------- START AUTO-ATTACK-FAB...")
        attack_param['norm'] = 'Linf'
        attack_param['eps'] = 8/255
        attack_param['version'] = 'standard'
        print('---- attack parameters')
        print(attack_param)

        acc_aa = []
        for idx in range(args.num_heads):
            corr_te_aa = attack(backbone, head, idx, testloader, attack_param)
            break
        acc_te_aa = corr_te_aa / float(test_num)
        acc_aa.append(acc_te_aa)
        print('--------')
        print('saved path: '+args.model_path)
        print("mean acc. under FAB attack:")
        print(np.mean(acc_aa))
        print("std. acc. under FAB attack:")
        print(np.std(acc_aa))   

    elif args.attack_type == 'aa':
        print("-------- START AUTO-ATTACK...") 
        attack_param['norm'] = 'Linf'
        attack_param['eps'] = 8/255
        attack_param['version'] = 'standard'
        print('---- attack parameters')
        print(attack_param)

        acc_aa = []
        for idx in range(args.num_heads):
            corr_te_aa = attack(backbone, head, idx, testloader, attack_param)
            break
        acc_te_aa = corr_te_aa / float(test_num)
        acc_aa.append(acc_te_aa)
        print('--------')
        print('saved path: '+args.model_path)
        print("mean acc. under AA attack:")
        print(np.mean(acc_aa))
        print("std. acc. under AA attack:")
        print(np.std(acc_aa))  

    print('-------- FINISHED.')

    return

# -------- evaluate --------
def evaluate(backbone, head, trainloader, testloader):
    
    backbone.eval()
    head.eval()

    correct_train, correct_test = np.zeros(args.num_heads), np.zeros(args.num_heads)
   
    with torch.no_grad():
            
        for train in trainloader:
            images, labels = train
            images, labels = images.cuda(), labels.cuda()

            all_logits = head(backbone(images), 'all')
            for idx in range(args.num_heads):
                logits = all_logits[idx]
                logits = logits.detach()
                _, pred = torch.max(logits.data, 1)
                correct_train[idx] += (pred == labels).sum().item()
        
        for test in testloader:
            images, labels = test
            images, labels = images.cuda(), labels.cuda()
            all_logits = head(backbone(images), 'all')
            for idx in range(args.num_heads):
                logits = all_logits[idx]
                _, pred = torch.max(logits.data, 1)
                correct_test[idx] += (pred == labels).sum().item() 
        
    return correct_train, correct_test

# -------- attack model --------
def attack(backbone, head, idx, testloader, attack_param):

    if args.attack_type == 'square' or args.attack_type == 'fab' or args.attack_type == 'aa':
        correct = np.zeros(args.num_heads)
    else:
        correct = 0

    backbone.eval()
    head.eval()


    if args.attack_type == 'square':
        def forward_pass(input):
            return head(backbone(input), 'random')
        adversary = AutoAttack(forward_pass, norm=attack_param['norm'], eps=attack_param['eps'], version=attack_param['version'])
        adversary.attacks_to_run = ['square']
    elif args.attack_type == 'fab':
        def forward_pass(input):
            return head(backbone(input), 'random')
        adversary = AutoAttack(forward_pass, norm=attack_param['norm'], eps=attack_param['eps'], version=attack_param['version'])
        adversary.attacks_to_run = ['fab']
    elif args.attack_type == 'aa':
        def forward_pass(input):
            return head(backbone(input), 'random')
        adversary = AutoAttack(forward_pass, norm=attack_param['norm'], eps=attack_param['eps'], version=attack_param['version'])

    for test in testloader:
        image, label = test
        image, label = image.cuda(), label.cuda()

        # generate adversarial examples
        if args.attack_type == "fgsm":
            perturbed_image = fgsm_attack(backbone, head, idx, image, label, attack_param['eps'])
        elif args.attack_type == "pgd":
            perturbed_image = pgd_attack(backbone, head, idx, image, label, attack_param['eps'])
        elif args.attack_type == "cw":
            c, kappa, n_iters, lr = attack_param['c'], attack_param['kappa'], attack_param['n_iters'], attack_param['lr']
            perturbed_image = cw_attack(backbone, head, idx, image, label, c=c, kappa=kappa, n_iters=n_iters, lr=lr)
        elif args.attack_type == 'square' or args.attack_type == 'fab' or args.attack_type == 'aa':
            perturbed_image = adversary.run_standard_evaluation(image, label, bs=image.size(0))

        # re-classify
        if args.attack_type == 'square' or args.attack_type == 'fab' or args.attack_type == 'aa':
            all_logits = head(backbone(perturbed_image), 'all')
            for index in range(args.num_heads):
                logits = all_logits[index]
                logits = logits.detach()
                _, pred = torch.max(logits.data, 1)
                correct[index] += (pred == label).sum().item()
        else:
            logits = head(backbone(perturbed_image), idx)
            logits = logits.detach()
            _, pred = torch.max(logits.data, 1)
            correct += (pred==label).sum().item()
    
    return correct


# -------- start point
if __name__ == '__main__':
    main()