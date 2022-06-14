"""
Created on Mon Mar 09 2020

@author: fanghenshao
"""

from __future__ import print_function


import torch
import torch.nn as nn

import os
import sys
import ast
import copy
import argparse

import numpy as np

from utils import setup_seed
from utils import get_datasets, get_model_dio
from utils import AverageMeter, accuracy
from utils import Logger

from advertorch.attacks import LinfPGDAttack

from autoattack import AutoAttack

# from autoattack import AutoAttack

# ======== fix data type ========
torch.set_default_tensor_type(torch.FloatTensor)

# ======== options ==============
parser = argparse.ArgumentParser(description='Attack DIO+AWP')
# -------- file param. --------------
parser.add_argument('--data_dir',type=str,default='/media/Disk1/KunFang/data/CIFAR10/',help='file path for data')
parser.add_argument('--output_dir',type=str,default='./output/',help='folder to store output')
parser.add_argument('--dataset',type=str,default='CIFAR10',help='data set name')
parser.add_argument('--arch',type=str,default='vgg16',help='model architecture')
parser.add_argument('--model_path',type=str,default='./save/CIFAR10-VGG.pth',help='saved model path')
# -------- training param. ----------
parser.add_argument('--batch_size',type=int,default=128,help='batch size for test (default: 256)')
# -------- save adv. images --------
parser.add_argument('--save_adv_img',type=ast.literal_eval,dest='save_adv_img',help='save adversarial examples')
# -------- hyper parameters -------
parser.add_argument('--num_heads',type=int,default=10,help='number of orthogonal paths')
# -------- attack param. ----------
parser.add_argument('--attack_type',type=str,default='fgsm',help='attack method')
parser.add_argument('--test_eps', default=8., type=float, help='epsilon of attack during testing')
parser.add_argument('--test_step', default=20, type=int, help='itertion number of attack during testing')
parser.add_argument('--test_gamma', default=2., type=float, help='step size of attack during testing')
args = parser.parse_args()

# -------- initialize output store dir.
save_name = os.path.split(args.model_path)[-1].replace(".pth", "-"+args.attack_type.upper()+".log")
save_param = os.path.split(os.path.split(args.model_path)[-2])[-1]
if not os.path.exists(os.path.join(args.output_dir,args.dataset,args.arch,'DIO+AWP',save_param)):
    os.makedirs(os.path.join(args.output_dir,args.dataset,args.arch,'DIO+AWP',save_param))
args.output_path = os.path.join(args.output_dir,args.dataset,args.arch,'DIO+AWP',save_param,save_name)
sys.stdout = Logger(filename=args.output_path,stream=sys.stdout)

# -------- main function
def main():

    # ======== fix seed =============
    setup_seed(666)

    # ======== get data set =============
    trainloader, testloader = get_datasets(args)
    print('-------- DATA INFOMATION --------')
    print('---- dataset: '+args.dataset)

    # ======== load network ========
    checkpoint = torch.load(args.model_path, map_location=torch.device("cpu"))
    backbone, head = get_model_dio(args)
    backbone, head = backbone.cuda(), head.cuda()
    backbone.load_state_dict(checkpoint['state_dict_backbone'])
    head.load_state_dict(checkpoint['state_dict_head'])
    backbone.eval()
    head.eval()
    print('-------- MODEL INFORMATION --------')
    print('---- architecture: '+args.arch)
    print('---- saved   path: '+args.model_path)
    if 'best' in args.model_path:
        print('---- best robust acc. achieved at epoch-%d.'%checkpoint['best-epoch'])

    args.test_eps /= 255.
    args.test_gamma /= 255.

    if args.attack_type == 'None':

        print('-------- START TESTING --------')
        print('Evaluating...')
        acc_tr, acc_te = val(backbone, head, trainloader), val(backbone, head, testloader)
        acc_tr_str, acc_te_str = '', ''
        acc_tr_val, acc_te_val = np.zeros(args.num_heads), np.zeros(args.num_heads)
        for idx in range(args.num_heads):
            acc_tr_str += '%.3f'%acc_tr[idx].avg+'\t'
            acc_te_str += '%.2f'%acc_te[idx].avg+'\t'
            acc_tr_val[idx] = acc_tr[idx].avg
            acc_te_val[idx] = acc_te[idx].avg
        print('training acc. on each path: \n'+acc_tr_str)
        print('test     acc. on each path: \n'+acc_te_str)
        print("mean/std. acc. on clean training set:\t"+"%.2f"%np.mean(acc_tr_val)+"\t"+"%.2f"%np.std(acc_tr_val))
        print("mean/std. acc. on clean test     set:\t"+"%.2f"%np.mean(acc_te_val)+"\t"+"%.2f"%np.std(acc_te_val))

    
    elif args.attack_type == 'pgd':

        print('-------- START ATTACKING --------')
        print('-------- ADVERSARY INFORMATION --------')
        print('---- PGD attack with %d/255 step size, %d iterations and bound %d/255.'%(args.test_gamma*255, args.test_step, args.test_eps*255))
        # --------
        print('-------- START PGD ATTACK...')
        print('-------- Randomly-forward...')
        acc_pgd = attack(backbone, head, testloader)
        acc_pgd_str = ''
        for head_idx in range(args.num_heads):
            acc = acc_pgd[head_idx]
            acc_pgd_str += '%.2f'%acc+'\t'
        print('--------')
        print('Attacked acc. on each path: \n'+acc_pgd_str)
        print("Attacked mean/std.    acc.:\t"+"%.2f"%np.mean(acc_pgd)+"\t"+"%.2f"%np.std(acc_pgd))

    elif args.attack_type == 'pgd100':

        args.test_step = 100
        print('-------- START ATTACKING --------')
        print('-------- ADVERSARY INFORMATION --------')
        print('---- PGD attack with %d/255 step size, %d iterations and bound %d/255.'%(args.test_gamma*255, args.test_step, args.test_eps*255))
        # --------
        print('-------- START PGD ATTACK...')
        print('-------- Randomly-forward...')
        acc_pgd = attack(backbone, head, testloader)
        acc_pgd_str = ''
        for head_idx in range(args.num_heads):
            acc = acc_pgd[head_idx]
            acc_pgd_str += '%.2f'%acc+'\t'
        print('--------')
        print('Attacked acc. on each path: \n'+acc_pgd_str)
        print("Attacked mean/std.    acc.:\t"+"%.2f"%np.mean(acc_pgd)+"\t"+"%.2f"%np.std(acc_pgd))

    elif args.attack_type == 'square':
        print('-------- START ATTACKING --------')
        print('-------- ADVERSARY INFORMATION --------')
        print('---- SQUARE attack with default settings in AutoAttack (bound=L2-0.5)')
        # --------
        print('-------- START SQUARE ATTACK...')
        print('-------- Randomly-forward...')
        acc_square = attack(backbone, head, testloader)
        acc_square_str = ''
        for head_idx in range(args.num_heads):
            acc = acc_square[head_idx]
            acc_square_str += '%.2f'%acc+'\t'
        print('--------')
        print('Attacked acc. on each path: \n'+acc_square_str)
        print("Attacked mean/std.    acc.:\t"+"%.2f"%np.mean(acc_square)+"\t"+"%.2f"%np.std(acc_square))

    elif args.attack_type == 'aa':
        print("-------- START AUTO-ATTACK...") 
        print('-------- ADVERSARY INFORMATION --------')
        print('---- AutoAttack with default settings (bound=Linf-%d/255)'%(args.test_eps*255))
        # --------
        print('-------- START AUTO-ATTACK...')
        print('-------- Randomly-forward...')
        acc_aa = attack(backbone, head, testloader)
        acc_aa_str = ''
        for head_idx in range(args.num_heads):
            acc = acc_aa[head_idx]
            acc_aa_str += '%.2f'%acc+'\t'
        print('--------')
        print('Attacked acc. on each path: \n'+acc_aa_str)
        print("Attacked mean/std.    acc.:\t"+"%.2f"%np.mean(acc_aa)+"\t"+"%.2f"%np.std(acc_aa))
    
    else:
        assert False, "Unknown attack : {}".format(args.attack_type)

    print('-------- Results saved path: ', args.output_path)
    print('-------- FINISHED.')

    return

# ======== evaluate model ========
def val(backbone, head, dataloader):
    
    backbone.eval()
    head.eval()

    acc = []
    for idx in range(args.num_heads):
        measure = AverageMeter()
        acc.append(measure)
    
    with torch.no_grad():
        
        # -------- compute the accs. of train, test set
        for test in dataloader:
            images, labels = test
            images, labels = images.cuda(), labels.cuda()

            # ------- forward 
            all_logits = head(backbone(images), 'all')
            for idx in range(args.num_heads):
                logits = all_logits[idx]
                logits = logits.detach().float()

                prec1 = accuracy(logits.data, labels)[0]
                acc[idx].update(prec1.item(), images.size(0))
            
    return acc

# -------- attack model --------
# -------- RANDOMLY  FORWARD PATH
def attack(backbone, head, testloader):

    backbone.eval()
    head.eval()

    top1 = []
    for _ in range(args.num_heads):
        top1.append(AverageMeter())

    if args.attack_type == 'pgd' or args.attack_type == 'pgd100':
        def forward(input):
            return head(backbone(input), 'random')
        adversary = LinfPGDAttack(forward, loss_fn=nn.CrossEntropyLoss(), eps=args.test_eps, nb_iter=args.test_step, eps_iter=args.test_gamma, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)

    elif args.attack_type == 'square':
        def forward(input):
            return head(backbone(input), 'random')
        adversary = AutoAttack(forward, norm='L2', eps=0.5, version='standard', verbose=False)
        adversary.attacks_to_run = ['square']

    elif args.attack_type == 'aa':
        def forward(input):
            return head(backbone(input), 'random')
        adversary = AutoAttack(forward, norm='Linf', eps=args.test_eps, version='standard', verbose=False)
    
    else:
        assert False, "Unknown attack : {}".format(args.attack_type)

    for test in testloader:
        image, label = test
        image, label = image.cuda(), label.cuda()

        # generate adversarial examples
        if args.attack_type == "fgsm" or args.attack_type == 'pgd' or args.attack_type == 'cw' or args.attack_type == 'pgd100':
            perturbed_image = adversary.perturb(image, label)
        elif args.attack_type == 'square' or args.attack_type == 'aa':
            perturbed_image = adversary.run_standard_evaluation(image, label, bs=image.size(0))

        # re-classify
        all_logits = head(backbone(perturbed_image), 'all')
        for index in range(args.num_heads):
            logits = all_logits[index]
            logits = logits.detach()
            prec1 = accuracy(logits.data, label)[0]
            top1[index].update(prec1.item(), image.size(0))    
    
    for index in range(args.num_heads):
        top1[index] = top1[index].avg

    return top1


# -------- start point
if __name__ == '__main__':
    main()