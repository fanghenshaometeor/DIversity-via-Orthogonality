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
from utils import get_datasets, get_model
from utils import AverageMeter, accuracy
from utils import Logger

from adapt_attackers import pgd_adapt

from advertorch.attacks import LinfPGDAttack

# ======== fix data type ========
torch.set_default_tensor_type(torch.FloatTensor)

# ======== options ==============
parser = argparse.ArgumentParser(description='Attack Deep Neural Networks')
# -------- file param. --------------
parser.add_argument('--data_dir',type=str,default='/media/Disk1/KunFang/data/CIFAR10/',help='file path for data')
parser.add_argument('--output_dir',type=str,default='./output/',help='folder to store output')
parser.add_argument('--dataset',type=str,default='CIFAR10',help='data set name')
parser.add_argument('--arch',type=str,default='vgg16',help='model architecture')
parser.add_argument('--model_path',type=str,default='./save/CIFAR10-VGG.pth',help='saved model path')
# -------- training param. ----------
parser.add_argument('--batch_size',type=int,default=256,help='batch size for test (default: 256)')
# -------- save adv. images --------
parser.add_argument('--save_adv_img',type=ast.literal_eval,dest='save_adv_img',help='save adversarial examples')
# -------- hyper parameters -------
parser.add_argument('--num_heads',type=int,default=10,help='number of orthogonal paths')
# -------- attack param. ----------
parser.add_argument('--attack_type',type=str,default='fgsm',help='attack method')
parser.add_argument('--test_eps', default=8., type=float, help='epsilon of attack during testing')
parser.add_argument('--test_step', default=20, type=int, help='itertion number of attack during testing')
parser.add_argument('--test_gamma', default=2., type=float, help='step size of attack during testing')
parser.add_argument('--adapt1', action='store_true')
args = parser.parse_args()

# -------- initialize output store dir.
if args.adapt1:
    save_name = os.path.split(args.model_path)[-1].replace(".pth", "-"+args.attack_type.upper()+"-ADAPT1.log")
else:
    save_name = os.path.split(args.model_path)[-1].replace(".pth", "-"+args.attack_type.upper()+"-ADAPT2.log")
save_param = os.path.split(os.path.split(args.model_path)[-2])[-1]
if 'adv' in args.model_path:
    if not os.path.exists(os.path.join(args.output_dir,args.dataset,args.arch+'-adv',save_param)):
        os.makedirs(os.path.join(args.output_dir,args.dataset,args.arch+'-adv',save_param))
    args.output_path = os.path.join(args.output_dir,args.dataset,args.arch+'-adv',save_param,save_name)
else:
    if not os.path.exists(os.path.join(args.output_dir,args.dataset,args.arch,save_param)):
        os.makedirs(os.path.join(args.output_dir,args.dataset,args.arch,save_param))
    args.output_path = os.path.join(args.output_dir,args.dataset,args.arch,save_param,save_name)
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
    backbone, head = get_model(args)
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

    print('-------- START ATTACKING --------')
    print('-------- ADVERSARY INFORMATION --------')
    if args.attack_type == 'pgd':
        args.test_step = 20
    elif args.attack_type == 'pgd100':
        args.test_step = 100
    else:
        assert False, "Attack type should be either 'pgd' or 'pgd100'."
    print('---- PGD attack with %d/255 step size, %d iterations and bound %d/255.'%(args.test_gamma*255, args.test_step, args.test_eps*255))
    # --------
    if args.adapt1:
        print('-------- START PGD ATTACK - ADAPTIVE-ATTACK-1...')
        acc_pgd = attack_adapt1(backbone, head, testloader)
        acc_pgd_str = ''
        for head_idx in range(args.num_heads):
            acc = acc_pgd[head_idx]
            acc_pgd_str += '%.2f'%acc+'\t'
    else:
        print('-------- START PGD ATTACK - ADAPTIVE-ATTACK-2...')
        acc_pgd = np.zeros(args.num_heads)
        acc_pgd_str = ''
        for head_idx in range(args.num_heads):
            print("-------- attacking network-%d..."%head_idx)
            acc = attack_adapt2(backbone, head, head_idx, testloader)
            acc_pgd[head_idx] = acc
            acc_pgd_str += '%.2f'%acc+'\t'
            print("acc. of path-%d under PGD attack = %.2f"%(head_idx, acc))
    print('--------')
    print('Attacked acc. on each path: \n'+acc_pgd_str)
    print("Attacked mean/std.    acc.:\t"+"%.2f"%np.mean(acc_pgd)+"\t"+"%.2f"%np.std(acc_pgd))


    print('-------- Results saved path: ', args.output_path)
    print('-------- FINISHED.')

    return

# -------- attack model --------
# -------- adaptive attack 1 ---
def attack_adapt1(backbone, head, testloader):

    backbone.eval()
    head.eval()

    top1 = []
    for _ in range(args.num_heads):
        top1.append(AverageMeter())

    for test in testloader:
        image, label = test
        image, label = image.cuda(), label.cuda()

        # generate adversarial examples
        perturbed_image = pgd_adapt(backbone=backbone, head=head, 
                                image=image, label=label, 
                                eps=args.test_eps, alpha=args.test_gamma, iters=args.test_step)

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

# -------- attack model --------
# -------- adaptive attack 2 ---
def attack_adapt2(backbone, head, head_idx, testloader):

    backbone.eval()
    head.eval()

    top1 = AverageMeter()

    def forward(input):
        return head(backbone(input), head_idx)
    adversary = LinfPGDAttack(forward, loss_fn=nn.CrossEntropyLoss(), eps=args.test_eps, nb_iter=args.test_step, eps_iter=args.test_gamma, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)

    for test in testloader:
        image, label = test
        image, label = image.cuda(), label.cuda()

        # generate adversarial examples
        perturbed_image = adversary.perturb(image, label)

        # re-classify
        logits = head(backbone(perturbed_image), head_idx).detach().float()
        prec1 = accuracy(logits.data, label)[0]
        top1.update(prec1.item(), image.size(0))
    
    return top1.avg

# -------- start point
if __name__ == '__main__':
    main()