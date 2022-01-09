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

from advertorch.attacks import GradientSignAttack
from advertorch.attacks import LinfPGDAttack

# from autoattack import AutoAttack

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
args = parser.parse_args()

# -------- initialize output store dir.
if 'adv' in args.model_path:
    if not os.path.exists(os.path.join(args.output_dir,args.dataset,args.arch+'-adv')):
        os.makedirs(os.path.join(args.output_dir,args.dataset,args.arch+'-adv'))
    args.output_path = os.path.split(args.model_path)[-1].replace(".pth", "-"+args.attack_type.upper()+".log")
    args.output_path = os.path.join(args.output_dir,args.dataset,args.arch+'-adv',args.output_path)
else:
    if not os.path.exists(os.path.join(args.output_dir,args.dataset,args.arch)):
        os.makedirs(os.path.join(args.output_dir,args.dataset,args.arch))
    args.output_path = os.path.split(args.model_path)[-1].replace(".pth", "-"+args.attack_type.upper()+".log")
    args.output_path = os.path.join(args.output_dir,args.dataset,args.arch,args.output_path)
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

    
    elif args.attack_type == 'fgsm':
        print('-------- START ATTACKING --------')
        print('-------- ADVERSARY INFORMATION --------')
        print('---- FGSM attack with bound %d/255.'%(args.test_eps*255))
        # --------
        print('-------- START FGSM ATTACK...')
        acc_fgsm = np.zeros(args.num_heads)
        acc_fgsm_str = ''
        for head_idx in range(args.num_heads):
            print("-------- attacking network-%d..."%head_idx)
            acc = attack(backbone, head, head_idx, testloader)
            acc_fgsm[head_idx] = acc
            acc_fgsm_str += '%.2f'%acc+'\t'
            print("acc. of path-%d under FGSM attack = %.2f"%(head_idx, acc))
        print('--------')
        print('Attacked acc. on each path: \n'+acc_fgsm_str)
        print("Attacked mean/std.    acc.:\t"+"%.2f"%np.mean(acc_fgsm)+"\t"+"%.2f"%np.std(acc_fgsm))

    elif args.attack_type == 'pgd':
        print('-------- START ATTACKING --------')
        print('-------- ADVERSARY INFORMATION --------')
        print('---- PGD attack with %d/255 step size, %d iterations and bound %d/255.'%(args.test_gamma*255, args.test_step, args.test_eps*255))
        # --------
        print('-------- START PGD ATTACK...')
        acc_pgd = np.zeros(args.num_heads)
        acc_pgd_str = ''
        for head_idx in range(args.num_heads):
            print("-------- attacking network-%d..."%head_idx)
            acc = attack(backbone, head, head_idx, testloader)
            acc_pgd[head_idx] = acc
            acc_pgd_str += '%.2f'%acc+'\t'
            print("acc. of path-%d under PGD attack = %.2f"%(head_idx, acc))
        print('--------')
        print('Attacked acc. on each path: \n'+acc_pgd_str)
        print("Attacked mean/std.    acc.:\t"+"%.2f"%np.mean(acc_pgd)+"\t"+"%.2f"%np.std(acc_pgd))



    # elif args.attack_type == 'pgd':
    #     print('-------- START PGD ATTACK...')
    #     # pgd_epsilons = [1/255, 2/255, 3/255, 4/255, 5/255, 6/255, 7/255, 8/255, 9/255, 10/255, 11/255, 12/255]
    #     pgd_epsilons = [1/255, 2/255, 4/255]
    #     avg_acc_pgd = np.zeros([args.num_heads,len(pgd_epsilons)])
    #     for idx in range(args.num_heads):
    #         print("-------- attacking network-%d..."%idx)
    #         acc_pgd = []
    #         for eps in pgd_epsilons:
    #             attack_param['eps'] = eps
    #             corr_te_pgd = attack(backbone, head, idx, testloader, attack_param)
    #             acc_pgd.append(corr_te_pgd/float(test_num))

    #         avg_acc_pgd[idx,:] = avg_acc_pgd[idx,:] + np.array(acc_pgd)
    #         print("acc. of path-%d under PGD attack:"%idx)
    #         print(acc_pgd)

    #     print('--------')
    #     print("mean acc. under PGD attack:")
    #     print(np.mean(avg_acc_pgd,axis=0))
    #     print("std. acc. under PGD attack:")
    #     print(np.std(avg_acc_pgd,axis=0))

    # elif args.attack_type == 'cw':
    #     print("-------- START CW ATTACK...")
    #     # attack_param['c'] = 0.1
    #     attack_param['c'] = 0.01
    #     attack_param['kappa'] = 0
    #     attack_param['n_iters'] = 10
    #     attack_param['lr'] = 0.001
    #     print('---- attack parameters')
    #     print(attack_param)

    #     acc_cw = []
    #     for idx in range(args.num_heads):
    #         print("-------- attacking network-%d..."%idx)
    #         corr_te_cw = attack(backbone, head, idx, testloader, attack_param)
    #         acc_te_cw = corr_te_cw / float(test_num)
    #         acc_cw.append(acc_te_cw)

    #         print("acc. of path-%d under C&W attack = %f"%(idx, acc_te_cw))

    #     print('--------')
    #     print("mean acc. under C&W attack:")
    #     print(np.mean(acc_cw))
    #     print("std. acc. under C&W attack:")
    #     print(np.std(acc_cw)) 

    
    # elif args.attack_type == 'square':
    #     print("-------- START AUTO-ATTACK-SQUARE...") 
    #     attack_param['norm'] = 'L2'
    #     attack_param['eps'] = 0.5
    #     attack_param['version'] = 'standard'
    #     print('---- attack parameters')
    #     print(attack_param)

    #     acc_aa = []
    #     for idx in range(args.num_heads):
    #         corr_te_aa = attack(backbone, head, idx, testloader, attack_param)
    #         break
    #     acc_te_aa = corr_te_aa / float(test_num)
    #     acc_aa.append(acc_te_aa)
    #     print('--------')
    #     print('saved path: '+args.model_path)
    #     print("mean acc. under SQUARE attack:")
    #     print(np.mean(acc_aa))
    #     print("std. acc. under SQUARE attack:")
    #     print(np.std(acc_aa))   
    
    # elif args.attack_type == 'fab':
    #     print("-------- START AUTO-ATTACK-FAB...")
    #     attack_param['norm'] = 'Linf'
    #     attack_param['eps'] = 8/255
    #     attack_param['version'] = 'standard'
    #     print('---- attack parameters')
    #     print(attack_param)

    #     acc_aa = []
    #     for idx in range(args.num_heads):
    #         corr_te_aa = attack(backbone, head, idx, testloader, attack_param)
    #         break
    #     acc_te_aa = corr_te_aa / float(test_num)
    #     acc_aa.append(acc_te_aa)
    #     print('--------')
    #     print('saved path: '+args.model_path)
    #     print("mean acc. under FAB attack:")
    #     print(np.mean(acc_aa))
    #     print("std. acc. under FAB attack:")
    #     print(np.std(acc_aa))   

    # elif args.attack_type == 'aa':
        # print("-------- START AUTO-ATTACK...") 
        # attack_param['norm'] = 'Linf'
        # attack_param['eps'] = 8/255
        # attack_param['version'] = 'standard'
        # print('---- attack parameters')
        # print(attack_param)

        # acc_aa = []
        # for idx in range(args.num_heads):
        #     corr_te_aa = attack(backbone, head, idx, testloader, attack_param)
        #     break
        # acc_te_aa = corr_te_aa / float(test_num)
        # acc_aa.append(acc_te_aa)
        # print('--------')
        # print('saved path: '+args.model_path)
        # print("mean acc. under AA attack:")
        # print(np.mean(acc_aa))
        # print("std. acc. under AA attack:")
        # print(np.std(acc_aa))  

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
def attack(backbone, head, head_idx, testloader):

    if args.attack_type == 'square' or args.attack_type == 'fab' or args.attack_type == 'aa':
        correct = np.zeros(args.num_heads)
    else:
        correct = 0

    backbone.eval()
    head.eval()

    top1 = AverageMeter()


    if args.attack_type == 'fgsm':
        def forward(input):
            return head(backbone(input), head_idx)
        adversary = GradientSignAttack(forward, loss_fn=nn.CrossEntropyLoss(), eps=args.test_eps, clip_min=0.0, clip_max=1.0, targeted=False)
    
    elif args.attack_type == 'pgd':
        def forward(input):
            return head(backbone(input), head_idx)
        adversary = LinfPGDAttack(forward, loss_fn=nn.CrossEntropyLoss(), eps=args.test_eps, nb_iter=args.test_step, eps_iter=args.test_gamma, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)

    elif args.attack_type == 'square':
        def forward(input):
            return head(backbone(input), 'random')
        adversary = AutoAttack(forward, norm=attack_param['norm'], eps=attack_param['eps'], version=attack_param['version'])
        adversary.attacks_to_run = ['square']
    elif args.attack_type == 'fab':
        def forward(input):
            return head(backbone(input), 'random')
        adversary = AutoAttack(forward, norm=attack_param['norm'], eps=attack_param['eps'], version=attack_param['version'])
        adversary.attacks_to_run = ['fab']
    elif args.attack_type == 'aa':
        def forward(input):
            return head(backbone(input), 'random')
        adversary = AutoAttack(forward, norm=attack_param['norm'], eps=attack_param['eps'], version=attack_param['version'])

    for test in testloader:
        image, label = test
        image, label = image.cuda(), label.cuda()

        # generate adversarial examples
        if args.attack_type == "fgsm" or args.attack_type == 'pgd':
            perturbed_image = adversary.perturb(image, label)
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
            logits = head(backbone(perturbed_image), head_idx).detach().float()

            prec1 = accuracy(logits.data, label)[0]
            top1.update(prec1.item(), image.size(0))
    
    return top1.avg


# -------- start point
if __name__ == '__main__':
    main()