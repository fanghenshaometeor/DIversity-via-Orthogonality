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
from utils import get_datasets, get_model_dio
from utils import get_model_teacher
from utils import AverageMeter, accuracy

# from lbgat import lbgat_dio_loss

from advertorch.attacks import LinfPGDAttack
from advertorch.context import ctx_noparamgrad_and_eval

# ======== fix data type ========
torch.set_default_tensor_type(torch.FloatTensor)

# ======== options ==============
parser = argparse.ArgumentParser(description='Training DIO+LBGAT')
# -------- file param. --------------
parser.add_argument('--data_dir',type=str,default='/media/Disk1/KunFang/data/CIFAR10/',help='file path for data')
parser.add_argument('--model_dir',type=str,default='./save/',help='file path for saving model')
parser.add_argument('--logs_dir',type=str,default='./runs/',help='log path')
parser.add_argument('--dataset',type=str,default='CIFAR10',help='data set name')
parser.add_argument('--arch',type=str,default='vgg16',help='model architecture')
# -------- training param. ----------
parser.add_argument('--batch_size',type=int,default=128,help='batch size for training (default: 256)')    
parser.add_argument('--epochs',type=int,default=100,help='number of epochs to train (default: 100)')
parser.add_argument('--save_freq',type=int,default=20,help='model save frequency (default: 20 epoch)')
# -------- hyper parameters -------
parser.add_argument('--alpha',type=float,default=0.1,help='coefficient of the orthogonality regularization term')
parser.add_argument('--beta',type=float,default=0.1,help='coefficient of the margin regularization term')
parser.add_argument('--num_heads',type=int,default=10,help='number of orthogonal paths')
parser.add_argument('--tau',type=float,default=0.2,help='upper bound of the margin')
# -------- enable adversarial training --------
parser.add_argument('--train_eps', default=8., type=float, help='epsilon of attack during training')
parser.add_argument('--train_step', default=10, type=int, help='itertion number of attack during training')
parser.add_argument('--train_gamma', default=2., type=float, help='step size of attack during training')
parser.add_argument('--test_eps', default=8., type=float, help='epsilon of attack during testing')
parser.add_argument('--test_step', default=20, type=int, help='itertion number of attack during testing')
parser.add_argument('--test_gamma', default=2., type=float, help='step size of attack during testing')
# -------- lBGAT training --------
parser.add_argument('--test-batch-size', type=int, default=32, help='input batch size for testing (default: 128)')
parser.add_argument('--beta_trades', default=6.0, help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--teacher_model', type=str, default='resnet18')
args = parser.parse_args()

# ======== initialize log writer
writer = SummaryWriter(os.path.join(args.logs_dir, args.dataset, args.arch, 'LBGAT+DIO', \
    'p-'+str(args.num_heads)+'-a-'+str(args.alpha)+'-b-'+str(args.beta)+'-tau-'+str(args.tau)+'/'))
# ----
model_name = 'p-'+str(args.num_heads)+'-a-'+str(args.alpha)+'-b-'+str(args.beta)+'-tau-'+str(args.tau)
# ----
if not os.path.exists(os.path.join(args.model_dir,args.dataset,args.arch,'LBGAT+DIO',model_name)):
    os.makedirs(os.path.join(args.model_dir,args.dataset,args.arch,'LBGAT+DIO',model_name))
# ----
args.save_path = os.path.join(args.model_dir,args.dataset,args.arch,'LBGAT+DIO',model_name)

# -------- main function
def main():
    
    # ======== fix random seed ========
    setup_seed(666)

    # ======== get data set =============
    trainloader, testloader = get_datasets(args)
    print('-------- DATA INFOMATION --------')
    print('---- dataset: '+args.dataset)

    # ======== initialize net
    backbone, head = get_model_dio(args)
    net_teacher = get_model_teacher(args)
    backbone, head = backbone.cuda(), head.cuda()
    net_teacher = net_teacher.cuda()
    print('-------- MODEL INFORMATION --------')
    print('---- architecture: '+args.arch)
    print('---- saved   path: '+args.save_path)

    # ======== set criterions & optimizers
    criterion_ce = nn.CrossEntropyLoss()
    criterion_mse = nn.MSELoss()
    optimizer = optim.SGD([{'params':backbone.parameters()},
                        {'params':head.parameters()},
                        {'params':net_teacher.parameters()}], 
                        lr=0.02, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [75,90], gamma=0.1)

    # ======== 
    args.train_eps /= 255.
    args.train_gamma /= 255.
    args.test_eps /= 255.
    args.test_gamma /= 255.
    best_robust_te_acc = .0
    best_robust_epoch = 0
    def forward(input):
        return head(backbone(input), 'random')
    def forward_val(input):
        return head(backbone(input), 0)
    adversary = LinfPGDAttack(forward, loss_fn=criterion_ce, eps=args.train_eps, nb_iter=args.train_step, eps_iter=args.train_gamma, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)
    adversary_val = LinfPGDAttack(forward_val, loss_fn=criterion_ce, eps=args.test_eps, nb_iter=args.test_step, eps_iter=args.test_gamma, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)
    
    print('-------- START TRAINING --------')
    for epoch in range(1, args.epochs+1):

        # -------- train
        print('Training(%d/%d)...'%(epoch, args.epochs))
        train_epoch([backbone, head], net_teacher, trainloader, optimizer, [criterion_ce,criterion_mse], epoch, adversary)

        # -------- adversarial validation
        valstats = {}
        print('Adversarial Validating...')
        robust_te_acc = val_adv(backbone, head, testloader, adversary_val)
        valstats['robustacc'] = robust_te_acc
        print('     Current robust accuracy = %.2f.'%robust_te_acc)

        # ---- best updated, print info. & save model
        if robust_te_acc >= best_robust_te_acc:
            best_robust_te_acc = robust_te_acc
            best_robust_epoch = epoch
            print('     Best robust accuracy %.2f updated  at epoch-%d.'%(best_robust_te_acc, best_robust_epoch))

            checkpoint = {'state_dict_backbone': backbone.state_dict(), 'state_dict_head': head.state_dict(), 'best-epoch': best_robust_epoch}
            args.model_path = 'best.pth'
            torch.save(checkpoint, os.path.join(args.save_path,args.model_path))
        else:
            print('     Best robust accuracy %.2f achieved at epoch-%d'%(best_robust_te_acc, best_robust_epoch))  

        # -------- validation, print info. & save model
        print('Validating...')
        acc_te = val(backbone, head, testloader)
        acc_te_str = ''
        for idx in range(args.num_heads):
            valstats['cleanacc-path-%d'%idx] = acc_te[idx].avg
            acc_te_str += '%.2f'%acc_te[idx].avg+'\t'
        writer.add_scalars('valacc', valstats, epoch)
        print('     Current test acc. on each path: \n'+acc_te_str)

        if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
            checkpoint = {'state_dict_backbone': backbone.state_dict(), 'state_dict_head': head.state_dict()}
            args.model_path = 'epoch%d'%epoch+'.pth'
            torch.save(checkpoint, os.path.join(args.save_path,args.model_path))

        scheduler.step()
        print('Current training model: ', args.save_path)
        print('===========================================')
    
    print('-------- TRAINING finished.')

# ======== train  model ========
def train_epoch(net, net_teacher, trainloader, optim, criterions, epoch, adversary):
    
    backbone, head = net
    backbone.train()
    head.train()
    net_teacher.train()

    ce, mse = criterions

    # -------- preparing recorder
    batch_time = AverageMeter()
    losses, losses_ortho, losses_margin = AverageMeter(), AverageMeter(), AverageMeter()
    losses_lbgat = AverageMeter()
    

    end = time.time()
    for batch_idx, (b_data, b_label) in enumerate(trainloader):
        
        # -------- move to gpu
        b_data, b_label = b_data.cuda(), b_label.cuda()

        
        # ==============================
        # ==============================
        # ======== LBGAT + DIO ========= 
        with ctx_noparamgrad_and_eval(backbone):
            with ctx_noparamgrad_and_eval(head):
                perturbed_data = adversary.perturb(b_data, b_label)

        all_logits_adv = head(backbone(perturbed_data), 'all')
        out = net_teacher(b_data)

        loss_lbgat = .0
        for idx in range(args.num_heads):
            out_adv = all_logits_adv[idx]
            loss_lbgat += 1/(args.num_heads) * (mse(out_adv,out)+ ce(out,b_label))
        # ======== LBGAT + DIO =========
        # ==============================
        # ==============================

        # -------- compute MARGIN loss
        loss_margin = head.compute_margin_loss(all_logits_adv, b_label, args.tau)
        
        # -------- compute the ORTHOGONALITY constraint
        loss_ortho = .0
        if args.num_heads > 1:
            loss_ortho = head.compute_ortho_loss()

        # -------- SUM the three losses
        total_loss = loss_lbgat + args.alpha * loss_ortho + args.beta * loss_margin
    
        # -------- backprop. & update
        optim.zero_grad()
        total_loss.backward()
        optim.step()

        # -------- record & print in termial
        losses.update(total_loss, b_data.size(0))
        losses_lbgat.update(loss_lbgat.float().item(), b_data.size(0))
        losses_ortho.update(loss_ortho, b_data.size(0))
        losses_margin.update(loss_margin, b_data.size(0))
        # ----
        batch_time.update(time.time()-end)
        end = time.time()

    writer.add_scalar('loss-lbgat', losses_lbgat.avg, epoch)
    writer.add_scalar('loss-ortho', losses_ortho.avg, epoch)
    writer.add_scalar('loss-margin', losses_margin.avg, epoch)
    print('     Epoch %d/%d costs %fs.'%(epoch, args.epochs, batch_time.sum))
    print('     LBGAT   loss = %f.'%losses_lbgat.avg)
    print('     ORTHO   loss = %f.'%losses_ortho.avg)
    print('     MARGIN  loss = %f.'%losses_margin.avg)

    return

# ======== evaluate model ========
def val(backbone, head, dataloader):
    
    backbone.eval()
    head.eval()

    batch_time = AverageMeter()

    acc = []
    for idx in range(args.num_heads):
        measure = AverageMeter()
        acc.append(measure)
    
    end = time.time()
    with torch.no_grad():
        
        # -------- compute the accs. 
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
            
            # ----
            batch_time.update(time.time()-end)
            end = time.time()
            
    print('     Validation costs %fs.'%(batch_time.sum))      
    return acc

# ======== evaluate adversarial ========
def val_adv(backbone, head, dataloader, adversary_val):

    backbone.eval()
    head.eval()

    top1 = AverageMeter()
    batch_time = AverageMeter()

    end = time.time()
    for _, test in enumerate(dataloader):
        images, labels = test
        images, labels = images.cuda(), labels.cuda()

        perturbed_images = adversary_val.perturb(images, labels)

        logits = head(backbone(perturbed_images), 0)
        prec1 = accuracy(logits.data, labels)[0]
        top1.update(prec1.item(), images.size(0))

        # ----
        batch_time.update(time.time()-end)
        end = time.time()
    
    print('     Adversarial validation costs %fs.'%(batch_time.sum))
    return top1.avg  

# ======== startpoint
if __name__ == '__main__':
    main()