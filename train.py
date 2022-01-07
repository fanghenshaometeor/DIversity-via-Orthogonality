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
# -------- hyper parameters -------
parser.add_argument('--alpha',type=float,default=0.1,help='coefficient of the orthogonality regularization term')
parser.add_argument('--beta',type=float,default=0.1,help='coefficient of the margin regularization term')
parser.add_argument('--num_heads',type=int,default=10,help='number of orthogonal paths')
parser.add_argument('--tau',type=float,default=0.2,help='upper bound of the margin')
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
        '-tau-'+str(args.tau)+'/'))
    # --------
    if not os.path.exists(os.path.join(args.model_dir,args.dataset,args.arch+'-adv')):
        os.makedirs(os.path.join(args.model_dir,args.dataset,args.arch+'-adv'))
    # --------
    model_name = 'p-'+str(args.num_heads) \
        +'-a-'+str(args.alpha)+'-b-'+str(args.beta)+ '-tau-'+str(args.tau)
    # --------
    args.save_path = os.path.join(args.model_dir,args.dataset,args.arch+'-adv',model_name)
else:
    writer = SummaryWriter(os.path.join(args.logs_dir, args.dataset, args.arch, \
        'p-'+str(args.num_heads)+'-a-'+str(args.alpha)+'-b-'+str(args.beta)+ \
        '-tau-'+str(args.tau)+'/'))
    # --------
    if not os.path.exists(os.path.join(args.model_dir,args.dataset,args.arch)):
        os.makedirs(os.path.join(args.model_dir,args.dataset,args.arch))
    # --------
    model_name = 'p-'+str(args.num_heads) \
        +'-a-'+str(args.alpha)+'-b-'+str(args.beta)+ '-tau-'+str(args.tau)
    # --------
    args.save_path = os.path.join(args.model_dir,args.dataset,args.arch,model_name)

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
    print('-------- MODEL INFORMATION --------')
    print('---- architecture: '+args.arch)
    print('---- adv.   train: '+str(args.adv_train))
    print('---- saved   path: '+args.save_path)

    # ======== set criterions & optimizers
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD([{'params':backbone.parameters()},{'params':head.parameters()}], lr=0.05, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [60,120,160], gamma=0.1)

    # ======== 
    args.train_eps /= 255.
    args.train_gamma /= 255.
    if args.adv_train:
        def forward(input):
            return head(backbone(input), 'random')
        adversary = LinfPGDAttack(forward, loss_fn=criterion, eps=args.train_eps, nb_iter=args.train_step, eps_iter=args.train_gamma, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)
    else:
        adversary = None


    print('-------- START TRAINING --------')
    for epoch in range(1, args.epochs+1):

        # -------- train
        train_epoch(backbone, head, trainloader, optimizer, criterion, epoch, adversary)

        # -------- validation, print info. & save model
        if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
            print('Evaluating...')
            acc_tr, acc_te = val(backbone, head, trainloader), val(backbone, head, testloader)
            acc_tr_str, acc_te_str = '', ''
            for idx in range(args.num_heads):
                acc_tr_str += '%.3f'%acc_tr[idx].avg+'\t'
                acc_te_str += '%.2f'%acc_te[idx].avg+'\t'
            print('training acc. on each path: \n'+acc_tr_str)
            print('test     acc. on each path: \n'+acc_te_str)
            # --------
            checkpoint = {'state_dict_backbone': backbone.state_dict(), 'state_dict_head': head.state_dict()}
            torch.save(checkpoint, args.save_path+"-epoch-%d"%epoch+".pth")

        scheduler.step()
        print('Current training model: ', args.save_path)
        print('===========================================')
    
    print('-------- TRAINING finished.')

# ======== train  model ========
def train_epoch(backbone, head, trainloader, optim, criterion, epoch, adversary):
    
    backbone.train()
    head.train()

    # -------- preparing recorder
    batch_time = AverageMeter()
    losses, losses_ortho, losses_margin = AverageMeter(), AverageMeter(), AverageMeter()
    losses_ce = []
    for idx in range(args.num_heads):
        losses_ce.append(AverageMeter())
    losses_ce.append(AverageMeter())
    

    end = time.time()
    for batch_idx, (b_data, b_label) in enumerate(trainloader):
        
        # -------- move to gpu
        b_data, b_label = b_data.cuda(), b_label.cuda()

        if args.adv_train:
            with ctx_noparamgrad_and_eval(backbone):
                with ctx_noparamgrad_and_eval(head):
                    perturbed_data = adversary.perturb(b_data, b_label)
            all_logits = head(backbone(perturbed_data), 'all')
        else:
            all_logits = head(backbone(b_data), 'all')

        # -------- compute MARGIN loss
        loss_margin = head.compute_margin_loss(all_logits, b_label, args.tau)
        
        # -------- compute CROSS ENTROPY loss
        loss_ce = .0
        for idx in range(args.num_heads):
            logits = all_logits[idx]
            loss = criterion(logits, b_label)
            loss_ce += 1/args.num_heads * loss

            losses_ce[idx].update(loss.float().item(), b_data.size(0))

        # -------- compute the ORTHOGONALITY constraint
        loss_ortho = .0
        if args.num_heads > 1:
            loss_ortho = head.compute_ortho_loss()

        # -------- SUM the three losses
        total_loss = loss_ce + args.alpha * loss_ortho + args.beta * loss_margin
    
        # -------- backprop. & update
        optim.zero_grad()
        total_loss.backward()
        optim.step()

        # -------- record & print in termial
        losses.update(total_loss, b_data.size(0))
        losses_ce[args.num_heads].update(loss_ce.float().item(), b_data.size(0))
        losses_ortho.update(loss_ortho, b_data.size(0))
        losses_margin.update(loss_margin, b_data.size(0))
        # ----
        batch_time.update(time.time()-end)
        end = time.time()

    losses_ce_record = {}
    losses_ce_str = ''
    for idx in range(args.num_heads):
        losses_ce_record['path-%d'%idx] = losses_ce[idx].avg
        losses_ce_str += "%.4f"%losses_ce[idx].avg +'\t'
    losses_ce_record['avg.'] = losses_ce[args.num_heads].avg
    writer.add_scalars('loss-ce', losses_ce_record, epoch)
    writer.add_scalar('loss-ortho', losses_ortho.avg, epoch)
    writer.add_scalar('loss-margin', losses_margin.avg, epoch)
    print('Epoch %d/%d costs %fs.'%(epoch, args.epochs, batch_time.sum))
    print('     CE      loss of each path: \n'+losses_ce_str)
    print('     Avg. CE loss = %f.'%losses_ce_record['avg.'])
    print('     ORTHO   loss = %f.'%losses_ortho.avg)
    print('     MARGIN  loss = %f.'%losses_margin.avg)

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


# ======== startpoint
if __name__ == '__main__':
    main()