"""
Created on Mon Feb 24 2020

@author: fanghenshao
"""

import torch
import numpy as np
import random

import os
import sys

sys.path.append("model")
import wideresnet
import dio_wideresnet
import preactresnet
import dio_preactresnet


from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from advertorch.utils import NormalizeByChannelMeanStd

# -------- fix random seed 
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# -------- for DDP training
def reduce_tensor(tensor: torch.Tensor):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    # rt /= torch.distributed.get_world_size()
    return rt

# -------- get the number of trainable parameters
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

########################################################################################################
########################################################################################################
########################################################################################################

def cifar10_dataloaders(data_dir, batch_size=256):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = CIFAR10(data_dir, train=True, transform=train_transform, download=True)
    test_set = CIFAR10(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return train_loader, test_loader

def cifar100_dataloaders(data_dir, batch_size=256):

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = CIFAR100(data_dir, train=True, transform=train_transform, download=True)
    test_set = CIFAR100(data_dir, train=False, transform=test_transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return train_loader, test_loader

def get_datasets(args):
    if args.dataset == 'CIFAR10':
        return cifar10_dataloaders(data_dir=args.data_dir, batch_size=args.batch_size)
    elif args.dataset == 'CIFAR100':
        return cifar100_dataloaders(data_dir=args.data_dir, batch_size=args.batch_size)
    else:
        assert False, "Unknown dataset : {}".format(args.dataset)


########################################################################################################
########################################################################################################
########################################################################################################

def get_model(args):
    if args.dataset == 'CIFAR10':
        num_class = 10
        dataset_normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])

    elif args.dataset == 'CIFAR100':
        num_class = 100
        dataset_normalization = NormalizeByChannelMeanStd(
            mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])

    else:
        assert False, "Unknown dataset : {}".format(args.dataset)

    if args.arch == 'preactresnet18':
        net = preactresnet.__dict__[args.arch](num_classes=num_class)
    elif 'wrn' in args.arch:
        net = wideresnet.__dict__[args.arch](num_classes=num_class)
    else:
        assert False, "Unknown model : {}".format(args.arch)

    net.normalize = dataset_normalization

    return net

def get_model_dio(args):
    if args.dataset == 'CIFAR10':
        num_class = 10
        dataset_normalization = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])

    elif args.dataset == 'CIFAR100':
        num_class = 100
        dataset_normalization = NormalizeByChannelMeanStd(
            mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762])

    else:
        assert False, "Unknown dataset : {}".format(args.dataset)

    if args.arch == 'preactresnet18':
        backbone, head = dio_preactresnet.__dict__[args.arch](num_classes=num_class,num_classifiers=args.num_heads)
    elif 'wrn' in args.arch:
        backbone, head = dio_wideresnet.__dict__[args.arch](num_classes=num_class,num_classifiers=args.num_heads)
    else:
        assert False, "Unknown model : {}".format(args.arch)

    # net.normalize = dataset_normalization
    backbone.normalize = dataset_normalization

    # return net
    return backbone, head

########################################################################################################
########################################################################################################
########################################################################################################

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def get_net_param_vec(net):
    net_vec = []
    with torch.no_grad():
        for _, param in net.named_parameters():
            param = param.view(-1)
            net_vec.append(param.detach().cpu().numpy())
        net_vec = np.concatenate(net_vec, 0)
    return net_vec

########################################################################################################
########################################################################################################
########################################################################################################

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass