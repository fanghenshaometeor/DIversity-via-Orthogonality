"""
Created on Sat Feb 15 2020

@author: fanghenshao
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from advertorch.utils import NormalizeByChannelMeanStd

__all__ = ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out

class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out

class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks): #, num_classes=100):
        super(PreActResNet, self).__init__()
        self.in_planes = 64
        
        # default normalization is for CIFAR10
        self.normalize = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.normalize(x)
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn(out))
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        # out = self.linear(out)
        return out

class Ohead(nn.Module):

    def __init__(self, embedding_size=512, num_classes=10, num_classifiers=10):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.num_classifiers = num_classifiers

        # ---- initialize 10 classifiers
        clfs = []
        for i in range(num_classifiers):
            clfs.append(nn.Linear(embedding_size, num_classes, bias=False))
        self.classifiers = nn.Sequential(*clfs)
    
    # ---- orthogonality constraint
    def compute_ortho_loss(self):
        total = 0
        for i in range(self.num_classifiers):
            for param in self.classifiers[i].parameters():
                clf_i_param = param     # 10 x 512
            for j in range(i+1, self.num_classifiers):
                for param in self.classifiers[j].parameters():
                    clf_j_param = param     # 10 x 512
                
                inner_prod = clf_i_param.mul(clf_j_param).sum()
                total = total + inner_prod * inner_prod

            # print norm
            # for index in range(clf_i_param.size(0)):
            #     print(torch.norm(clf_i_param[index,:],p=2))

        return total

    # ---- compute l2 norm of ALL hyperplanes
    def _compute_l2_norm(self):

        norm_results = torch.ones(self.num_classifiers, self.num_classes).cuda()

        for idx in range(self.num_classifiers):
            clf = self.classifiers[idx]
            for param in clf.parameters():
                clf_param = param
            
            for jdx in range(clf_param.size(0)):
                hyperplane_norm = torch.norm(clf_param[jdx,:],p=2)

                norm_results[idx,jdx] = hyperplane_norm
        
        return norm_results
    
    # ---- compute l2 norm of SPECIFIED hyperplane
    def _compute_l2_norm_specified(self, idx):

        norm_results = torch.ones(self.num_classes).cuda()
        clf = self.classifiers[idx]
        for param in clf.parameters():
            clf_param = param
        
        for i in range(clf_param.size(0)):
            hyperplane_norm = torch.norm(clf_param[i,:],p=2)
            norm_results[i] = hyperplane_norm
        
        return norm_results

    def compute_margin_loss(self, all_logits, label, tau):
        loss_margin = .0
        # -------- find correctly-classified samples-clfs (x, clf-i)
        # -------- 对每条路径遍历
        for idx in range(self.num_classifiers):         # 对每条路径
            logits = all_logits[idx]                    # 读取当前路径对当前 batch 数据的预测 logits
            _, pred = torch.max(logits.data, 1)         # 获取当前路径对当前 batch 数据的预测结果
            corr_idx = (pred == label)                  # 获取当前路径预测正确的数据的索引

            if corr_idx.any() == False:                 # 如果当前路径对 batch 数据全部预测错误，即，b_data_idx 全部 false
                continue

            logits_correct = logits[corr_idx,:]                                                         # 依据索引，读取当前路径预测正确的那些数据的 logits
            corr_num = logits_correct.size(0)
            logits_correct_max, _ = torch.max(logits_correct, 1)                                        # 获取预测正确数据的 logits 的最大值，即，groundtruth 所在类的 logits
            logits_correct_max = logits_correct_max.unsqueeze(1).expand(corr_num, self.num_classes)     # 最大值复制

            hyperplane_norm = self._compute_l2_norm_specified(idx)                                      # 获取当前路径的超平面的 l2 norm
            hyperplane_norm = hyperplane_norm.repeat(corr_num,1)                                        # 数据复制，方便计算距离

            distance = torch.div(torch.abs(logits_correct-logits_correct_max), hyperplane_norm)         # 计算到最大值的 distance，这其中存在 0 值 
            distance = torch.where(distance>0, distance, torch.tensor(1000.0).cuda())                   # 去除 0 值
            margin, _ = torch.min(distance, 1)                                                          # 获取 margin

            loss_margin += (tau - margin).clamp(min=0).mean()
        
        return loss_margin

    # ---- forward 
    def forward(self, embedding, forward_type=0):

        """
        :param forward_type:
            'all':    return the predictions of ALL mutually-orthogonal paths
            'random': return the prediction  of ONE RANDOM path
            number:   return the prediction  of the SELECTED path
        """

        if forward_type == 'all':
            all_logits = []
            for idx in range(self.num_classifiers):
                logits = self.classifiers[idx](embedding)
                all_logits.append(logits)
            return all_logits
        
        elif forward_type == 'random':
            return self.classifiers[torch.randint(self.num_classifiers,(1,))](embedding)

        else:
            return self.classifiers[forward_type](embedding)

def preactresnet18(num_classes=10,num_classifiers=10):
    backbone = PreActResNet(PreActBlock, [2,2,2,2])
    head = Ohead(embedding_size=512*PreActBlock.expansion, num_classes=num_classes, num_classifiers=num_classifiers)
    # return PreActResNet(PreActBlock, [2,2,2,2], num_classes)
    return backbone, head