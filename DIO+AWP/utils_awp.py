from ast import Or
import torch
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
EPS = 1E-20


def diff_in_weights(model, proxy):
    diff_dict = OrderedDict()
    model_state_dict = model.state_dict()
    proxy_state_dict = proxy.state_dict()
    for (old_k, old_w), (new_k, new_w) in zip(model_state_dict.items(), proxy_state_dict.items()):
        if len(old_w.size()) <= 1:
            continue
        if 'weight' in old_k:
            diff_w = new_w - old_w
            diff_dict[old_k] = old_w.norm() / (diff_w.norm() + EPS) * diff_w
    return diff_dict


def add_into_weights(model, diff, coeff=1.0):
    names_in_diff = diff.keys()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff:
                param.add_(coeff * diff[name])


class AdvWeightPerturb(object):
    def __init__(self, model, proxy, proxy_optim, gamma):
        super(AdvWeightPerturb, self).__init__()
        self.model = model
        self.proxy = proxy
        self.proxy_optim = proxy_optim
        self.gamma = gamma

    def calc_awp(self, inputs_adv, targets):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()
        
        loss = - F.cross_entropy(self.proxy(inputs_adv), targets)

        self.proxy_optim.zero_grad()
        loss.backward()
        self.proxy_optim.step()

        # the adversary weight perturb
        diff = diff_in_weights(self.model, self.proxy)
        return diff

    def perturb(self, diff):
        add_into_weights(self.model, diff, coeff=1.0 * self.gamma)

    def restore(self, diff):
        add_into_weights(self.model, diff, coeff=-1.0 * self.gamma)

############################################
############################################
############################################

def diff_in_weights_dio(model, proxy):
    # diff_dict = OrderedDict()
    diff_dict_backbone = OrderedDict()
    diff_dict_head = OrderedDict()
    backbone, head = model
    backbone_proxy, head_proxy = proxy
    # model_state_dict = model.state_dict()
    # proxy_state_dict = proxy.state_dict()
    # for (old_k, old_w), (new_k, new_w) in zip(model_state_dict.items(), proxy_state_dict.items()):
    #     if len(old_w.size()) <= 1:
    #         continue
    #     if 'weight' in old_k:
    #         diff_w = new_w - old_w
    #         diff_dict[old_k] = old_w.norm() / (diff_w.norm() + EPS) * diff_w
    backbone_state_dict = backbone.state_dict()
    head_state_dict = head.state_dict()
    backbone_proxy_state_dict = backbone_proxy.state_dict()
    head_proxy_state_dict = head_proxy.state_dict()
    for (old_k, old_w), (new_k, new_w) in zip(backbone_state_dict.items(), backbone_proxy_state_dict.items()):
        if len(old_w.size()) <= 1:
            continue
        if 'weight' in old_k:
            diff_w = new_w - old_w
            diff_dict_backbone[old_k] = old_w.norm() / (diff_w.norm() + EPS) * diff_w
    for (old_k, old_w), (new_k, new_w) in zip(head_state_dict.items(), head_proxy_state_dict.items()):
        if len(old_w.size()) <= 1:
            continue
        if 'weight' in old_k:
            diff_w = new_w - old_w
            diff_dict_head[old_k] = old_w.norm() / (diff_w.norm() + EPS) * diff_w
    return [diff_dict_backbone, diff_dict_head]

def add_into_weights_dio(model, diff, coeff=1.0):
    # names_in_diff = diff.keys()
    # with torch.no_grad():
    #     for name, param in model.named_parameters():
    #         if name in names_in_diff:
    #             param.add_(coeff * diff[name])
    backbone, head = model
    diff_backbone, diff_head = diff
    names_in_diff_backbone = diff_backbone.keys()
    names_in_diff_head = diff_head.keys()
    with torch.no_grad():
        for name, param in backbone.named_parameters():
            if name in names_in_diff_backbone:
                param.add_(coeff * diff_backbone[name])
        for name, param in head.named_parameters():
            if name in names_in_diff_head:
                param.add_(coeff * diff_head[name])


class AdvWeightPerturb_DIO(object):
    def __init__(self, model, proxy, proxy_optim, gamma):
        super(AdvWeightPerturb_DIO, self).__init__()
        ############################################
        self.backbone, self.head = model
        self.backbone_proxy, self.head_proxy = proxy
        ############################################
        self.proxy_optim = proxy_optim
        self.gamma = gamma
        
    def calc_awp(self, inputs_adv, targets):
        # self.proxy.load_state_dict(self.model.state_dict())
        # self.proxy.train()
        
        # loss = - F.cross_entropy(self.proxy(inputs_adv), targets)

        ############################################
        self.backbone_proxy.load_state_dict(self.backbone.state_dict())
        self.head_proxy.load_state_dict(self.head.state_dict())
        self.backbone_proxy.train()
        self.head_proxy.train()

        loss = - F.cross_entropy(self.head_proxy(self.backbone_proxy(inputs_adv),'random'), targets)
        ############################################

        self.proxy_optim.zero_grad()
        loss.backward()
        self.proxy_optim.step()

        # the adversary weight perturb
        diff = diff_in_weights_dio([self.backbone, self.head], [self.backbone_proxy, self.head_proxy])
        return diff

    def perturb(self, diff):
        add_into_weights_dio([self.backbone, self.head], diff, coeff=1.0 * self.gamma)

    def restore(self, diff):
        add_into_weights_dio([self.backbone, self.head], diff, coeff=-1.0 * self.gamma)



