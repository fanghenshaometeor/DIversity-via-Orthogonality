import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride,
            padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=1,
            padding=1, bias=False)
        self.drop_rate = drop_rate
        self.in_out_equal = (in_planes == out_planes)

        if not self.in_out_equal:
            self.conv_shortcut = nn.Conv2d(
                in_planes, out_planes, kernel_size=1, stride=stride,
                padding=0, bias=False)

    def forward(self, x):
        out = self.relu1(self.bn1(x))
        if not self.in_out_equal:
            x = self.conv_shortcut(out)
        out = self.relu2(self.bn2(self.conv1(out)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        out += x
        return out


class ConvGroup(nn.Module):
    def __init__(
            self, num_blocks, in_planes, out_planes, block, stride,
            drop_rate=0.0):
        super(ConvGroup, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, num_blocks, stride, drop_rate)

    def _make_layer(
            self, block, in_planes, out_planes, num_blocks, stride, drop_rate):
        layers = []
        for i in range(int(num_blocks)):
            layers.append(
                block(in_planes=in_planes if i == 0 else out_planes,
                      out_planes=out_planes,
                      stride=stride if i == 0 else 1,
                      drop_rate=drop_rate)
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    # def __init__(self, depth, num_classes, widen_factor=1, drop_rate=0.0,
    #              color_channels=3, block=BasicBlock):
    def __init__(self, depth, widen_factor=1, drop_rate=0.0,
                 color_channels=3, block=BasicBlock):
        super(WideResNet, self).__init__()
        num_channels = [
            16, int(16 * widen_factor),
            int(32 * widen_factor), int(64 * widen_factor)]
        assert((depth - 4) % 6 == 0)
        num_blocks = (depth - 4) / 6

        self.conv1 = nn.Conv2d(
            color_channels, num_channels[0], kernel_size=3, stride=1,
            padding=1, bias=False)
        self.convgroup1 = ConvGroup(
            num_blocks, num_channels[0], num_channels[1], block, 1, drop_rate)
        self.convgroup2 = ConvGroup(
            num_blocks, num_channels[1], num_channels[2], block, 2, drop_rate)
        self.convgroup3 = ConvGroup(
            num_blocks, num_channels[2], num_channels[3], block, 2, drop_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(num_channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.num_channels = num_channels[3]
        # self.fc = nn.Linear(num_channels[3], num_classes)

        for mod in self.modules():
            if isinstance(mod, nn.Conv2d):
                n = mod.kernel_size[0] * mod.kernel_size[1] * mod.out_channels
                mod.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(mod, nn.BatchNorm2d):
                mod.weight.data.fill_(1)
                mod.bias.data.zero_()
            elif isinstance(mod, nn.Linear):
                mod.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.convgroup1(out)
        out = self.convgroup2(out)
        out = self.convgroup3(out)
        out = self.relu(self.bn1(out))
        out = out.mean(dim=-1).mean(dim=-1)
        # print(out.size())
        # out = self.fc(out)
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
    def _orthogonal_costr(self):
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

def wideresnet_28_5(num_classes=10, num_classifiers=10):
    backbone = WideResNet(depth=28, widen_factor=5)
    head = Ohead(embedding_size=64*5, num_classes=num_classes, num_classifiers=num_classifiers)
    return backbone, head

def wideresnet_28_10(num_classes=10, num_classifiers=10):
    backbone = WideResNet(depth=28, widen_factor=10)
    head = Ohead(embedding_size=64*10, num_classes=num_classes, num_classifiers=num_classifiers)
    return backbone, head

def wideresnet_34_10(num_classes=10, num_classifiers=10):
    backbone = WideResNet(depth=34, widen_factor=10)
    head = Ohead(embedding_size=64*10, num_classes=num_classes, num_classifiers=num_classifiers)
    return backbone, head

# if __name__ == '__main__':
#     # backbone, head = wideresnet_28_5()
#     backbone, head = wideresnet_34_10()
#     print(backbone)
#     print(head)
#     check_input = torch.randn(10,3,32,32)
#     embedding = backbone(check_input)
#     all_logits = head(embedding, 'all')
#     rad_logits = head(embedding, 'random')
#     logits = head(embedding)
#     print('input size: ', check_input.size())
#     print('embedding size: ', embedding.size())
#     print(len(all_logits))
#     print(rad_logits.size())
#     print(logits.size())