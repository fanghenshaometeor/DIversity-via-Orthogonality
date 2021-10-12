import torch
import torch.nn as nn

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):

    def __init__(self, features):
        super().__init__()
        self.features = features
    
    def forward(self, x):
        embedding = self.features(x)
        embedding = embedding.view(embedding.size(0), -1)
        return embedding

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



def make_layers(cfg, batch_norm=True):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]#, bias=False)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]
        
        layers += [nn.ReLU(inplace=True)]
        input_channel = l
    
    return nn.Sequential(*layers)



def vgg11_bn(embedding_size=512, num_classes=10, num_classifiers=10):
    backbone = VGG(make_layers(cfg['A']))
    head = Ohead(embedding_size=embedding_size, num_classes=num_classes, num_classifiers=num_classifiers)
    return backbone, head

def vgg13_bn(embedding_size=512, num_classes=10, num_classifiers=10):
    backbone = VGG(make_layers(cfg['B']))
    head = Ohead(embedding_size=embedding_size, num_classes=num_classes, num_classifiers=num_classifiers)
    return backbone, head

def vgg16_bn(embedding_size=512, num_classes=10, num_classifiers=10):
    backbone = VGG(make_layers(cfg['D']))
    head = Ohead(embedding_size=embedding_size, num_classes=num_classes, num_classifiers=num_classifiers)
    return backbone, head

def vgg19_bn(embedding_size=512, num_classes=10, num_classifiers=10):
    backbone = VGG(make_layers(cfg['E']))
    head = Ohead(embedding_size=embedding_size, num_classes=num_classes, num_classifiers=num_classifiers)
    return backbone, head

# if __name__ == '__main__':
#     net = vgg11_bn()
#     backbone, head = net
#     print(backbone)
#     print(head)

#     print(backbone.parameters())
#     print(head.parameters())

    # check norm
    # head._orthogonal_costr()
    # check_norm = head._compute_l2_norm()
    # check_norm_idx = head._compute_l2_norm_specified(0)
    # print(check_norm)
    # print(check_norm_idx)


    # check size
    # backbone, head = vgg11_bn()
    # print(backbone)
    # print(head)
    # check_input = torch.randn(10,3,32,32)
    # embedding = backbone(check_input)
    # all_logits = head(embedding, 'all')
    # rad_logits = head(embedding, 'random')
    # logits = head(embedding)
    # print('input size: ', check_input.size())
    # print('embedding size: ', embedding.size())
    # print(len(all_logits))
    # print(rad_logits.size())
    # print(logits.size())



