import torch
import torch.nn.functional as F


# -------- FGSM attack --------
def fgsm_attack(backbone, head, idx, image, label, epsilon):
    image.requires_grad = True

    logits = head(backbone(image), idx)
    loss = F.cross_entropy(logits, label)
    backbone.zero_grad()
    head.zero_grad()
    loss.backward()

    # collect data grad
    perturbed_image = image + epsilon*image.grad.data.sign()
    # clip the perturbed image into [0,1]
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

# -------- PGD attack --------
def pgd_attack(backbone, head, image, label, eps, alpha=0.01, iters=7, random_start=True, d_min=0, d_max=1):

    perturbed_image = image.clone()
    perturbed_image.requires_grad = True

    image_max = image + eps
    image_min = image - eps
    image_max.clamp_(d_min, d_max)
    image_min.clamp_(d_min, d_max)

    if random_start:
        with torch.no_grad():
            perturbed_image.data = image + perturbed_image.uniform_(-1*eps, eps)
            perturbed_image.data.clamp_(d_min, d_max)
    
    for idx in range(iters):
        backbone.zero_grad()
        head.zero_grad()
        logits = head(backbone(perturbed_image), 'random')

        loss = F.cross_entropy(logits, label)
        if perturbed_image.grad is not None:
            perturbed_image.grad.data.zero_()
        
        loss.backward()
        data_grad = perturbed_image.grad.data

        with torch.no_grad():
            perturbed_image.data += alpha * torch.sign(data_grad)
            perturbed_image.data = torch.max(torch.min(perturbed_image, image_max), image_min)
    perturbed_image.requires_grad = False
    
    return perturbed_image

# -------- c&w attack --------
def tanh_space(x):
    return 1/2*(torch.tanh(x) + 1)

def inverse_tanh_space(x):
    # torch.atanh is only for torch >= 1.7.0
    return atanh(x*2-1) 

def atanh(x):
    return 0.5*torch.log((1+x)/(1-x))

def f(outputs, labels, kappa):
    one_hot_labels = torch.eye(len(outputs[0]))[labels].cuda()

    i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
    j = torch.masked_select(outputs, one_hot_labels.bool())

    return torch.clamp((-1)*(i-j), min=-kappa)

def cw_attack(backbone, head, idx, image, label, c, kappa, n_iters, lr):
    image = image.clone().detach().cuda()
    label = label.clone().detach().cuda()

    w = inverse_tanh_space(image).detach()
    w.requires_grad = True

    best_adv_images = image.clone().detach()
    best_L2 = 1e10*torch.ones((len(image))).cuda()
    prev_cost = 1e10
    dim = len(image.shape)

    MSELoss = nn.MSELoss(reduction='none')
    Flatten = nn.Flatten()

    optimizer = optim.Adam([w], lr=lr)

    for step in range(n_iters):
        adv_images = tanh_space(w)

        current_L2 = MSELoss(Flatten(adv_images), Flatten(image)).sum(dim=1)
        L2_loss = current_L2.sum()
        
        outputs = head(backbone(adv_images), idx)
        f_loss = f(outputs, label, kappa).sum()

        cost = L2_loss + c*f_loss

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        # Update Adversarial Images
        _, pre = torch.max(outputs.detach(), 1)
        correct = (pre == label).float()
        
        mask = (1-correct)*(best_L2 > current_L2.detach())
        best_L2 = mask*current_L2.detach() + (1-mask)*best_L2
        
        mask = mask.view([-1]+[1]*(dim-1))
        best_adv_images = mask*adv_images.detach() + (1-mask)*best_adv_images
        
        # Early Stop when loss does not converge.
        if step % (n_iters//10) == 0:
            if cost.item() > prev_cost:
                return best_adv_images
            prev_cost = cost.item()
            
    return best_adv_images