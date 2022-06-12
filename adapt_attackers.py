import torch
import torch.nn.functional as F



# -------- PGD attack --------
def pgd_adapt(backbone, head, image, label, eps=0.031, alpha=0.008, iters=20, random_start=True, d_min=0, d_max=1):

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
        all_logits = head(backbone(perturbed_image), 'all')

        loss = .0
        for i in range(head.num_classifiers):
            logits = all_logits[i]
            loss += 1/head.num_classifiers * F.cross_entropy(logits, label)

        # loss = F.cross_entropy(logits, label)
        if perturbed_image.grad is not None:
            perturbed_image.grad.data.zero_()
        
        loss.backward()
        data_grad = perturbed_image.grad.data

        with torch.no_grad():
            perturbed_image.data += alpha * torch.sign(data_grad)
            perturbed_image.data = torch.max(torch.min(perturbed_image, image_max), image_min)
    perturbed_image.requires_grad = False
    
    return perturbed_image
