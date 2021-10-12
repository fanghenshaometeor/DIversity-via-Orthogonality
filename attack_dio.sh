# --------- CIFAR10-vgg/resnet -----------------
# model=vgg11
# model_path='./save/CIFAR10/vgg11-p-10-a-0.1-b-0.1-tau-0.5.pth'
# model_path='./save/CIFAR10/vgg11-p-10-a-0.1-b-0.1-tau-1.0.pth'
# model_path='./save/CIFAR10/vgg11-p-10-a-0.1-b-0.1-tau-2.0.pth'
# model_path='./save/CIFAR10/vgg11-p-10-a-0.1-b-0.1-tau-3.0.pth'    # hhh
# model_path='./save/CIFAR10/vgg11-p-10-a-0.1-b-0.1-tau-4.0.pth'
# model_path='./save/CIFAR10/ablation-dist-vgg11-p-10-a-0.1-b-0.0-tau-3.0.pth'
# model_path='./save/CIFAR10/ablation-ortho-vgg11-p-1-a-0.1-b-0.1-tau-3.0.pth'
# model=vgg13
# model_path='./save/CIFAR10/vgg13-p-10-a-0.1-b-0.1-tau-2.0.pth'
# model=vgg16
# model_path='./save/CIFAR10/vgg16-p-10-a-0.1-b-0.1-tau-0.5.pth'
# model_path='./save/CIFAR10/vgg16-p-10-a-0.1-b-0.1-tau-0.5-taua-0.5-adv.pth'
# model=vgg19
# model_path='./save/CIFAR10/vgg19-p-10-a-0.1-b-0.1-tau-0.02.pth'
# model=resnet20
# model_path='./save/SVHN/resnet20-p-10-a-0.1-b-0.1-tau-0.5.pth'
# model_path='./save/SVHN/resnet20-p-10-a-0.1-b-0.1-tau-0.2.pth'
# model_path='./save/SVHN/resnet20-p-10-a-0.1-b-0.1-tau-0.1.pth'      # hhh
# model_path='./save/SVHN/resnet20-p-10-a-0.1-b-0.1-tau-0.05.pth'
# model=resnet32
# model_path='./save/SVHN/resnet32-p-10-a-0.1-b-0.1-tau-0.5.pth'
# model_path='./save/SVHN/resnet32-p-10-a-0.1-b-0.1-tau-0.2.pth'
# model_path='./save/SVHN/resnet32-p-10-a-0.1-b-0.1-tau-0.1.pth'
# model_path='./save/SVHN/resnet32-p-10-a-0.1-b-0.1-tau-0.05.pth'       # hhh
# model_path='./save/SVHN/resnet32-p-10-a-0.1-b-0.1-tau-0.02.pth'
# model=wrn28x5
# model_path='./save/CIFAR100/wrn28x5-p-10-a-0.1-b-0.1-tau-0.05.pth'
# model_path='./save/CIFAR100/wrn28x5-p-10-a-0.1-b-0.1-tau-0.1.pth'
# model_path='./save/CIFAR100/wrn28x5-p-10-a-0.1-b-0.1-tau-0.2.pth'       # hhh
# model_path='./save/CIFAR100/wrn28x5-p-10-a-0.1-b-0.1-tau-0.5.pth'
# model_path='./save/CIFAR100/wrn28x5-p-10-a-0.1-b-0.1-tau-1.0.pth'
# model_path='./save/CIFAR100/ablation-ortho-wrn28x5-p-1-a-0.1-b-0.1-tau-0.2.pth'
# model_path='./save/CIFAR100/ablation-dist-wrn28x5-p-10-a-0.1-b-0.0-tau-0.2.pth'
model=wrn28x10
# model_path='./save/CIFAR100/wrn28x10-p-10-a-0.1-b-0.1-tau-0.1-temp.pth'
# model_path='./save/CIFAR100/wrn28x10-p-10-a-0.1-b-0.1-tau-0.05-temp.pth'    # hhh
model_path='./save/CIFAR100/wrn28x10-p-10-a-0.1-b-0.1-tau-0.1-taua-0.01-adv.pth'
# -------- hyper-parameters ---------------------
# num_heads=1
# num_heads=5
num_heads=10
# num_heads=20
# num_heads=40
# -------- CIFAR10 ------------------------------
# dataset=CIFAR10
# data_dir='/media/Disk1/KunFang/data/CIFAR10/'
# -------- CIFAR100 -----------------------------
dataset=CIFAR100
data_dir='/media/Disk1/KunFang/data/CIFAR100/'
# -------- SVNH ---------------------------------
# dataset=SVHN
# data_dir='/media/Disk1/KunFang/data/SVHN/'
# -----------------------------------------------
# attack_type='fgsm'
# attack_type='pgd'
# attack_type='cw'
# attack_type='square'
# attack_type='fab'
attack_type='aa'
# -----------------------------------------------

CUDA_VISIBLE_DEVICES=0 python attack_dio.py \
    --model ${model} \
    --model_path ${model_path} \
    --dataset ${dataset} \
    --data_dir ${data_dir} \
    --num_heads ${num_heads} \
    --attack_type ${attack_type}

