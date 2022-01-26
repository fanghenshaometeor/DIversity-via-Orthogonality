# --------- CIFAR10-vgg/resnet -----------------
arch=preactresnet18
# model_path='./save/CIFAR10/preactresnet18/p-10-a-0.1-b-0.1-tau-0.5/epoch100.pth'
model_path='./save/CIFAR100/preactresnet18/p-10-a-0.1-b-0.1-tau-0.5/epoch100.pth'
# -------- hyper-parameters ---------------------
# num_heads=1
# num_heads=5
num_heads=10
# num_heads=20
# num_heads=40
# -------- CIFAR10 ------------------------------
# dataset=CIFAR10
# data_dir='~/KunFang/data/CIFAR10/'
# -------- CIFAR100 -----------------------------
dataset=CIFAR100
data_dir='~/KunFang/data/CIFAR100/'
# -------- SVNH ---------------------------------
# dataset=SVHN
# data_dir='/media/Disk1/KunFang/data/SVHN/'
# -----------------------------------------------
attack_type='fgsm'
# attack_type='pgd'
# attack_type='cw'
# attack_type='square'
# attack_type='fab'
# attack_type='aa'
# -----------------------------------------------

CUDA_VISIBLE_DEVICES=0 python attack_dio.py \
    --arch ${arch} \
    --model_path ${model_path} \
    --dataset ${dataset} \
    --data_dir ${data_dir} \
    --num_heads ${num_heads} \
    --attack_type ${attack_type}

