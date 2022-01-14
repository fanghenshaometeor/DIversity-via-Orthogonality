# -------- model:vgg16/resnet18 -----------------
# arch=wrn34x10
arch=preactresnet18
# -------- hyper-parameters ---------------------
alpha=0.1
beta=0.1
tau=1.0
# ----
num_heads=10
# -------- CIFAR10 ------------------------------
dataset=CIFAR10
data_dir='~/KunFang/data/CIFAR10/'
# -------- CIFAR100 -----------------------------
# dataset=CIFAR100
# data_dir='~/KunFang/data/CIFAR100/'
# -------- SVNH ---------------------------------
# dataset=SVHN
# data_dir='~/KunFang/data/SVHN/'
# -----------------------------------------------
# adv_train=False
adv_train=True
# -----------------------------------------------

CUDA_VISIBLE_DEVICES=1 python train.py \
    --arch ${arch} \
    --num_heads ${num_heads} \
    --alpha ${alpha} \
    --beta ${beta} \
    --tau ${tau} \
    --dataset ${dataset} \
    --data_dir ${data_dir} \
    --adv_train ${adv_train}
