# -------- model:vgg16/resnet18 -----------------
# model=vgg11
# model=vgg13
model=vgg16
# model=vgg19
# model=resnet20
# model=resnet32
# model=wrn28x5
# model=wrn28x10
# -------- hyper-parameters ---------------------
alpha=0.1
beta=0.1
# tau=4.0
# tau=3.0
# tau=2.5
# tau=2.0
# tau=1.5
tau=1.0
# tau=0.5
# tau=0.2
# tau=0.1
# tau=0.05
# tau=0.02
# tau=0.01
# ----
# tau_adv=2.5
# tau_adv=2.0
# tau_adv=1.0
# tau_adv=0.5
# tau_adv=0.2
# tau_adv=0.1
# tau_adv=0.05
# tau_adv=0.02
tau_adv=0.01
# ----
num_heads=10
# num_heads=20
# num_heads=40
# -------- CIFAR10 ------------------------------
dataset=CIFAR10
data_dir='/media/Disk1/KunFang/data/CIFAR10/'
# -------- CIFAR100 -----------------------------
# dataset=CIFAR100
# data_dir='/media/Disk1/KunFang/data/CIFAR100/'
# -------- SVNH ---------------------------------
# dataset=SVHN
# data_dir='/media/Disk1/KunFang/data/SVHN/'
# -----------------------------------------------
# adv_train=False
adv_train=True
# -----------------------------------------------

CUDA_VISIBLE_DEVICES=3 python train.py \
    --model ${model} \
    --alpha ${alpha} \
    --beta ${beta} \
    --num_heads ${num_heads} \
    --tau ${tau} \
    --tau_adv ${tau_adv} \
    --dataset ${dataset} \
    --data_dir ${data_dir} \
    --adv_train ${adv_train}
