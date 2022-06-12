# -------- Vanilla-training DIO: CIFAR10/CIFAR100-PRN18
# ---- CIFAR10-PRN18
CUDA_VISIBLE_DEVICES=2 python train.py --arch preactresnet18 --dataset CIFAR10 --data_dir '~/KunFang/data/CIFAR10/' --adv_train False \
                                    --alpha 0.1 --beta 0.1 --tau 0.2 --num_heads 10
# ---- CIFAR100-PRN18
CUDA_VISIBLE_DEVICES=2 python train.py --arch preactresnet18 --dataset CIFAR100 --data_dir '~/KunFang/data/CIFAR100/' --adv_train False \
                                    --alpha 0.1 --beta 0.1 --tau 0.01 --num_heads 40
                                
# -------- Adversarial-training DIO (AT+DIO): CIFAR10-PRN18/WRN34X10
# ---- CIFAR10-PRN18
CUDA_VISIBLE_DEVICES=2 python train.py --arch preactresnet18 --dataset CIFAR10 --data_dir '~/KunFang/data/CIFAR10/' --adv_train True \
                                    --alpha 0.1 --beta 0.1 --tau 0.2 --num_heads 10
# ---- CIFAR10-WRN34X10
CUDA_VISIBLE_DEVICES=2 python train.py --arch wrn34x10 --dataset CIFAR10 --data_dir '~/KunFang/data/CIFAR10/' --adv_train True \
                                    --alpha 0.1 --beta 0.1 --tau 0.5 --num_heads 10
