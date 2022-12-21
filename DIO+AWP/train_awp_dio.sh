# -------- Training AWP+DIO: CIFAR10-PRN18/WRN34X10
# ---- CIFAR10-PRN18
CUDA_VISIBLE_DEVICES=2 python train_awp_dio.py --arch preactresnet18 --dataset CIFAR10 --data_dir '~/KunFang/data/CIFAR10/' \
                            --alpha 0.1 --beta 0.1 --tau 0.001 --num_heads 20 -epochs 200
# ---- CIFAR10-WRN34X10
CUDA_VISIBLE_DEVICES=2 python train_awp_dio.py --arch wrn34x10 --dataset CIFAR10 --data_dir '~/KunFang/data/CIFAR10/' \
                            --alpha 0.1 --beta 0.1 --tau 0.01 --num_heads 10 -epochs 200