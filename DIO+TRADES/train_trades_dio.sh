# -------- Training TRADES+DIO: CIFAR10-PRN18/WRN34X10/WRN34X20
# ---- CIFAR10-PRN18
CUDA_VISIBLE_DEVICES=2 python train_trades_dio.py --arch preactresnet18 --dataset CIFAR10 --data_dir '~/KunFang/data/CIFAR10/' \
                            --alpha 0.1 --beta 0.1 --tau 0.1 --num_heads 10
# ---- CIFAR10-WRN34X10
CUDA_VISIBLE_DEVICES=2 python train_trades_dio.py --arch wrn34x10 --dataset CIFAR10 --data_dir '~/KunFang/data/CIFAR10/' \
                            --alpha 0.1 --beta 0.1 --tau 0.1 --num_heads 10

# ---- CIFAR10-WRN34X20
CUDA_VISIBLE_DEVICES=2 python train_trades_dio.py --arch wrn34x20 --dataset CIFAR10 --data_dir '~/KunFang/data/CIFAR10/' \
                            --alpha 0.1 --beta 0.1 --tau 0.1 --num_heads 10

# -------- Training TRADES+DIO: CIFAR100-WRN34X10
CUDA_VISIBLE_DEVICES=2 python train_trades_dio.py --arch wrn34x10 --dataset CIFAR100 --data_dir '~/KunFang/data/CIFAR100/' \
                            --alpha 0.1 --beta 0.1 --tau 0.2 --num_heads 10
