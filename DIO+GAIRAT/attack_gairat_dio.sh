# -------- Attack GAIRAT+DIO: CIFAR10-PRN18/WRN34X10
# ---- CIFAR10-PRN18 last model
CUDA_VISIBLE_DEVICES=2 python attack_gairat_dio.py --arch preactresnet18 --num_heads 40 --dataset CIFAR10 --data_dir '~/KunFang/data/CIFAR10/' \
                    --model_path './save/CIFAR10/preactresnet18/AWP+DIO/p-40-a-0.1-b-0.1-tau-0.02/epoch100.pth' \        # last model
                    --attack_type 'None' #['None', 'pgd, 'pgd100', 'square', 'aa']
# ---- CIFAR10-PRN18 best model
CUDA_VISIBLE_DEVICES=2 python attack_gairat_dio.py --arch preactresnet18 --num_heads 40 --dataset CIFAR10 --data_dir '~/KunFang/data/CIFAR10/' \
                    --model_path './save/CIFAR10/preactresnet18/AWP+DIO/p-40-a-0.1-b-0.1-tau-0.02/best.pth' \            # best model
                    --attack_type 'None' #['None', 'pgd, 'pgd100', 'square', 'aa']
# ---- CIFAR10-WRN34X10 last model
CUDA_VISIBLE_DEVICES=2 python attack_gairat_dio.py --arch wrn34x10 --num_heads 40 --dataset CIFAR10 --data_dir '~/KunFang/data/CIFAR10/' \
                    --model_path './save/CIFAR10/wrn34x10/AWP+DIO/p-40-a-0.1-b-0.1-tau-0.02/epoch100.pth' \        # last model
                    --attack_type 'None' #['None', 'pgd, 'pgd100', 'square', 'aa']
# ---- CIFAR10-WRN34X10 best model
CUDA_VISIBLE_DEVICES=2 python attack_gairat_dio.py --arch wrn34x10 --num_heads 40 --dataset CIFAR10 --data_dir '~/KunFang/data/CIFAR10/' \
                    --model_path './save/CIFAR10/wrn34x10/AWP+DIO/p-40-a-0.1-b-0.1-tau-0.02/best.pth' \        # best model
                    --attack_type 'None' #['None', 'pgd, 'pgd100', 'square', 'aa']