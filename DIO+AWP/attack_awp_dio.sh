# -------- Attack AWP+DIO: CIFAR10-PRN18/WRN34X10
# ---- CIFAR10-PRN18 last model
CUDA_VISIBLE_DEVICES=2 python attack_awp_dio.py --arch preactresnet18 --num_heads 20 --dataset CIFAR10 --data_dir '~/KunFang/data/CIFAR10/' \
                    --model_path './save/CIFAR10/preactresnet18/DIO+AWP/p-20-a-0.1-b-0.1-tau-0.001/epoch200.pth' \        # last model
                    --attack_type 'None' #['None', 'pgd, 'pgd100', 'square', 'aa']
# ---- CIFAR10-PRN18 best model
CUDA_VISIBLE_DEVICES=2 python attack_awp_dio.py --arch preactresnet18 --num_heads 20 --dataset CIFAR10 --data_dir '~/KunFang/data/CIFAR10/' \
                    --model_path './save/CIFAR10/preactresnet18/DIO+AWP/p-20-a-0.1-b-0.1-tau-0.001/best.pth' \            # best model
                    --attack_type 'None' #['None', 'pgd, 'pgd100', 'square', 'aa']
# ---- CIFAR10-WRN34X10 last model
CUDA_VISIBLE_DEVICES=2 python attack_awp_dio.py --arch wrn34x10 --num_heads 10 --dataset CIFAR10 --data_dir '~/KunFang/data/CIFAR10/' \
                    --model_path './save/CIFAR10/wrn34x10/DIO+AWP/p-10-a-0.1-b-0.1-tau-0.01/epoch200.pth' \        # last model
                    --attack_type 'None' #['None', 'pgd, 'pgd100', 'square', 'aa']
# ---- CIFAR10-WRN34X10 best model
CUDA_VISIBLE_DEVICES=2 python attack_awp_dio.py --arch wrn34x10 --num_heads 10 --dataset CIFAR10 --data_dir '~/KunFang/data/CIFAR10/' \
                    --model_path './save/CIFAR10/wrn34x10/DIO+AWP/p-10-a-0.1-b-0.1-tau-0.01/best.pth' \        # best model
                    --attack_type 'None' #['None', 'pgd, 'pgd100', 'square', 'aa']