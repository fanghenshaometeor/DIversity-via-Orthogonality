# --------- Attack on DIO models

# --------- Attack vanilla-trained CIFAR10/CIFAR100-PRN18 DIO models
CUDA_VISIBLE_DEVICES=2 python attack_dio.py --arch preactresnet18 --num_heads 10 --dataset CIFAR10 --data_dir '~/KunFang/data/CIFAR10/' \
                    --model_path './save/CIFAR10/preactresnet18/p-10-a-0.1-b-0.1-tau-0.2/epoch100.pth' \
                    --attack_type 'None' #['None', 'fgsm', 'cw', 'square']
CUDA_VISIBLE_DEVICES=2 python attack_dio.py --arch preactresnet18 --num_heads 40 --dataset CIFAR100 --data_dir '~/KunFang/data/CIFAR100/' \
                    --model_path './save/CIFAR10/preactresnet18/p-40-a-0.1-b-0.1-tau-0.01/epoch100.pth' \
                    --attack_type 'None' #['None', 'fgsm', 'cw', 'square']

# -------- Attack adversarial-trained CIFAR10-PRN18/WRN34X10 DIO models (AT+DIO)
# ---- CIFAR10-PRN18 last model
CUDA_VISIBLE_DEVICES=2 python attack_dio.py --arch preactresnet18 --num_heads 10 --dataset CIFAR10 --data_dir '~/KunFang/data/CIFAR10/' \
                    --model_path './save/CIFAR10/preactresnet18-adv/p-10-a-0.1-b-0.1-tau-0.2/epoch100.pth' \        # last model
                    --attack_type 'None' #['None', 'pgd', 'pgd100', 'square', 'aa']
# ---- CIFAR10-PRN18 best model
CUDA_VISIBLE_DEVICES=2 python attack_dio.py --arch preactresnet18 --num_heads 10 --dataset CIFAR10 --data_dir '~/KunFang/data/CIFAR10/' \
                    --model_path './save/CIFAR10/preactresnet18-adv/p-10-a-0.1-b-0.1-tau-0.2/best.pth' \            # best model
                    --attack_type 'None' #['None', 'pgd', 'pgd100', 'square', 'aa']
# ---- CIFAR10-WRN34X10 last model
CUDA_VISIBLE_DEVICES=2 python attack_dio.py --arch wrn34x10 --num_heads 10 --dataset CIFAR10 --data_dir '~/KunFang/data/CIFAR10/' \
                    --model_path './save/CIFAR10/wrn34x10-adv/p-10-a-0.1-b-0.1-tau-0.5/epoch100.pth' \        # last model
                    --attack_type 'None' #['None', 'pgd', 'pgd100', 'square', 'aa']
# ---- CIFAR10-WRN34X10 best model
CUDA_VISIBLE_DEVICES=2 python attack_dio.py --arch wrn34x10 --num_heads 10 --dataset CIFAR10 --data_dir '~/KunFang/data/CIFAR10/' \
                    --model_path './save/CIFAR10/wrn34x10-adv/p-10-a-0.1-b-0.1-tau-0.5/best.pth' \        # best model
                    --attack_type 'None' #['None', 'pgd', 'pgd100', 'square', 'aa']

# -------- Attack adversarial-trained CIFAR100-WRN34X10/WRN34X20 DIO models (AT+DIO)
# ---- CIFAR100-WRN34X10 best model
CUDA_VISIBLE_DEVICES=2 python attack_dio.py --arch wrn34x10 --num_heads 10 --dataset CIFAR100 --data_dir '~/KunFang/data/CIFAR100/' \
                    --model_path './save/CIFAR10/wrn34x10-adv/p-10-a-0.1-b-0.1-tau-0.1/best.pth' \        # best model
                    --attack_type 'None' #['None', 'pgd', 'pgd100', 'square', 'aa']

# ---- CIFAR100-WRN34X20 best model
CUDA_VISIBLE_DEVICES=2 python attack_dio.py --arch wrn34x20 --num_heads 10 --dataset CIFAR100 --data_dir '~/KunFang/data/CIFAR100/' \
                    --model_path './save/CIFAR10/wrn34x20-adv/p-10-a-0.1-b-0.1-tau-0.2/best.pth' \        # best model
                    --attack_type 'None' #['None', 'pgd', 'pgd100', 'square', 'aa']

# -------- Attack adversarial-trained TinyImageNet-WRN34X10 DIO models (AT+DIO)
# ---- TinyImageNet-WRN34X10 best model
CUDA_VISIBLE_DEVICES=2 python attack_dio.py --arch wrn34x10 --num_heads 10 --dataset TinyImageNet --data_dir '~/KunFang/data/tinyimagenet/' \
                    --model_path './save/TinyImageNet/wrn34x10-adv/p-10-a-0.1-b-0.1-tau-0.1/best.pth' \        # best model
                    --attack_type 'None' #['None', 'pgd', 'pgd100', 'square', 'aa']
