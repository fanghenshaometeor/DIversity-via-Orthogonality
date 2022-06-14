# --------- Adaptive attack on adversarial-trained CIFAR10-WRN34X10 DIO models (AT+DIO)
# ---- Adaptive-Attack-1
CUDA_VISIBLE_DEVICES=2 python adapt_attack.py --arch wrn34x10 --num_heads 10 --dataset CIFAR10 --data_dir '~/KunFang/data/CIFAR10/' \
                    --model_path './save/CIFAR10/wrn34x10-adv/p-10-a-0.1-b-0.1-tau-0.5/epoch100.pth' \        # last model
                    --attack_type 'pgd' #['pgd', 'pgd100']
                    --adapt1
# ---- Adaptive-Attack-2
CUDA_VISIBLE_DEVICES=2 python adapt_attack.py --arch wrn34x10 --num_heads 10 --dataset CIFAR10 --data_dir '~/KunFang/data/CIFAR10/' \
                    --model_path './save/CIFAR10/wrn34x10-adv/p-10-a-0.1-b-0.1-tau-0.5/epoch100.pth' \        # last model
                    --attack_type 'pgd' #['pgd', 'pgd100']

