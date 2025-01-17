import os
import time
import argparse

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Domain generalization')

parser.add_argument('--device', type=str, default='cuda:1')
parser.add_argument('--seed', type=int, default=1)

args = parser.parse_args()

hparams_dict = {
    "UShift2":"""{"batch_size": 256, 
                    "arch": "resnet18",
                    "class_balanced": false, 
                    "data_augmentation": true, 
                    "lr": 5.6841898471378446e-05, 
                    "nonlinear_classifier": false, 
                    "resnet_dropout": 0.1,
                    "weight_decay": 0.005100223533962902}""",
}

for algo in ["UShift2"]: 
    for dataset in ["ColoredMNIST_WL"]:
        hparams = hparams_dict[algo]
        print(f"Train {algo} on {dataset}")
        os.system(f"""python3 -m domainbed.scripts.train_wl --data_dir ./data/mnist --task domain_generalization --algorithm {algo} --test_env 1 2 3 4 5 6 7 8 9 --dataset {dataset} --hparams='{hparams}' --seed {args.seed} --device {args.device} --output_dir cmnist_output_ushift --steps 5000 --n_iter 3""")