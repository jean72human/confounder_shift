import os
import time
import argparse

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Domain generalization')

parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--seed', type=int, default=1)

args = parser.parse_args()

hparams_dict = {
    "ERM_WL":"""{"batch_size": 256, 
                    "arch": "resnet18",
                    "class_balanced": false, 
                    "data_augmentation": true, 
                    "lr": 5e-5, 
                    "nonlinear_classifier": false, 
                    "resnet_dropout": 0.0,
                    "weight_decay": 0.0}""",
    "UShift":"""{"batch_size": 256, 
                    "arch": "resnet18",
                    "class_balanced": false, 
                    "data_augmentation": true, 
                    "lr": 5e-5, 
                    "nonlinear_classifier": false, 
                    "resnet_dropout": 0.0,
                    "mmd_gamma": 1e2,
                    "weight_decay": 0.0}""",
    "UShift2":"""{"batch_size": 256, 
                    "arch": "resnet18",
                    "class_balanced": false, 
                    "data_augmentation": true, 
                    "lr": 5e-5, 
                    "nonlinear_classifier": false, 
                    "resnet_dropout": 0.0,
                    "weight_decay": 0.0}""",
    "GroupDRO_WL":"""{"batch_size": 256, 
                    "arch": "resnet18",
                    "class_balanced": false, 
                    "data_augmentation": true, 
                    "lr": 5e-05, 
                    "nonlinear_classifier": false, 
                    "resnet_dropout": 0.1,
                    "groupdro_eta": 1e2,
                    "weight_decay": 0.0}""",
    "CORAL_WL":"""{"batch_size": 256, 
                    "arch": "resnet18",
                    "class_balanced": false, 
                    "data_augmentation": true, 
                    "lr": 5e-05, 
                    "nonlinear_classifier": false, 
                    "resnet_dropout": 0.0,
                    "mmd_gamma": 1e2,
                    "weight_decay": 0.0}""",
}

for algo in ["ERM_WL","CORAL_WL","GroupDRO_WL","UShift2","UShift"]: #"GroupDRO_WL","SUBG","CORAL_WL","GroupDRO_WL","CORAL_WL","UShift2","UShift"
    for dataset in ["ColoredMNIST_WL"]:
        hparams = hparams_dict[algo]
        print(f"Train {algo} on {dataset}")
        os.system(f"""python3 -m domainbed.scripts.train_n --data_dir ./data/mnist --task domain_generalization --algorithm {algo} --test_env 1 2 3 4 5 6 7 8 9 --dataset {dataset} --hparams='{hparams}' --seed {args.seed} --output_dir cmnist_output --n_iter 3""")