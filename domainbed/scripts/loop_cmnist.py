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
    "SUBG":"""{"batch_size": 256, 
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
    "GroupDRO_WL":"""{"batch_size": 256, 
                    "arch": "resnet18",
                    "class_balanced": false, 
                    "data_augmentation": true, 
                    "lr": 0.00016629177873519647, 
                    "nonlinear_classifier": false, 
                    "resnet_dropout": 0.1,
                    "groupdro_eta": 0.15479266228690425,
                    "weight_decay": 1.1975155295174919e-06}""",
    "CORAL_WL":"""{"batch_size": 256, 
                    "arch": "resnet18",
                    "class_balanced": false, 
                    "data_augmentation": true, 
                    "lr": 6.238845035844661e-05, 
                    "nonlinear_classifier": false, 
                    "resnet_dropout": 0.1,
                    "mmd_gamma": 23.99545530150662,
                    "weight_decay": 0.0038261244966893123}""",
    "UShift2":"""{"batch_size": 256, 
                    "arch": "resnet18",
                    "class_balanced": false, 
                    "data_augmentation": true, 
                    "lr": 5.6841898471378446e-06, 
                    "nonlinear_classifier": false, 
                    "resnet_dropout": 0.1,
                    "weight_decay": 0.005100223533962902}""",
}

for algo in ["UShift2"]: #"GroupDRO_WL","SUBG","CORAL_WL","GroupDRO_WL","CORAL_WL","UShift2","UShift"
    for dataset in ["ColoredMNIST_WL"]:
        hparams = hparams_dict[algo]
        print(f"Train {algo} on {dataset}")
        os.system(f"""python3 -m domainbed.scripts.train_n --data_dir ./data/mnist --task domain_generalization --algorithm {algo} --test_env 1 2 3 4 5 6 7 8 9 --dataset {dataset} --hparams='{hparams}' --seed {args.seed} --output_dir cmnist_output_ushift --n_iter 3""")