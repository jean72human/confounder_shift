import os
import time
import argparse

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Domain generalization')

parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--seed', type=int, default=1)

args = parser.parse_args()

data_dir = "./data/datadir"
batch_size = 128

hparams_dict = {
    "O2O_easy": {
        "ERM": """{"batch_size": batchsize, 
                "class_balanced": false, 
                "data_augmentation": true, 
                "lr": 0.00016629177873519647, 
                "nonlinear_classifier": false, 
                "arch": "archused", 
                "resnet_dropout": 0.1,
                "weight_decay": 1.1975155295174919e-06}""",
        "MMD": """{"batch_size": batchsize, 
                "class_balanced": false, 
                "data_augmentation": true, 
                "lr": 0.00016629177873519647, 
                "nonlinear_classifier": false, 
                "arch": "archused", 
                "resnet_dropout": 0.1,
                "weight_decay": 1.1975155295174919e-06,
                "mmd_gamma": 5.636749849666358} """,
    },
    "M2M_hard": {
        "ERM": """{"batch_size": batchsize, 
                "class_balanced": false, 
                "data_augmentation": true, 
                "lr": 0.0001653813153854724, 
                "nonlinear_classifier": false, 
                "arch": "archused", 
                "resnet_dropout": 0.1,
                "weight_decay": 2.7643974709171963e-05}""",
        "MMD": """{"batch_size": batchsize, 
                "class_balanced": false, 
                "data_augmentation": true, 
                "lr": 0.0001653813153854724, 
                "nonlinear_classifier": false, 
                "arch": "archused", 
                "resnet_dropout": 0.1,
                "weight_decay": 2.7643974709171963e-05,
                "mmd_gamma": 1.0215072228839979} """,
    }
}

hparams_dict["O2O_medium"] = hparams_dict["O2O_easy"]
hparams_dict["O2O_hard"] = hparams_dict["O2O_easy"]

hparams_dict["M2M_easy"] = hparams_dict["M2M_hard"]
hparams_dict["M2M_medium"] = hparams_dict["M2M_hard"]

for arch in ["resnet50"]:
    for algo in ["MMD"]: 
        for dataset in ["O2O_easy","O2O_medium","O2O_hard","M2M_hard","M2M_easy","M2M_medium"]:
            hparams = hparams_dict[dataset][algo].replace("batchsize", str(batch_size)).replace("archused", arch)
            hparams = hparams.replace("\n", "").replace(" ", "")
            print(f"Train {algo} on {dataset}")
            os.system(f"""python3 -m domainbed.scripts.train_n --data_dir={data_dir}  --algorithm {algo} --test_env 0 --dataset {dataset} --hparams='{hparams}' --seed {args.seed} --output_dir new_resnet50_final_output --n_iter 3""")