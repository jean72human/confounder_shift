import os
import time
import argparse

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Domain generalization')

parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--seed', type=int, default=1)

args = parser.parse_args()

data_dir = "/home/aengusl/Desktop/Projects/OOD_workshop/Stable_Diffusion_Generation/gen_images/datadir"

hparams_dict = {
    "SpuriousLocationType1_1": {
        "ERM": """{"batch_size": 128, 
                "class_balanced": false, 
                "data_augmentation": true, 
                "lr": 0.00016629177873519647, 
                "nonlinear_classifier": false, 
                "arch": "resnet18", 
                "resnet_dropout": 0.1,
                "weight_decay": 1.1975155295174919e-06}""",
        "GroupDRO": """{"batch_size": 128, 
                "class_balanced": false, 
                "data_augmentation": true, 
                "lr": 0.00016629177873519647, 
                "nonlinear_classifier": false, 
                "arch": "resnet18", 
                "resnet_dropout": 0.1,
                "weight_decay": 1.1975155295174919e-06,
                "groupdro_eta": 0.0053050580120662895} """,
        "IRM": """{"batch_size": 128, 
                "class_balanced": false, 
                "data_augmentation": true, 
                "lr": 0.00016629177873519647, 
                "nonlinear_classifier": false, 
                "arch": "resnet18", 
                "resnet_dropout": 0.1,
                "weight_decay": 1.1975155295174919e-06,
                "irm_lambda": 1.8838285530562104,
                "irm_penalty_anneal_iters": 247} """,
        "MMD": """{"batch_size": 128, 
                "class_balanced": false, 
                "data_augmentation": true, 
                "lr": 0.00016629177873519647, 
                "nonlinear_classifier": false, 
                "arch": "resnet18", 
                "resnet_dropout": 0.1,
                "weight_decay": 1.1975155295174919e-06,
                "mmd_gamma": 7.289784897124338} """,
        "CORAL": """{"batch_size": 128, 
                "class_balanced": false, 
                "data_augmentation": true, 
                "lr": 0.00016629177873519647, 
                "nonlinear_classifier": false, 
                "arch": "resnet18", 
                "resnet_dropout": 0.1,
                "weight_decay": 1.1975155295174919e-06,
                "mmd_gamma": 6.9018246989615895} """,
        "CausIRL_CORAL": """{"batch_size": 128, 
                "class_balanced": false, 
                "data_augmentation": true, 
                "lr": 0.00016629177873519647, 
                "nonlinear_classifier": false, 
                "arch": "resnet18", 
                "resnet_dropout": 0.1,
                "weight_decay": 1.1975155295174919e-06,
                "mmd_gamma": 3.5146823420446407} """,
    },
    "SpuriousLocationType2_1": {
        "ERM": """{"batch_size": 128, 
                "class_balanced": false, 
                "data_augmentation": true, 
                "lr": 0.0001653813153854724, 
                "nonlinear_classifier": false, 
                "arch": "resnet18", 
                "resnet_dropout": 0.1,
                "weight_decay": 2.7643974709171963e-05}""",
        "GroupDRO": """{"batch_size": 128, 
                "class_balanced": false, 
                "data_augmentation": true, 
                "lr": 0.0001653813153854724, 
                "nonlinear_classifier": false, 
                "arch": "resnet18", 
                "resnet_dropout": 0.1,
                "weight_decay": 2.7643974709171963e-05,
                "groupdro_eta": 0.013378423587817576} """,
        "IRM": """{"batch_size": 128, 
                "class_balanced": false, 
                "data_augmentation": true, 
                "lr": 0.0001653813153854724, 
                "nonlinear_classifier": false, 
                "arch": "resnet18", 
                "resnet_dropout": 0.1,
                "weight_decay": 2.7643974709171963e-05,
                "irm_lambda": 29.3676220201571,
                "irm_penalty_anneal_iters": 3001} """,
        "MMD": """{"batch_size": 128, 
                "class_balanced": false, 
                "data_augmentation": true, 
                "lr": 0.0001653813153854724, 
                "nonlinear_classifier": false, 
                "arch": "resnet18", 
                "resnet_dropout": 0.1,
                "weight_decay": 2.7643974709171963e-05,
                "mmd_gamma": 1.0215072228839979} """,
        "CORAL": """{"batch_size": 128, 
                "class_balanced": false, 
                "data_augmentation": true, 
                "lr": 0.0001653813153854724, 
                "nonlinear_classifier": false, 
                "arch": "resnet18", 
                "resnet_dropout": 0.1,
                "weight_decay": 2.7643974709171963e-05,
                "mmd_gamma": 0.5870292457165399} """,
        "CausIRL_CORAL": """{"batch_size": 128, 
                "class_balanced": false, 
                "data_augmentation": true, 
                "lr": 0.0001653813153854724, 
                "nonlinear_classifier": false, 
                "arch": "resnet18", 
                "resnet_dropout": 0.1,
                "weight_decay": 2.7643974709171963e-05,
                "mmd_gamma": 0.5870292457165399} """,
    }
}

hparams_dict["SpuriousLocationType1_2"] = hparams_dict["SpuriousLocationType1_1"]
hparams_dict["SpuriousLocationType1_3"] = hparams_dict["SpuriousLocationType1_1"]

hparams_dict["SpuriousLocationType2_2"] = hparams_dict["SpuriousLocationType2_1"]
hparams_dict["SpuriousLocationType2_3"] = hparams_dict["SpuriousLocationType2_1"]

for algo in ["ERM","GroupDRO","IRM","CORAL","CausIRL_CORAL","MMD"]: #
    for dataset in ["SpuriousLocationType1_1","SpuriousLocationType1_2","SpuriousLocationType1_3"]:
        hparams = hparams_dict[dataset][algo].replace("\n", "").replace(" ", "")
        print(f"Train {algo} on {dataset}")
        os.system(f"""python3 -m domainbed.scripts.train_n --data_dir={data_dir}  --algorithm {algo} --test_env 0 --dataset {dataset} --hparams='{hparams}' --seed {args.seed} --output_dir final_output --n_iter 3""")