import os
import time
import argparse

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Domain generalization')

parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--seed', type=int, default=1)

args = parser.parse_args()

data_dir = "/home/aengusl/Desktop/Projects/OOD_workshop/Stable_Diffusion_Generation/gen_images/full_folder"
batch_size = 32

hparams_dict = {
    "SpuriousLocationType1_1": {
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
                "mmd_gamma": 7.289784897124338} """,
    },
    "SpuriousLocationType2_1": {
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

hparams_dict["SpuriousLocationType1_2"] = hparams_dict["SpuriousLocationType1_1"]
hparams_dict["SpuriousLocationType1_3"] = hparams_dict["SpuriousLocationType1_1"]

hparams_dict["SpuriousLocationType2_2"] = hparams_dict["SpuriousLocationType2_1"]
hparams_dict["SpuriousLocationType2_3"] = hparams_dict["SpuriousLocationType2_1"]

for arch in ["dino","convnext","vit-b"]:
    for algo in ["ERM","MMD"]: 
        for dataset in ["SpuriousLocationType1_1","SpuriousLocationType1_2","SpuriousLocationType1_3","SpuriousLocationType2_1","SpuriousLocationType2_2","SpuriousLocationType2_3"]:
            hparams = hparams_dict[dataset][algo].replace("batchsize", str(batch_size)).replace("archused", arch)
            hparams = hparams.replace("\n", "").replace(" ", "")
            print(f"Train {algo} on {dataset}")
            os.system(f"""python3 -m domainbed.scripts.train_n --data_dir={data_dir}  --algorithm {algo} --test_env 0 --dataset {dataset} --hparams='{hparams}' --seed {args.seed} --output_dir final_output --n_iter 3""")