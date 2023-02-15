# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Subset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate

from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from wilds.datasets.fmow_dataset import FMoWDataset

from torch.utils.data import ConcatDataset, DataLoader

import re

import random

def find_match(string):
    match = None
    if 'image' in string:
        match = re.search(r'\d{1,3}', string.split('image')[-1])
    if match:
        return int(match.group())
    else:
        return 0

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATASETS = [
    # Debug
    "Debug28",
    "Debug224",
    # Small images
    "ColoredMNIST",
    "RotatedMNIST",
    # Big images
    "VLCS",
    "PACS",
    "OfficeHome",
    "TerraIncognita",
    "DomainNet",
    "SVIRO",
    # WILDS datasets
    "WILDSCamelyon",
    "WILDSFMoW",
    # our datasets,
    "SpuriousLocation1",
    "LocationShift1",
    "Hybrid1",
]

def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 8            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class Debug(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []
        for _ in [0, 1, 2]:
            self.datasets.append(
                TensorDataset(
                    torch.randn(16, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (16,))
                )
            )

class Debug28(Debug):
    INPUT_SHAPE = (3, 28, 28)
    ENVIRONMENTS = ['0', '1', '2']

class Debug224(Debug):
    INPUT_SHAPE = (3, 224, 224)
    ENVIRONMENTS = ['0', '1', '2']


class MultipleEnvironmentMNIST(MultipleDomainDataset):
    def __init__(self, root, environments, dataset_transform, input_shape,
                 num_classes):
        super().__init__()
        if root is None:
            raise ValueError('Data directory not specified!')

        original_dataset_tr = MNIST(root, train=True, download=True)
        original_dataset_te = MNIST(root, train=False, download=True)

        original_images = torch.cat((original_dataset_tr.data,
                                     original_dataset_te.data))

        original_labels = torch.cat((original_dataset_tr.targets,
                                     original_dataset_te.targets))

        shuffle = torch.randperm(len(original_images))

        original_images = original_images[shuffle]
        original_labels = original_labels[shuffle]

        self.datasets = []

        for i in range(len(environments)):
            images = original_images[i::len(environments)]
            labels = original_labels[i::len(environments)]
            self.datasets.append(dataset_transform(images, labels, environments[i]))

        self.input_shape = input_shape
        self.num_classes = num_classes


class ColoredMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['+90%', '+80%', '-90%']

    def __init__(self, root, test_envs, hparams):
        super(ColoredMNIST, self).__init__(root, [0.1, 0.2, 0.9],
                                         self.color_dataset, (2, 28, 28,), 2)

        self.input_shape = (2, 28, 28,)
        self.num_classes = 2

    def color_dataset(self, images, labels, environment):
        # # Subsample 2x for computational convenience
        # images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit
        labels = (labels < 5).float()
        # Flip label with probability 0.25
        labels = self.torch_xor_(labels,
                                 self.torch_bernoulli_(0.25, len(labels)))

        # Assign a color based on the label; flip the color with probability e
        colors = self.torch_xor_(labels,
                                 self.torch_bernoulli_(environment,
                                                       len(labels)))
        images = torch.stack([images, images], dim=1)
        # Apply the color to the image by zeroing out the other color channel
        images[torch.tensor(range(len(images))), (
            1 - colors).long(), :, :] *= 0

        x = images.float().div_(255.0)
        y = labels.view(-1).long()

        return TensorDataset(x, y)

    def torch_bernoulli_(self, p, size):
        return (torch.rand(size) < p).float()

    def torch_xor_(self, a, b):
        return (a - b).abs()


class RotatedMNIST(MultipleEnvironmentMNIST):
    ENVIRONMENTS = ['0', '15', '30', '45', '60', '75']

    def __init__(self, root, test_envs, hparams):
        super(RotatedMNIST, self).__init__(root, [0, 15, 30, 45, 60, 75],
                                           self.rotate_dataset, (1, 28, 28,), 10)

    def rotate_dataset(self, images, labels, angle):
        rotation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Lambda(lambda x: rotate(x, angle, fill=(0,),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR)),
            transforms.ToTensor()])

        x = torch.zeros(len(images), 1, 28, 28)
        for i in range(len(images)):
            x[i] = rotation(images[i])

        y = labels.view(-1)

        return TensorDataset(x, y)


class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, augment, hparams):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []
        for i, environment in enumerate(environments):

            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path,
                transform=env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = len(self.datasets[-1].classes)

class VLCS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["C", "L", "S", "V"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "VLCS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "S"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "PACS/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class DomainNet(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 1000
    ENVIRONMENTS = ["clip", "info", "paint", "quick", "real", "sketch"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "domain_net/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class OfficeHome(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "R"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "office_home/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class TerraIncognita(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["L100", "L38", "L43", "L46"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "terra_incognita/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)

class SVIRO(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["aclass", "escape", "hilux", "i3", "lexus", "tesla", "tiguan", "tucson", "x5", "zoe"]
    def __init__(self, root, test_envs, hparams):
        self.dir = os.path.join(root, "sviro/")
        super().__init__(self.dir, test_envs, hparams['data_augmentation'], hparams)


class WILDSEnvironment:
    def __init__(
            self,
            wilds_dataset,
            metadata_name,
            metadata_value,
            transform=None):
        self.name = metadata_name + "_" + str(metadata_value)

        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_array = wilds_dataset.metadata_array
        subset_indices = torch.where(
            metadata_array[:, metadata_index] == metadata_value)[0]

        self.dataset = wilds_dataset
        self.indices = subset_indices
        self.transform = transform

    def __getitem__(self, i):
        x = self.dataset.get_input(self.indices[i])
        if type(x).__name__ != "Image":
            x = Image.fromarray(x)

        y = self.dataset.y_array[self.indices[i]]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.indices)


class WILDSDataset(MultipleDomainDataset):
    INPUT_SHAPE = (3, 224, 224)
    def __init__(self, dataset, metadata_name, test_envs, augment, hparams):
        super().__init__()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        augment_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.datasets = []

        for i, metadata_value in enumerate(
                self.metadata_values(dataset, metadata_name)):
            if augment and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            env_dataset = WILDSEnvironment(
                dataset, metadata_name, metadata_value, env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, 224, 224,)
        self.num_classes = dataset.n_classes

    def metadata_values(self, wilds_dataset, metadata_name):
        metadata_index = wilds_dataset.metadata_fields.index(metadata_name)
        metadata_vals = wilds_dataset.metadata_array[:, metadata_index]
        return sorted(list(set(metadata_vals.view(-1).tolist())))


class WILDSCamelyon(WILDSDataset):
    ENVIRONMENTS = [ "hospital_0", "hospital_1", "hospital_2", "hospital_3",
            "hospital_4"]
    def __init__(self, root, test_envs, hparams):
        dataset = Camelyon17Dataset(root_dir=root)
        super().__init__(
            dataset, "hospital", test_envs, hparams['data_augmentation'], hparams)


class WILDSFMoW(WILDSDataset):
    ENVIRONMENTS = [ "region_0", "region_1", "region_2", "region_3",
            "region_4", "region_5"]
    def __init__(self, root, test_envs, hparams):
        dataset = FMoWDataset(root_dir=root)
        super().__init__(
            dataset, "region", test_envs, hparams['data_augmentation'], hparams)


class OODBenchmark(MultipleDomainDataset):
    def __init__(self, train_combinations, test_combinations, root_dir, augment=True, use_vit=False):
        self.input_shape = (3,224,224)
        self.num_classes = 4
        self.N_STEPS = 1000

        dataset_lower_bound=0
        dataset_upper_bound=100

        train_data_list = []
        val_data_list = []
        test_data_list = []

        #self.class_list = ["bald","bulldog","dachshund","goose","hamster","house","labrador","owl","stallion","welsh"]
        self.class_list = ["bulldog","dachshund","labrador","welsh"]

        test_transforms_list = [
            transforms.transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]

        train_transforms_list = [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        # Build test and validation transforms
        test_transforms = transforms.transforms.Compose(test_transforms_list)

        # Build training data transforms
        if augment:
            train_transforms = transforms.transforms.Compose(train_transforms_list)
        else:
            train_transforms = test_transforms

        if isinstance(train_combinations, dict):
            for_each_class_group = []
            cg_index = 0
            for classes,comb_list in train_combinations.items():
                for_each_class_group.append([])
                for (location,limit) in comb_list:

                    path = os.path.join(root_dir, f"{location}/")
                    data = ImageFolder(
                        root=path, transform=train_transforms, is_valid_file=lambda x: find_match(x)>dataset_lower_bound and find_match(x)>0
                    )
                    #print([data.class_to_idx[c] for c in self.class_list])

                    classes_idx = [data.class_to_idx[c] for c in classes]
                    to_keep_idx = []
                    for class_to_limit in classes_idx:
                        count_limit=0
                        for i in range(len(data)):
                            if data[i][1] == class_to_limit:
                                to_keep_idx.append(i)
                                count_limit+=1
                            if count_limit>=limit:
                                break

                    subset = Subset(data, to_keep_idx)

                    for_each_class_group[cg_index].append(subset) 
                cg_index+=1
            for group in range(len(for_each_class_group[0])):
                train_data_list.append(ConcatDataset([
                    for_each_class_group[k][group] for k in range(len(for_each_class_group))
                ]))
        else:
            for location in train_combinations:

                path = os.path.join(root_dir, f"{location}/")
                data = ImageFolder(
                    root=path, transform=train_transforms, is_valid_file=lambda x: find_match(x)>dataset_lower_bound and find_match(x)>0
                )

                train_data_list.append(data) 

        # Make test_data_list
        if isinstance(test_combinations, dict):
            for_each_class_group = []
            cg_index = 0
            for classes,comb_list in test_combinations.items():
                for_each_class_group.append([])
                for location in comb_list:

                    path = os.path.join(root_dir, f"{location}/")
                    data = ImageFolder(
                        root=path, transform=test_transforms, is_valid_file=lambda x: find_match(x)<dataset_upper_bound and find_match(x)>0
                    )

                    classes_idx = [data.class_to_idx[c] for c in classes]
                    to_keep_idx = [i for i in range(len(data)) if data.imgs[i][1] in classes_idx]

                    subset = Subset(data, to_keep_idx)

                    for_each_class_group[cg_index].append(subset) 
                cg_index+=1
            for group in range(len(for_each_class_group[0])):
                test_data_list.append(ConcatDataset([
                    for_each_class_group[k][group] for k in range(len(for_each_class_group))
                ]))
        else:
            for location in test_combinations:

                path = os.path.join(root_dir, f"{location}/")
                data = ImageFolder(root=path, transform=test_transforms, is_valid_file=lambda x: find_match(x)<dataset_upper_bound and find_match(x)>0)

                test_data_list.append(data) 

        # Concatenate test datasets 
        test_data = ConcatDataset(test_data_list)

        self.datasets = [test_data] + train_data_list
        for k,dataset in enumerate(self.datasets[1:]): 
            print(f"Group {k+1}: {len(dataset)} images")


class LocationShift1(OODBenchmark):
    ENVIRONMENTS = ["Beach","Dirt"]
    def __init__(self, root_dir, test_envs, hparams):
        exp1_TI = {}
        exp1_TI['train_combinations'] = ["dirt"]
        exp1_TI['test_combinations'] = ["snow"]
        super().__init__(exp1_TI['train_combinations'], exp1_TI['test_combinations'], root_dir, hparams["data_augmentation"], not hparams["resnet18"])



# class SpuriousLocation1(OODBenchmark):
#     ENVIRONMENTS = ["Jungle","SC_group_1","SC_group_2"]
#     def __init__(self, root_dir, test_envs, hparams):
#         exp1_TI = {}
#         exp1_TI['train_combinations'] = {
#             ## correlated class
#             ("bulldog",):[("desert",490),("desert",420)],
#             ("dachsund",):[("jungle",490),("jungle",420)],
#             ("labrador","welsh"):[("mountain",490),("mountain",420)],
#             ## grass
#             ("bulldog","dachsund","labrador","welsh"):[("grass",10),("grass",80)],
#         }
#         exp1_TI['test_combinations'] = {
#             ("bulldog",):["mountain"],
#             ("dachsund",):["desert"],
#             ("labrador","welsh"):["jungle"],
#         }
#         super().__init__(exp1_TI['train_combinations'], exp1_TI['test_combinations'], root_dir, hparams["data_augmentation"], not hparams["resnet18"])

class SpuriousLocation1(OODBenchmark):
    ENVIRONMENTS = ["Jungle","SC_group_1","SC_group_2"]
    def __init__(self, root_dir, test_envs, hparams):
        counts = [290,260]
        total = 300
        exp1_TI = {}
        exp1_TI['train_combinations'] = {
            ## correlated class
            ("bulldog",):[("desert",counts[0]),("desert",counts[1])],
            ("dachshund",):[("jungle",counts[0]),("jungle",counts[1])],
            ("labrador",):[("dirt",counts[0]),("dirt",counts[1])],
            ("welsh",):[("snow",counts[0]),("snow",counts[1])],
            ## grass
            ("bulldog","dachshund","labrador","welsh"):[("beach",total-counts[0]),("beach",total-counts[1])],
        }
        exp1_TI['test_combinations'] = {
            ("bulldog",):["jungle"],
            ("dachshund",):["dirt"],
            ("labrador",):["snow"],
            ("welsh",):["desert"],
        }
        super().__init__(exp1_TI['train_combinations'], exp1_TI['test_combinations'], root_dir, hparams["data_augmentation"], not hparams["resnet18"])

class SpuriousLocation2(OODBenchmark):
    ENVIRONMENTS = ["Jungle","SC_group_1","SC_group_2"]
    def __init__(self, root_dir, test_envs, hparams):
        counts = [280,250]
        total = 300
        exp1_TI = {}
        exp1_TI['train_combinations'] = {
            ## correlated class
            ("bulldog",):[("dirt",counts[0]),("dirt",counts[1])],
            ("dachshund",):[("desert",counts[0]),("desert",counts[1])],
            ("labrador",):[("snow",counts[0]),("snow",counts[1])],
            ("welsh",):[("beach",counts[0]),("beach",counts[1])],
            ## grass
            ("bulldog","dachshund","labrador","welsh"):[("jungle",total-counts[0]),("jungle",total-counts[1])],
        }
        exp1_TI['test_combinations'] = {
            ("bulldog",):["desert"],
            ("dachshund",):["snow"],
            ("labrador",):["beach"],
            ("welsh",):["dirt"],
        }
        super().__init__(exp1_TI['train_combinations'], exp1_TI['test_combinations'], root_dir, hparams["data_augmentation"], not hparams["resnet18"])


class UnseenLocation1(OODBenchmark):
    ENVIRONMENTS = ["Jungle","SC_group_1","SC_group_2"]
    def __init__(self, root_dir, test_envs, hparams):
        counts = [280,250]
        total = 300
        exp1_TI = {}
        exp1_TI['train_combinations'] = {
            ## correlated class
            ("bulldog",):[("desert",counts[0]),("desert",counts[1])],
            ("dachshund",):[("jungle",counts[0]),("jungle",counts[1])],
            ("labrador",):[("dirt",counts[0]),("dirt",counts[1])],
            ("welsh",):[("snow",counts[0]),("snow",counts[1])],
            ## grass
            ("bulldog","dachshund","labrador","welsh"):[("beach",total-counts[0]),("beach",total-counts[1])],
        }
        exp1_TI['test_combinations'] = ['mountain']
        super().__init__(exp1_TI['train_combinations'], exp1_TI['test_combinations'], root_dir, hparams["data_augmentation"], not hparams["resnet18"])


#class Hybrid1(OODBenchmark):
#    ENVIRONMENTS = ["Desert","SC_group_1","SC_group_2"]
#    def __init__(self, root_dir, test_envs, hparams):
#        exp1_TI = {}
#        exp1_TI['train_combinations'] = {
#            ## correlated class
#            ("bulldog","dachsund","welsh","house","labrador"):[("desert",480),("desert",420)],
#            ("bald","goose","owl","stallion","hamster"):[("mountain",480),("mountain",420)],
#            ## mountain
#            ("bulldog","dachsund","hamster","house","labrador","welsh","bald","goose","owl","stallion"):[("jungle",20),("jungle",80)],
#        }
#        exp1_TI['test_combinations'] = ["grass"]
#        super().__init__(exp1_TI['train_combinations'], exp1_TI['test_combinations'], root_dir, hparams["data_augmentation"], not hparams["resnet18"])



