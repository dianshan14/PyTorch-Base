import os 
import numpy as np

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from utils.dataset import *

def get_transforms(CONFIG):
    train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    val_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    return train_transform, val_transform, test_transform




def get_dataset(train_transform, val_transform, test_transform, CONFIG):
    if CONFIG.dataset == "cifar10":
        train_dataset, val_dataset, test_dataset = get_cifar10(train_transform, val_transform, test_transform, CONFIG)

    elif CONFIG.dataset == "cifar100":
        train_dataset, val_dataset, test_dataset = get_cifar100(train_transform, val_transform, test_transform, CONFIG)

    elif CONFIG.datasets == "imagenet_lmdb":
        train_dataset, val_dataset, test_dataset = get_imagenet_lmdb(train_transform, val_transform, test_transform, CONFIG)

    elif CONFIG.datasets == "imagenet":
        train_dataset, val_dataset, test_dataset = get_imagenet(train_transform, val_transform, test_transform, CONFIG)

    else:
       raise

    return train_dataset, val_dataset, test_dataset


def get_dataloader(train_dataset, val_dataset, test_dataset, CONFIG):
    def _build_loader(dataset, shuffle, sampler=None):
        return torch.utils.data.DataLoader(
                    dataset,
                    batch_size=CONFIG.batch_size,
                    pin_memory=True,
                    num_workers=CONFIG.num_workers,
                    sampler=sampler
                )

    train_loader = _build_loader(train_dataset, True)
    val_loader = _build_loader(val_dataset, True)
    test_loader = _build_loader(test_dataset, True)

    return train_loader, val_loader, test_loader


