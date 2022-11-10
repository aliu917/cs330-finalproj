from typing import Optional, Tuple

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils import data
from torchvision.datasets import CIFAR10
from torchvision.models import ResNet
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split, Dataset
from models.ResNet import ResNetCifar as ResNet
import torchvision.transforms as transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_model(group_norm=8):
    def gn_helper(planes):
        return nn.GroupNorm(group_norm, planes)
    model = ResNet(26, 1, channels=3, classes=10, norm_layer=gn_helper).to(device)
    if device == 'cuda':
        ckpt = torch.load('models/ckpt.pth')
    else:
        ckpt = torch.load('models/ckpt.pth', map_location=torch.device('cpu'))
    model.load_state_dict(ckpt['net'])
    return model


# def get_train_data():
#     dataset = CIFAR10(root='data',
#                             train=True,
#                             download=True)
#     val_num = int(.2 * len(dataset))
#     train_num = int(.8 * len(dataset))
#     training, validation = torch.utils.data.random_split(dataset, [train_num, val_num],
#                                                          generator=torch.Generator().manual_seed(42))
#     train_dataloader = torch.utils.data.DataLoader(dataset=training,
#                                            batch_size=128,
#                                            shuffle=True)
#     val_dataloader = torch.utils.data.DataLoader(dataset=validation,
#                                           batch_size=128,
#                                           shuffle=True)
#     return train_dataloader, val_dataloader

def get_data(dataset="cifar10"):
    NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    te_transforms = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(*NORM)])
    dataset = CIFAR10(root='data',
                            train=False,
                            download=True, transform=te_transforms)
    val_num = int(.2 * len(dataset))
    train_num = int(.8 * len(dataset))

    training, validation = torch.utils.data.random_split(dataset, [train_num, val_num],
                                                         generator=torch.Generator().manual_seed(42))
    train_loader = data.DataLoader(training,
                                  batch_size=128,
                                  shuffle=True,
                                  num_workers=0)
    val_loader = data.DataLoader(validation,
                                  batch_size=128,
                                  shuffle=False,
                                  num_workers=0)
    return train_loader, val_loader

def get_test_data_old(dataset, n_examples: Optional[int] = None):
    # dataset = load_dataset("cifar10", split="train")
    # test_dataset = load_dataset("cifar10", split="test")
    dataset = CIFAR10(root='data',
                            train=False,
                            download=True)
    return _load_dataset(dataset, n_examples)

# https://github.com/RobustBench/robustbench/blob/513d60c80965e601ba994f0b78b5d8a9dabe85e3/robustbench/data.py#L77
def _load_dataset(
        dataset: Dataset,
        n_examples: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = 100
    test_loader = data.DataLoader(dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=0)

    x_test, y_test = [], []
    for i, (x, y) in enumerate(test_loader):
        x_test.append(x)
        y_test.append(y)
        if n_examples is not None and batch_size * i >= n_examples:
            break
    x_test_tensor = torch.cat(x_test)
    y_test_tensor = torch.cat(y_test)

    if n_examples is not None:
        x_test_tensor = x_test_tensor[:n_examples]
        y_test_tensor = y_test_tensor[:n_examples]

    return x_test_tensor, y_test_tensor