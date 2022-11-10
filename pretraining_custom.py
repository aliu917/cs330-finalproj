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
import wandb
import tqdm

import utils
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 100

# wandb.init(project="cs330-finalproj", entity="aliu917")
# wandb.config = {
#   "learning_rate": 0.001,
#   "epochs": 100,
#   "batch_size": 128
# }
# sweep_config = {
#    'method': 'grid',
#    'parameters': {
#        'learning_rate': {
#            'values': []
#        }
#    }
# }


def get_model():
    return ResNet.resnet26()

def train():
    train_loader, val_loader = utils.get_train_data()
    model = get_model()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    cifar_train_loss = []
    cifar_val_loss = []
    cifar_train_accuracy = []
    cifar_val_accuracy = []

    model.train()
    pbar = tqdm.tqdm(range(epochs))
    for epoch in range(pbar):
        training_losses = []
        num_correct = 0

        for i, (images, labels) in tqdm.tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            wandb.log({"loss": loss})
            loss.backward()
            optimizer.step()

            training_losses.append(loss.item())
            predictions = torch.argmax(pred, dim=1)
            num_correct += torch.sum(predictions == y).item()

            # if (i + 1) % 100 == 0:
            #     print("Epoch [{}/{}], Loss: {:.4f}"
            #           .format(epoch + 1, epochs, loss.item()))

        print("Finished Epoch", epoch + 1, ", training loss:", np.mean(training_losses))
        cifar_train_loss.append(np.mean(training_losses))
        cifar_train_accuracy.append(num_correct / len(training_losses))

        with torch.no_grad():
            model.eval()  # Put model in eval mode
            num_correct = 0
            val_losses = []
            for x, y in tqdm.tqdm(val_loader):
                x, y = x.float().to(device), y.float().to(device)
                pred = model(x)
                val_losses.append(loss.item())
                predictions = torch.argmax(pred, dim=1)
                num_correct += torch.sum(predictions == y).item()
            print("Epoch", epoch + 1, " Accuracy:", num_correct / len(val_losses))
            cifar_val_accuracy.append(num_correct / len(val_losses))
            cifar_val_loss.append(np.mean(val_losses))
            model.train()  # Put model back in train mode

        # Decay learning rate
        # if (epoch + 1) % 20 == 0:
        #     curr_lr /= 3
        #     update_lr(optimizer, curr_lr)

if __name__ == '__main__':
    train()
