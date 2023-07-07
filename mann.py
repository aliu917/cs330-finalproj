import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

def initialize_weights(model):
    if type(model) in [nn.Linear]:
        nn.init.xavier_uniform_(model.weight)
        nn.init.zeros_(model.bias)
    elif type(model) in [nn.LSTM, nn.RNN, nn.GRU]:
        nn.init.orthogonal_(model.weight_hh_l0)
        nn.init.xavier_uniform_(model.weight_ih_l0)
        nn.init.zeros_(model.bias_hh_l0)
        nn.init.zeros_(model.bias_ih_l0)


class MANN(nn.Module):
    def __init__(self, input_dim, num_classes, samples_per_class, hidden_dim):
        super(MANN, self).__init__()
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class

        self.layer1 = nn.LSTM(num_classes + input_dim, hidden_dim, batch_first=True)
        self.layer2 = nn.LSTM(hidden_dim, num_classes, batch_first=True)
        initialize_weights(self.layer1)
        initialize_weights(self.layer2)

    def forward(self, input_images, input_labels):
        """
        MANN
        Args:
            input_images: [B, K+1, N, 784] flattened images
            labels: [B, K+1, N, N] ground truth labels
        Returns:
            [B, K+1, N, N] predictions
        """
        train_split = torch.cat((input_images[:, :-1, :, :], input_labels[:, :-1, :, :]), dim=-1)
        b, k, n, dim = train_split.shape
        test_split = torch.cat((input_images[:, -1, :, :], torch.zeros(input_labels[:, -1, :, :].shape)), dim=-1)
        train = train_split.reshape((b, k*n, dim))
        test = test_split.reshape((b, n, dim))
        input = torch.cat((train, test), dim=1)
        out1, _ = self.layer1(torch.tensor(input).float())
        out2, _ = self.layer2(out1)
        return out2.reshape(b, k+1, n, input_labels.shape[-1])


    def loss_function(self, preds, labels):
        """
        Computes MANN loss
        Args:
            preds: [B, K+1, N, N] network output
            labels: [B, K+1, N, N] labels
        Returns:
            scalar loss
        Note:
            Loss should only be calculated on the N test images
        """
        test_preds = preds[:, -1, :, :]
        test_labels = labels[:, -1, :, :]
        return F.cross_entropy(test_preds, test_labels)


def train_step(images, labels, model, optim, eval=False):
    predictions = model(images, labels)
    loss = model.loss_function(predictions, labels)
    if not eval:
        optim.zero_grad()
        loss.backward()
        optim.step()
    return predictions.detach(), loss.detach()