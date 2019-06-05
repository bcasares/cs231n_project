"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import huber

def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image


class CNNImages(nn.Module):
    def __init__(self, in_channel, channel_1, channel_2, num_classes):
        super().__init__()
        self.conv_w1 = nn.Conv2d(in_channel, channel_1, kernel_size=5, padding=2, bias=True)
        nn.init.kaiming_normal_(self.conv_w1.weight)
        self.conv_w2 = nn.Conv2d(channel_1, channel_2, kernel_size=3, padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv_w2.weight)
        self.fc = nn.Linear(channel_2*64*64, num_classes, bias=True)
        nn.init.kaiming_normal_(self.fc.weight)

    def forward(self, x):
        out = x
        out = self.conv_w1(out)
        out = F.relu(out)
        out = self.conv_w2(out)
        out = F.relu(out)
        out = flatten(out)
        out = self.fc(out)
        return out

def build_mlp():
  return nn.Sequential(
    nn.Linear(512*2, 1024),
    nn.ReLU(),
    nn.Linear(1024, 1024),
    nn.ReLU(),
    nn.Linear(512*2, 1024),
    nn.ReLU(),
    nn.Linear(1024, 1),
  )

class Net(nn.Module):
    def __init__(self, params):
        super().__init__()
        in_channel, channel_1, channel_2, num_classes_cnn, num_classes_info_vec = (3, 32, 16, 512, 1)
        self.steeet_view = CNNImages(in_channel, channel_1, channel_2, num_classes_cnn)
        self.satellite = CNNImages(in_channel, channel_1, channel_2, num_classes_cnn)
        self.mlp = build_mlp()

    def forward(self, x):
        street_view_data, satellite_data, _ = x
        out = self.steeet_view(street_view_data)
        out2 = self.satellite(satellite_data)

        combined = torch.cat((out, out2), dim=1)

        out = self.mlp(combined)

        scores = out
        return scores

def loss_fn(outputs, labels):
    """
    Computes the loss: Huber loss for this project

    """
    # loss = F.smooth_l1_loss(outputs, labels)
    loss = F.mse_loss(outputs, labels)
    return loss

def huberLoss(outputs, labels):
    delta = 1
    r = np.abs(outputs-labels)
    loss = huber(delta, r)
    return loss


def calculateMSE(outputs, labels):
    """
    Calculates the RMSE
    """
    mse = ((outputs - labels)**2).mean()
    return mse

def calculateRMSE(outputs, labels):
    """
    Calculates the RMSE
    """
    mse = calculateMSE(outputs, labels)
    return np.sqrt(mse)

def dollarValue(outputs, labels):
    """
    Calculates the dollar value from the RMSE
    """
    rmse = calculateRMSE(outputs, labels)
    dollar_value = np.exp(rmse)
    return dollar_value


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'mse' : calculateMSE,
    'rmse': calculateRMSE,
    'dollar_value' : dollarValue
}
