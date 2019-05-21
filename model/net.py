"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class InfoVector(nn.Module):
    #  Vector
    def __init__(self, in_channel, num_classes):
        super().__init__()
        self.linear_vec = nn.Linear(in_channel, num_classes)  # in and one out

    def forward(self, x):
        out = x
        out = self.linear_vec(out)
        return out


class Net(nn.Module):
    def __init__(self, params):
        super().__init__()
        in_channel, channel_1, channel_2, num_classes = (3, 32, 16, 1)
        self.steeet_view = CNNImages(in_channel, channel_1, channel_2, num_classes)
        self.satellite = CNNImages(in_channel, channel_1, channel_2, num_classes)
        self.info_vector = InfoVector(8, 1)

        # Last Linear Layer
        self.fc_last = nn.Linear(2, num_classes, bias=True)
        nn.init.kaiming_normal_(self.fc_last.weight)

    def forward(self, x):
        x1, x2, x3 = x
        out = self.steeet_view(x1)
        # out2 = self.satellite(x2)
        # out3 = self.info_vector(x3)

#         print(out.size())
#         print(out2.size())

#         combined = torch.cat((out, out2), 1)
#         print(combined.size())

#         out_tot = self.fc_last(combined)
#         print(out_tot.size())

#         scores = out_tot

        scores = out

        return scores

def loss_fn(outputs, labels):
    """
    Computes the loss: Huber loss for this project

    """
    loss = F.smooth_l1_loss(outputs, labels)
    return loss

def calculateMSE(outputs, labels):
    """
    Calculates the RMSE
    """
    rmse_sum = ((outputs - labels)**2).sum()
    rmse = float(rmse_sum) / float(labels.size)
    return rmse

def calculateRMSE(outputs, labels):
    """
    Calculates the RMSE
    """
    rmse_sum = ((outputs - labels)**2).sum()
    rmse = np.sqrt(float(rmse_sum)) / float(labels.size)
    return rmse

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
