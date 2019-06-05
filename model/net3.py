"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import huber

import torchvision

def Net(params):
    model_conv = torchvision.models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features

    model_conv.fc = nn.Linear(num_ftrs, 1)
    return model_conv

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
