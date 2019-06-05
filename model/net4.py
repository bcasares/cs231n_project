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

class Net(nn.Module):
    def __init__(self, params):
        super().__init__()
        in_channel, channel_1, channel_2, num_classes_cnn, num_classes_info_vec = (3, 32, 16, 512, 1)
        self.steeet_view = CNNImages(in_channel, channel_1, channel_2, 6)

    def forward(self, x):
        out = self.steeet_view(x)
        scores = out
        return F.log_softmax(scores, dim=1)

def loss_fn(outputs, labels):
    """
    Computes the loss: Huber loss for this project

    """
    # loss = F.smooth_l1_loss(outputs, labels)
    # loss = F.mse_loss(outputs, labels)
    num_examples = outputs.size()[0]
    return -torch.sum(outputs[range(num_examples), labels])/num_examples
    # loss = F.cross_entropy(outputs, labels)
    # return loss

def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.
    Args:
        outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]
    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs==labels)/float(labels.size)


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
}
