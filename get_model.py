"""Evaluates the model"""

import argparse
import logging
import os

import numpy as np
import torch
from torch.autograd import Variable
import utils
import model.net4 as net
import model.data_loader4 as data_loader

def getModel(model_dir, data_dir, restore_file ="best"):
    json_path = os.path.join(model_dir, 'params.json')
    params = utils.Params(json_path)

    params.cuda = torch.cuda.is_available()     # use GPU is available
    torch.manual_seed(231)
    if params.cuda: torch.cuda.manual_seed(231)

    dataloaders = data_loader.fetch_dataloader(['test'], data_dir, params)
    test_dl = dataloaders['test']

    model = net.Net(params).cuda() if params.cuda else net.Net(params)

    utils.load_checkpoint(os.path.join(model_dir, restore_file + '.pth.tar'), model)

    return model, test_dl

def compute_saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, 3, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    # Make sure the model is in "test" mode
    model.eval()

    # Make input tensor require gradient
    X.requires_grad_()
    scores = model(X)
    scores = scores.gather(1, y.view(-1, 1)).squeeze()
    scores.backward(torch.ones(scores.size()))
    saliency = torch.max(torch.abs(X.grad), dim=1)[0]
    return saliency

def show_saliency_maps(X, y):
    # Convert X and y from numpy arrays to Torch Tensors
    X_tensor = X
    y_tensor = y
    saliency = compute_saliency_maps(X_tensor, y_tensor, model)
    saliency = saliency.numpy()
    N = X.shape[0]
    for i in range(N):
        Xi = X[i].detach().numpy()
        Xi = np.transpose(Xi, (1, 2, 0))
        plt.subplot(2, N, i + 1)
        plt.imshow(Xi)
        plt.axis('off')
#         plt.title(class_names[y[i]])
        plt.subplot(2, N, N + i + 1)
        plt.imshow(saliency[i], cmap=plt.cm.hot)
        plt.axis('off')
        plt.gcf().set_size_inches(12, 5)
    plt.show()



if __name__ == '__main__':
    pass
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--data_dir', default=['data/HOUSES_SPLIT', "data/HOUSES_SATELLITE_SPLIT"], help="Directory containing the dataset")
    # parser.add_argument('--model_dir', default='experiments/MSE/base_model_both_images/', help="Directory containing params.json")
    # parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
    #                      containing weights to load")

    # args = parser.parse_args()
    # model, test_dl = getModel(model_dir=args.model_dir, data_dir=args.data_dir, restore_file=args.restore_file)
    # for t, (X, y) in enumerate(loader_train):
    #     X = X[0]
    #     break
    # show_saliency_maps(X, y)
