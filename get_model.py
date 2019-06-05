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
