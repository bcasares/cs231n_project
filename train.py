"""Train the model"""

import argparse
import logging
import os

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

import utils
import model.net2 as net
import model.data_loader2 as data_loader
from evaluate import evaluate

from tensorboardX import SummaryWriter

def train(model, optimizer, loss_fn, dataloader, metrics, params, writer, global_step):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            # move to GPU if available
            if params.cuda:
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')

            # train_batch, labels_batch = train_batch.cuda(non_blocking=True), labels_batch.cuda(non_blocking=True)
            # convert to torch Variables
            dtype = torch.float32 # we will be using float throughout this tutorial
            x, x2, x3 = train_batch
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            x2 = x2.to(device=device, dtype=dtype)
            x3 = x3.to(device=device, dtype=dtype)
            train_batch = (x, x2, x3)

            # train_batch = train_batch.to(device=device, dtype=dtype)
            labels_batch = labels_batch.to(device=device, dtype=dtype)

            # compute model output and loss
            output_batch = model(train_batch)
            loss = loss_fn(output_batch, labels_batch)

            # clear previous gradients, compute gradients of all variables wrt loss
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                output_batch = output_batch.data.cpu().numpy()
                labels_batch = labels_batch.data.cpu().numpy()
                # residual = x3.cpu().numpy()

                # compute all metrics on this batch
                summary_batch = {metric:metrics[metric](output_batch, labels_batch)
                                 for metric in metrics}
                summary_batch['loss'] = loss.data.item()
                # summary_batch["explaining_variation"] = np.abs(residual - output_batch)

                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.data.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)
    for k, v in metrics_mean.items():
        if k != "dollar_value":
            writer.add_scalar(tag=k, global_step=global_step, scalar_value=v)
    return metrics_mean


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, metrics, params, model_dir,
                       restore_file, writer):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_acc = np.inf
    best_train_acc = np.inf

    # best_val_acc = 0
    # best_train_acc = 0

    global_step = 0
    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train_metrics = train(model, optimizer, loss_fn, train_dataloader, metrics, params, writer["train"], global_step)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params, writer["eval"], global_step)
        global_step+=1

        train_acc = val_metrics['rmse']
        # train_acc = val_metrics['accuracy']
        # train_acc = train_metrics["huber_loss"]
        is_best_train = train_acc<=best_train_acc
        # is_best_train = train_acc>=best_train_acc

        if is_best_train:
            best_train_acc = train_acc
            best_json_path_train = os.path.join(model_dir, "metrics_training_.json")
            utils.save_dict_to_json(train_metrics, best_json_path_train)

        # val_acc = val_metrics['accuracy']
        val_acc = val_metrics['rmse']
        # val_acc = val_metrics['huber_loss']
        is_best = val_acc<=best_val_acc
        # is_best = val_acc>=best_val_acc


        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()},
                               is_best=is_best,
                               checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best:
            # logging.info("- Found new best rmse")
            logging.info("- Found new best huber_loss")
            best_val_acc = val_acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)



def runTraining(model_dir, data_dir, restore_file):
    json_path = os.path.join(model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()
    print(params.cuda)

    # Set the random seed for reproducible experiments
    torch.manual_seed(231)
    if params.cuda: torch.cuda.manual_seed(231)

    # Addint tensorbard
    if restore_file == None:
        writer_train = SummaryWriter("Tensorboard/" + os.path.join(model_dir,"train") + ".SUNet")
        writer_eval = SummaryWriter("Tensorboard/" + os.path.join(model_dir, "eval") + ".SUNet")
        writer = {"train": writer_train, "eval": writer_eval}

        # writer = SummaryWriter()
    else:
        writer_train = SummaryWriter(log_dir="Tensorboard/" + os.path.join(restore_file, "train") + ".SUNet")
        writer_eval = SummaryWriter(log_dir="Tensorboard/" + os.path.join(restore_file, "eval") + ".SUNet")
        writer = {"train": writer_train, "eval": writer_eval}

    # Set the logger
    utils.set_logger(os.path.join(model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(['train', 'val'], data_dir, params)
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']

    logging.info("- done.")

    # Define the model and optimizer
    model = net.Net(params).cuda() if params.cuda else net.Net(params)
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # fetch loss function and metrics
    loss_fn = net.loss_fn
    metrics = net.metrics

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_dl, val_dl, optimizer, loss_fn, metrics, params, model_dir,
                       restore_file, writer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/HOUSES_SPLIT_SMALL,data/HOUSES_SATELLITE_SPLIT_SMALL', help="Directory containing the dataset")
    parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
    parser.add_argument('--restore_file', default=None,
                        help="Optional, name of the file in --model_dir containing weights to reload before \
                        training")  # 'best' or 'train'
    # Load the parameters from json file
    args = parser.parse_args()
    data_dir = args.data_dir.split(",")
    print(data_dir)
    print(args.model_dir)
    runTraining(model_dir=args.model_dir, data_dir=data_dir, restore_file=args.restore_file)


