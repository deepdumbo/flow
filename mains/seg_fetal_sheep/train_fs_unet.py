"""Main function for training."""

import os
import time
import argparse
import logging
import pickle

import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from flow.utils.config import Config
from flow.utils.logger import configure_logger, log_start, log_end
from flow.models.u_net import UNet3D
from flow.data_loaders.fetalsheepseg import FetalSheepSegDataset
from flow.utils.metrics import dice_coef


def plot_history(hist, config):
    f = plt.figure(figsize=(14, 6))
    ax = f.subplots(1, 2)
    ax[0].plot(range(len(hist['train_loss'])), hist['train_loss'])
    ax[0].set_xlabel('Step')
    ax[0].set_ylabel('Training Loss')
    ax[0].grid(which='both', alpha=0.25)
    ax[1].plot(range(len(hist['metric'])), hist['metric'])
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Dice Coefficient (Validation Data)')
    ax[1].grid(which='both', alpha=0.25)
    f.savefig(f'{config.results_dir}/train_history.png', dpi=200)
    ax[0].clear()
    ax[1].clear()
    f.clear()
    plt.close(f)
    return


def main(config):
    log = logging.getLogger()

    # Chooses device. Prefers GPU.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_dir = config.data_loader.data_dir
    trainset = FetalSheepSegDataset(data_dir, train=True)
    validset = FetalSheepSegDataset(data_dir, train=False)
    # Create iterable
    trainloader = DataLoader(trainset,
                             batch_size=config.data_loader.batch_size,
                             shuffle=config.data_loader.shuffle,
                             num_workers=config.data_loader.num_workers)
    validloader = DataLoader(validset,
                             batch_size=config.data_loader.batch_size*4,
                             shuffle=False,
                             num_workers=config.data_loader.num_workers)

    # Load neural net
    model = UNet3D()

    # Move parameters to chosen device
    model.to(device)

    # Create loss function and optimizer
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Load model
    model.load(config.model_path, optimizer, device=device)

    # Load histories
    if os.path.isfile(config.history_filename):
        with open(config.history_filename, 'rb') as h:
            hist = pickle.load(h)
    else:
        hist = {'train_loss': [],
                'valid_loss': [],
                'metric': []}

    log.info('\n---------- TRAINING ----------')
    log.info(f'Number of samples in training set: {len(trainset)}')
    log.info(f'Number of samples in validation set: {len(validset)}')
    log.info(f'Training batch size: {config.data_loader.batch_size}')
    num_batches = int(np.ceil(len(trainset)/config.data_loader.batch_size))
    max_epoch = config.trainer.max_epoch

    for epoch in range(model.epoch, max_epoch):
        log.info('\nEpoch {} out of {}.'.format(epoch + 1, max_epoch))
        start_time = time.time()

        # Validation
        if epoch % config.validation_period == 0:
            log.info('..Running validation.')
            with torch.no_grad():
                for i, minibatch in enumerate(validloader):
                    inputs, truth = minibatch
                    inputs, truth = inputs.to(device), truth.to(device)
                    outputs = model(inputs)
                    hist['valid_loss'].append(loss_function(outputs, truth))
                    hist['metric'].append(dice_coef(outputs, truth))
                plot_history(hist, config)

        for i, minibatch in enumerate(trainloader):
            inputs, truth = minibatch
            inputs, truth = inputs.to(device), truth.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, truth)

            loss.backward()  # Evaluate gradients

            optimizer.step()  # Update network parameters
            l = loss.item()
            hist['train_loss'].append(l)

            model.global_step = model.global_step + 1
            log.info(f'....Batch {i+1}/{num_batches}. Training loss: {l:.6f}')

        model.epoch = model.epoch + 1
        model.save(config.model_path, optimizer, max_to_keep=2)
        with open(config.history_filename, 'wb') as h:
            pickle.dump(hist, h, protocol=pickle.HIGHEST_PROTOCOL)

        log.info('Epoch time: {:.4f} s'.format(time.time() - start_time))

    return


if __name__ == '__main__':
    # Command line arguments for config file and log location
    parser = argparse.ArgumentParser()
    parser.add_argument('configfile', help='Uses json file.', nargs='?',
                        default=f'{os.path.splitext(__file__)[0]}.json')
    parser.add_argument('-s', '--screen', help='Log to screen instead.',
                        action='store_true')
    args = parser.parse_args()

    # Read configuration file and return object
    config = Config(args.configfile)

    # Set options for logging
    configure_logger(args.screen, config.log_level)

    log_start(config)
    main(config)
    log_end()
