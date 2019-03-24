"""Main function for training."""

import time
import argparse
import logging

import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from flow.utils.config import Config
from flow.utils.logger import config_logger, log_start, log_end
from flow.models.u_net import UNet
from flow.data_loaders.fetalsheepseg import FetalSheepSegDataset


def main(config):
    log = logging.getLogger()
    log_start(config)

    log.info('\n---------- TRAINING ----------')
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
                             batch_size=config.data_loader.batch_size,
                             shuffle=False,
                             num_workers=config.data_loader.num_workers)

    # Load neural net
    model = UNet()

    # Move parameters to chosen device
    model.to(device)

    # Create loss function and optimizer
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Load model
    model.load(config.model_path, optimizer, device=device)

    max_epoch = config.trainer.epochs

    log.info('Number of samples in training set: {}'.format(len(trainset)))
    log.info('Batch size: {}'.format(config.data_loader.batch_size))
    num_batches = int(np.ceil(len(trainset)/config.data_loader.batch_size))

    for epoch in range(model.epoch, max_epoch):
        log.info('\nEpoch {} out of {}.'.format(epoch + 1, max_epoch))
        start_time = time.time()

        for i, minibatch in enumerate(trainloader):
            inputs, truth = minibatch
            inputs, truth = inputs.to(device), truth.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, truth)

            loss.backward()  # Evaluate gradients

            optimizer.step()  # Update network parameters
            l = loss.item()
            model.loss.append(l)

            model.global_step = model.global_step + 1
            log.info(f'....Batch {i+1}/{num_batches}. Training loss: {l:.6f}')

        model.epoch = model.epoch + 1
        model.save(config.model_path, optimizer)
        log.info('Epoch time: {:.4f} s'.format(time.time() - start_time))

    log_end()
    return


if __name__ == '__main__':
    # Set up command line argument for specifying config file.
    # Uses default config file if no command line argument is present.
    parser = argparse.ArgumentParser()
    default_config = __file__.split('.')[0] + '.json'
    parser.add_argument('configfile', help='Config .json file.', nargs='?',
                        default=default_config)
    args = parser.parse_args()

    # Read configuration file and return object
    config = Config(args.configfile)

    # Set options for logging
    config_logger(__file__, config.log_level)

    main(config)
