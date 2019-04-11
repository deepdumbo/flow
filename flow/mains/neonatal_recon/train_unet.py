"""Main function for training."""

import os
from pathlib import Path
import argparse
import pickle

import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from flow.utils.config import Config
from flow.utils.logger import configure_logger, log_start, log_end
from flow.models.unet import UNet3D
from flow.data.neonatalJML.neonatalJML import NeonatalPCDataset
from flow.base.trainer import BaseTrainer
from flow.base.trainer import log_train, log_epoch, log_step, log_validate


def plot_history(hist, config):
    f = plt.figure(figsize=(14, 6))
    ax = f.subplots(1, 2)
    ax[0].plot(range(len(hist['train_loss'])), hist['train_loss'])
    ax[0].set_xlabel('Step')
    ax[0].set_ylabel('Training Loss')
    ax[0].grid(which='both', alpha=0.25)
    ax[1].plot(range(len(hist['valid_loss'])), hist['valid_loss'])
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('MSE (Validation Data)')
    ax[1].grid(which='both', alpha=0.25)
    f.savefig(f'{config.results_dir}/train_history.png', dpi=200)
    ax[0].clear()
    ax[1].clear()
    f.clear()
    plt.close(f)
    return


class Trainer(BaseTrainer):
    def __init__(self, model, config, loss_function, optimizer, hist, trainset,
                 validset, trainloader, validloader, device):
        super(Trainer, self).__init__(model, config, loss_function, optimizer,
                                      hist, trainset, validset, trainloader,
                                      validloader, device)

    @log_train
    def train(self):
        for self.epoch in range(self.model.epoch, self.max_epoch):
            self.run_epoch()

    @log_epoch
    def run_epoch(self):
        if self.epoch % self.config.validation_period == 0:
            self.validate()

        for self.step, self.minibatch in enumerate(self.trainloader):
            self.run_step()

        self.model.epoch = self.model.epoch + 1
        self.model.save(self.config.model_path, self.optimizer, max_to_keep=1)
        with open(self.config.history_filename, 'wb') as h:
            pickle.dump(self.hist, h, protocol=pickle.HIGHEST_PROTOCOL)

    @log_step
    def run_step(self):
        inputs, truth = self.minibatch
        inputs, truth = inputs.to(self.device), truth.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.loss_function(outputs, truth)
        loss.backward()  # Evaluate gradients
        self.optimizer.step()  # Update network parameters
        self.l = loss.item()
        self.hist['train_loss'].append(self.l)
        self.model.global_step = self.model.global_step + 1

    @log_validate
    def validate(self):
        with torch.no_grad():
            for step, minibatch in enumerate(self.validloader):
                inputs, truth = minibatch
                inputs = inputs.to(self.device)
                truth = truth.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_function(outputs, truth)
                self.hist['valid_loss'].append(loss.item())
        plot_history(self.hist, self.config)


def main(config):
    # Chooses device. Prefers GPU.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_dir = config.data_loader.data_dir
    trainset = NeonatalPCDataset(data_dir, train=True)
    validset = NeonatalPCDataset(data_dir, train=False)
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
    model = UNet3D(in_channels=2, out_channels=2, task='regression')

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

    trainer = Trainer(model, config, loss_function, optimizer, hist, trainset,
                      validset, trainloader, validloader, device)
    trainer.train()


if __name__ == '__main__':
    # Command line arguments for config file and log location
    parser = argparse.ArgumentParser()
    parser.add_argument('configfile',
                        help='Uses json file for configurations.',
                        nargs='?',
                        default=Path(__file__).with_suffix('.json'))
    parser.add_argument('-s', '--screen',
                        help='Log to screen instead.',
                        action='store_true')
    args = parser.parse_args()

    # Read configuration file, set up directories and return object
    config = Config(args.configfile)

    # Set options for logging
    configure_logger(config, args.screen)

    log_start(config)
    main(config)
    log_end()
