"""Main function for training."""

import argparse

import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from flow.utils.config import Config
from flow.models.u_net import UNet
from flow.data_loaders.fetalsheepseg import FetalSheepSegDataset


def main(config):
    # Chooses device. Prefers GPU.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Handle to dataset
    data_dir = '/home/chris/flow/data/FetalSheepSegmentation/processed'
    trainset = FetalSheepSegDataset(data_dir, train=True)
    validset = FetalSheepSegDataset(data_dir, train=False)
    # Create iterable
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True)
    validloader = DataLoader(validset, batch_size=4, shuffle=True)

    # Load neural net
    net = UNet()

    # Create loss function and optimizer
    loss_function = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    # Move parameters to chosen device
    net.to(device)

    max_epoch = 1

    for epoch in range(max_epoch):
        print('Epoch # {}'.format(epoch + 1))
        for i, minibatch in enumerate(trainloader):
            inputs, truth = minibatch
            print('Batch # {}'.format(i + 1))
            print(type(inputs))
            print(inputs.dtype)
            print(inputs.shape)
            print(type(truth))
            print(truth.dtype)
            print(truth.shape)
            """
            inputs, truth = inputs.to(device), truth.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_function(outputs, truth)

            loss.backward()

            optimizer.step()
            """
    return


if __name__ == '__main__':
    # Set up command line argument for the config file.
    # Uses default config file if no command line argument is present.
    default_config = '/home/chris/flow/configs/fetalsheepseg.json'
    parser = argparse.ArgumentParser(description='Training.')
    parser.add_argument('configfile', help='Config .json file.', nargs='?',
                        default=default_config)
    args = parser.parse_args()
    print(args.configfile)
    config = Config(args.configfile)
    main(config)
