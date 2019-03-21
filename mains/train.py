"""Main function for training."""

import numpy as np

import torch
import torch.nn as nn
import torch.optim. as optim

from flow.models.u_net import UNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = UNet()

    loss_function = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    net.to(device)

    max_epoch = 2

    for epoch in range(max_epoch):
        for minibatch in range(num_batch):
            inputs, truth = get_next_batch()

            inputs, truth = inputs.to(device), truth.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_function(outputs, truth)

            loss.backward()

            optimizer.step()

    return


if __name__ == '__main__':
    main()
