import os
import logging
import time
from functools import wraps

import numpy as np

import torch
import torch.nn as nn


def timer(func):
    """Print the runtime of the decorated function"""
    @wraps(func)
    def wrapper_timer(self, *args, **kwargs):
        print(f'Running epoch {self.epoch}.')
        return func(self, *args, **kwargs)
    return wrapper_timer


class BaseTrainer:
    def __init__(self, model, config, loss_function, optimizer, hist, trainset,
                 validset, trainloader, validloader, device):
        self.model = model
        self.config = config
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.hist = hist
        self.trainset = trainset
        self.validset = validset
        self.trainloader = trainloader
        self.validloader = validloader
        self.device = device
        self.num_train = len(trainset)
        self.num_valid = len(validset)
        self.batch_size = self.config.data_loader.batch_size
        self.num_batches = int(np.ceil(self.num_train/self.batch_size))
        self.max_epoch = self.config.trainer.max_epoch

    def train(self):
        for self.epoch in range(self.model.epoch, self.max_epoch):
            self.run_epoch()

    def run_epoch(self):
        if self.epoch % self.config.validation_period == 0:
            self.validate()

        for self.step, self.minibatch in enumerate(self.trainloader):
            self.run_step()

    def run_step(self):
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError
