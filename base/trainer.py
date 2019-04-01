import os
import logging
import time
from functools import wraps

import numpy as np

import torch
import torch.nn as nn


def log_train(func):
    """Decorates the class method train."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        logging.info('\n---------- TRAINING ----------')
        logging.info(f'Number of samples in training set: {self.num_train}')
        logging.info(f'Number of samples in validation set: {self.num_valid}')
        logging.info(f'Training batch size: {self.batch_size}')

        func(self, *args, **kwargs)

        logging.info('\nTraining completed!')

    return wrapper


def log_epoch(func):
    """Decorates the class method run_epoch."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        logging.info(f'\nEpoch {self.epoch + 1} out of {self.max_epoch}.')
        start_time = time.time()

        func(self, *args, **kwargs)

        logging.info(f'Epoch time: {time.time() - start_time:.4f} s')

    return wrapper


def log_step(func):
    """Decorates the class method run_step."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):

        func(self, *args, **kwargs)

        progress = f'....Batch {self.step+1}/{self.num_batches}. '
        training_loss = f'Training loss: {self.l:.6f}'
        logging.info(progress + training_loss)

    return wrapper


def log_validate(func):
    """Decorates the class method validate."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()

        func(self, *args, **kwargs)

        validation_time = f'Validation time: {time.time() - start_time:.4f} s'
        logging.info('..Running validation. ' + validation_time)

    return wrapper


class BaseTrainer:
    """Base class with basic training procedure.

    All methods should be overridden, except __init__ maybe. Methods do not
    include results saving.
    """
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
