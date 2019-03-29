import os
import logging
import time
from functools import wraps

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
    """"""
    def __init__(self, epoch):
        self.epoch = epoch

    def train(self):
        self.run_epoch()

    @timer
    def run_epoch(self):
        self.run_step()

    def run_step(self):
        print('This needs to be overrided.')


class Trainer(BaseTrainer):
    def __init__(self, epoch):
        super(Trainer, self).__init__(epoch)

    @timer
    def run_epoch(self):
        self.run_step()

    def run_step(self):
        print('Updating weights.')


trainer = Trainer(5)
trainer.run_epoch()
