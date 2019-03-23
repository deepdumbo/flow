import logging

import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """Implements saving and loading methods. To be inherited."""
    def __init__(self):
        super(BaseModel, self).__init__()
        self.epoch = torch.tensor(0)
        self.global_step = torch.tensor(0)
        self.loss = torch.tensor(0.0, dtype=torch.float32)

    def save(self, model_path):
        torch.save({'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': self.epoch,
                    'global_step': self.global_step,
                    'loss': self.loss},
                   model_path)
        logging.info('Model saved.')

    def load(self, model_path, device=torch.device('cpu')):
        checkpoint = torch.load(model_path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.loss = checkpoint['loss']

    def set_optimizer(self, optimizer):
        """Not sure if can save/load optimizer like this."""
        self.optimizer
