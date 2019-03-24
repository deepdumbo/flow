import os
import logging

import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """Implements saving and loading methods. To be inherited."""
    def __init__(self):
        super(BaseModel, self).__init__()
        self.epoch = 0  # Number of epochs completed
        self.global_step = 0
        self.loss = []

    def save(self, model_path, optimizer):
        # Append epoch to save name
        filename_parts = model_path.split('.')
        savename = f'{filename_parts[0]}_{self.epoch:05d}.{filename_parts[1]}'
        # Save
        torch.save({'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': self.epoch,
                    'global_step': self.global_step,
                    'loss': self.loss},
                   savename)
        logging.info(f'Model saved: {savename}')

    def load(self, model_path, optimizer, device):
        save_dir = '/'.join(model_path.split('/')[:-1])
        files = os.listdir(save_dir)
        if not files:  # If empty do not load anything
            return
        # Newest save
        savename = f'{save_dir}/{sorted(files)[-1]}'
        # Load
        checkpoint = torch.load(savename, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.loss = checkpoint['loss']
        logging.info(f'Model loaded: {savename}')

    def print_state_dict(self):
        for param_tensor in self.state_dict():
            print(param_tensor, '\t', self.state_dict()[param_tensor].device,
                  '\t', self.state_dict()[param_tensor].size())
