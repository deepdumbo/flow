"""Test network."""
import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from flow.models.unet import UNet3D
from flow.data.neonatalJML.neonatalJML import NeonatalPCDataset


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data_dir = 'C:/Users/Chris/flow/data/neonatalJML/processed/DAO'

validset = NeonatalPCDataset(data_dir, train=False)
# Create iterable
validloader = DataLoader(validset,
                         batch_size=4,
                         shuffle=False,
                         num_workers=4)

model = UNet3D(in_channels=2, out_channels=2, task='regression')
model.to(device)

loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

model_path = 'C:/Users/Chris/flow/neonatal_recon/saved_models/unet.pth'
model.load(model_path, optimizer, device=device)

dataiter = iter(validloader)

minibatch = dataiter.next()

inputs = minibatch[0]
inputs = inputs.to(device)

model.eval()
recon = model(inputs)

recon = recon.detach().cpu().numpy()

out0 = recon[0]
out1 = recon[1]
out2 = recon[2]

out0 = np.sqrt(out0[0]**2 + out0[1]**2)

plt.imshow(img0[:, :, 20])
plt.show()

truth = minibatch[1]
truth = truth.detach().cpu().numpy()

img0 = truth[0]
img1 = truth[1]
img2 = truth[2]

img0 = np.sqrt(img0[0]**2 + img0[1]**2)

plt.imshow(np.concatenate((out0[:, :, 5], img0[:, :, 5]), axis=1))
plt.show()
