"""Simulates synthetic k-space.

Takes Cartesian fully sampled phase contrast images from JML's CHD study and
simulates synthetic radial k-space, undersamples, and reconstructs
'zero-filled' images. So far this script is for DAO images.
"""

import os
from math import pi
from pathlib import Path

import numpy as np
import scipy.io as sio

from flow.pynufft.simulate import calculate_full_kspace, simulate_acq
from flow.pynufft.simulate import gate, recon_cine
from flow.pynufft.kspace import get_traj, create_nufft_list


in_dir = Path('/media/chris/Data/neonatalJML/interim_3/DAO')
out_dir = Path('/media/chris/Data/neonatalJML/processed/DAO')

# All the images in this folder should be (132 x 176 x 25)
# Crop the centre of the image to shape:
nx = 132
ny = nx
ns = nx*2
na = round(nx*pi/2)  # Number of radial profiles per time frame
nt = 25
spokes = round(na*nt)  # Max number of spokes in the simulation
tr = 6.5  # Repetition time
i1 = int(176/2 - ny/2)
i2 = i1 + ny
# Acceleration factor
af = 4
# Heart rate (bpm)
hr = 150
hr = np.ones(spokes, dtype=np.float32) * hr

# Create the coordinates of the golden-angle radial k-space trajectory
radial_traj = get_traj(ns, spokes, theta_init=90)
radial_traj = np.reshape(radial_traj, (ns, spokes, 2))
# Create PyNUFFT
max_spokes = 500
nufft_list, nop = create_nufft_list(radial_traj, max_spokes, nx, ny, ns)

files = os.listdir(in_dir)
idx = files.index('keep_key.mat')
del files[idx]  # Not an image

num_files = len(files)

for i, file in enumerate(files):
    print(f'..Simulating {i}/{num_files}.')
    img = sio.loadmat(in_dir/file)['img']
    img = img[:, i1:i2, :]  # Crop
    # Calculate the full k-space
    ks = calculate_full_kspace(img, nufft_list, nop, nx, ny, nt, ns, spokes)
    # Use simulated heart rate to simulate k-space acquisition
    kspace = simulate_acq(ks, hr, tr, spokes, nt, ns)
    # Keep only the beginning portion to simulate undersampling
    ks_us = np.copy(kspace)
    ks_us[:, round(spokes / af):] = 0
    # Gate and reconstruct with the heart rate
    kspace_gated, cp_count_gated = gate(ks_us, hr, nt, tr, method='nearest')
    cine = recon_cine(kspace_gated, nx, ny, nt, ns, nufft_list, max_spokes,
                      cp_count_gated, density='voronoi')
    # Save
    savename = out_dir/file
    sio.savemat(savename, {'original': img, 'undersampled': cine})
