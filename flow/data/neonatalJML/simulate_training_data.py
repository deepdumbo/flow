'''Simulates training data for the MOG discriminator.

Takes the Sunnybrook Cardiac Data and crops the centre 128x128 pixels, resizes
to 16 frames, simulates corrected and incorrectly gated images, and then saves
to disk.
'''

from math import pi
import os
import time

import numpy as np
import scipy.io as sio

from simulate import recon_cine, gate, simulate_acq, calculate_full_kspace
from nufft_functions import get_traj, create_nufft_list


def get_image(fn):
    '''Opens and resizes the image.'''
    img = sio.loadmat(fn)['Y']
    # Crop (xy)
    crop_length = 128
    i1 = int(256/2 - crop_length/2)
    i2 = i1 + crop_length
    img = img[i1:i2, i1:i2, :]
    # Downsample in the time dimension
    nx, ny, nt = img.shape
    xp = np.linspace(1, 20, num=20)
    x = np.linspace(1, 20, num=16)
    new_img = np.zeros((nx, ny, 16), dtype=np.float32)
    for j in range(ny):
        for i in range(nx):
            fp = img[i, j, :]
            new_img[i, j, :] = np.interp(x, xp, fp)
    return new_img


def process_dir(dir_curr, nx, ny, nt, ns, spokes, tr, nufft_list, nop,
                max_spokes, save_dir, dataset_type):
    file_list = os.listdir(dir_curr)
    for file in file_list:
        t0 = time.time()
        # Current filename
        fn = os.path.join(dir_curr, file)
        print('Processing %s.' % fn, end=' ')
        # Parse file name (for saving later)
        a, b = file.split('.')
        # Open image
        img = sio.loadmat(fn)['dummy']

        img_mean = np.mean(img)
        img_std = np.std(img)
        # Acceleration factors to simulate
        af = [1, 2, 3, 4]
        # Number of incorrectly gated examples to simulate
        num_inc = 30

        # Generate ground truth and incorrect heart rates (bpm)
        mybool = True
        while mybool:
            hr0 = np.random.normal(150, 10)
            hr1 = []
            for _ in range(num_inc):
                hr1.append(np.random.normal(hr0, 5))

            # Convert to RR interval and round (ms)
            rr0 = round(1/hr0*60*1000)
            rr1 = []
            for i in range(num_inc):
                rr1.append(round(1/hr1[i]*60*1000))

            # Convert back to heart rate (bpm)
            hr0 = 1/rr0*1000*60
            for i in range(num_inc):
                hr1[i] = 1/rr1[i]*1000*60

            if hr0 not in hr1:
                if not hr0 < 80:
                    if not any(i < 80 for i in hr1):
                        mybool = False

        # The (ground truth) heart rate is constant for all acquired spokes
        hr = np.ones(spokes, dtype=np.float32) * hr0  # TODO: THIS SHOULD BE IN THE LOOP
        # Calculate the full k-space
        ks = calculate_full_kspace(img, nufft_list, nop, nx, ny, nt, ns,
                                   spokes)
        # Use simulated heart rate to simulate k-space acquisition
        kspace = simulate_acq(ks, hr, tr, spokes, nt, ns)

        # Loop over the acceleration factors
        for curr_af in af:
            # Make a new directory for each acceleration factor
            dirname = os.path.join(save_dir, 'data_af_' + str(curr_af),
                                   dataset_type)
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            # Keep only the beginning portion to simulate undersampling
            ks_us = np.copy(kspace)
            ks_us[:, (round(spokes / curr_af)):] = 0

            # Gate and reconstruct with the correct heart rate
            kspace_gated, cp_count_gated = gate(ks_us, hr, nt, tr,
                                                method='datashare')
            cine_true = recon_cine(kspace_gated, nx, ny, nt, ns, nufft_list,
                                   max_spokes, cp_count_gated,
                                   density='voronoi')
            # Only keep the magnitude
            cine_true = np.float32(np.abs(cine_true))
            # Normalize the image with its mean and std
            cine_true = np.expand_dims((cine_true-img_mean) / img_std, axis=0)
            y_true = np.expand_dims(1, axis=0)
            # Save the same correctly gated cine 30 times
            for i in range(num_inc):
                # Save the correctly gated example
                newname = a + '_1_' + str(i) + '.' + b
                sn = os.path.join(dirname, newname)
                sio.savemat(sn, {'X': cine_true, 'Y': y_true})

            for i in range(num_inc):  # Make 30 incorrectly gated examples
                # Gate and reconstruct with an incorrect heart rate
                hr = np.ones(spokes, dtype=np.float32) * hr1[i]
                kspace_gated, cp_count_gated = gate(ks_us, hr, nt, tr,
                                                    method='datashare')
                cine_false = recon_cine(kspace_gated, nx, ny, nt, ns,
                                        nufft_list, max_spokes, cp_count_gated,
                                        density='voronoi')
                # Only keep the magnitude
                cine_false = np.float32(np.abs(cine_false))
                # Normalize the image with its mean and std
                cine_false = np.expand_dims((cine_false-img_mean) / img_std,
                                            axis=0)
                y_false = np.expand_dims(0, axis=0)
                # Save the incorrectly gated example
                newname = a + '_0_' + str(i) + '.' + b
                sn = os.path.join(dirname, newname)
                sio.savemat(sn, {'X': cine_false, 'Y': y_false})

        print('Elapsed time: %.5fs.' % (time.time() - t0))
    return


def main():
    nx = 128
    ny = 128
    ns = nx*2
    na = round(nx*pi/2)  # Number of radial profiles per time frame
    nt = 16
    spokes = round(na*nt)  # Max number of spokes in the simulation
    tr = 5

    # Create the coordinates of the golden-angle radial k-space trajectory
    radial_traj = get_traj(ns, spokes, theta_init=90)
    radial_traj = np.reshape(radial_traj, (ns, spokes, 2))
    # Create PyNUFFT
    max_spokes = 500
    nufft_list, nop = create_nufft_list(radial_traj, max_spokes, nx, ny, ns)

    # Directory of data
    data_dir = '/project/6016195/fetalmri/Stasis/data'
    dataset_types = ['train', 'valid']
    exp_dir = '/project/6016195/fetalmri/Stasis/experiments/mog_disc_curr'
    save_dir = os.path.join(exp_dir, 'data')
    for dataset_type in dataset_types:
        t0 = time.time()
        # Process the data in this dir
        dir_curr = os.path.join(data_dir, dataset_type)
        print('Processing images in %s.' % dir_curr)
        process_dir(dir_curr, nx, ny, nt, ns, spokes, tr, nufft_list, nop,
                    max_spokes, save_dir, dataset_type)
        print('Elapsed time: %.5fs.' % (time.time() - t0))
        print('')
    return


if __name__ == '__main__':
    main()
