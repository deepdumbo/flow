"""Simulations for variable heart rate and data acquisition."""

import math
from math import pi

import numpy as np
import scipy.io as sio
import scipy.stats as ss
from matplotlib import pyplot as plt
import imageio

from flow.pynufft.kspace import get_ramp, get_v_ramp


def plot_hr(rr_t, t, base_rr, std, delta, hr, base_hr):
    f, ax = plt.subplots(2, 1, figsize=(6, 9), sharex=True,
                         constrained_layout=True)

    t = t/1000  # Convert from ms to s

    ax[0].plot(t, rr_t, color='#BE5050')
    ax[0].set_ylabel('RR Interval (ms)')
    txtstr = ('Bounded Random Walk Params:\n'
              + '$RR_{baseline}$ = %.1f ms\n' % base_rr
              + '$σ_{walk}$ = %.2f ms\n' % std
              + '$Δ_{walk}$ = %.2f ms' % delta)
    ax[0].text(0.05, 0.95, txtstr, verticalalignment='top', bbox=dict(fc='w'),
               transform=ax[0].transAxes)

    ax[1].plot(t, hr, color='#BE5050', lw='2')
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Heart Rate (bpm)')
    '''
    txtstr = ('$HR_{baseline}$ = %.1f bpm' % base_hr)
    ax[1].text(0.05, 0.95, txtstr, verticalalignment='top', bbox=dict(fc='w'),
               transform=ax[1].transAxes)
    '''

    f.savefig('hr_sim.png', dpi=200)
    plt.close()
    return


def gate(kspace, hr, nt, tr, method='nearest'):
    """Gates the radial k-space using the given heart rate.

    Args:
        kspace: Simulated k-space of shape [ns, spokes].
        hr: Vector. Given heart rate signal.
        nt: Number of cardiac phases.
        tr: MRI repetition time.
        method: Interpolation method. One of 'nearest', 'linear' or
            'datashare'.

    Returns:
        kspace_gated: Gated k-space of shape [ns, spokes, nt].
        cp_count: Vector of length nt. Number of non-zero spokes in each
            cardiac phase, which is less than or equal to the size of the
            spokes dimension.
    """
    ns, spokes = kspace.shape
    kspace_gated = np.zeros((ns, spokes, nt), dtype=np.complex64)
    # Counter for number of spokes gated to each cardiac phase
    cp_count = np.zeros((nt), dtype=np.float32)
    # Number of beats passed
    ib = 0
    for a in range(spokes):
        # Portion of a heart beat that passed
        db = hr[a] / 60 / 1000 * tr
        # Total number of beats passed
        ib = ib + db
        # Portion of a beat that passed since the last completed beat
        mb = ib % 1
        ind = mb * nt  # Convert to index of cine

        if method == 'nearest':
            # Nearest neighbour
            ind0 = int(round(ind)) % nt
            # Sort
            kspace_gated[:, a, ind0] = kspace[:, a]
            if np.sum(abs(kspace[:, a])) > 0:
                # Condition necessary for undersampled k-spaces
                cp_count[ind0] = cp_count[ind0] + 1  # Add spoke to counter
        elif method == 'linear':
            # Index of the nearest cardiac phase before
            ind1 = math.floor(ind)
            # Index of the nearest cardiac phase after
            ind2 = ind1 + 1
            # Sort
            # Put spoke in the nearest cardiac phase before
            kspace_gated[:, a, ind1] = kspace[:, a] * (ind2 - ind)
            # Put spoke in the nearest cardiac phase after
            kspace_gated[:, a, ind2 % nt] = kspace[:, a] * (ind - ind1)
            if np.sum(abs(kspace[:, a])) > 0:  # If not not sampled
                cp_count[ind1] = cp_count[ind1] + (ind2 - ind)
                cp_count[ind2 % nt] = cp_count[ind2 % nt] + (ind - ind1)
        elif method == 'datashare':
            # Index of the nearest cardiac phase before
            ind1 = math.floor(ind)
            # Index of the nearest cardiac phase after
            ind2 = ind1 + 1
            # Put spoke in the nearest cardiac phase before
            kspace_gated[:, a, ind1] = kspace[:, a]
            # Put spoke in the nearest cardiac phase after
            kspace_gated[:, a, ind2 % nt] = kspace[:, a]
            if np.sum(abs(kspace[:, a])) > 0:  # If not not sampled
                cp_count[ind1] = cp_count[ind1] + 1
                cp_count[ind2 % nt] = cp_count[ind2 % nt] + 1
        else:
            print('Error in choosing the interpolation method.')

    return kspace_gated, cp_count


def recon_cine(ksf, nx, ny, nt, ns, nufft_list, max_spokes, cp_count,
               density='absk'):
    """Reconstructs a cine.

    Args:
        ksf: K-space of shape [ns, spokes, nt]. This array contains allocated
            space for all spokes for each cardiac phase, i.e. spokes is the
            total number of spokes in the acquisition.
        nt: Number of cardiac phases in the image.
        ns: Number of k-space samples per radial profile.
        nufft_list: List of planned PyNUFFT objects. Each object performs the
            transform for a subset of spokes. E.g. first operator is for spokes
            1-500, second operator for spokes 501-1000, etc.
        max_spokes: Maximum number of spokes in an NUFFT operator.
        cp_count: Vector of length nt. Number of non-zero spokes in each
            cardiac phase, which is less than or equal to the size of the
            spokes dimension.
        density: Method for density compensation. One of 'absk' or 'voronoi'.

    Returns:
        cine: Reconstructed image. Shape [nx, ny, nt].
    """
    fs = nx*pi/2
    spokes = ksf.shape[1]
    # Create ramp filter
    if density == 'absk':
        ramp = get_ramp(ns, 1)
        ramp = np.reshape(ramp, (ns, 1))
        ramp = np.tile(ramp, (1, spokes))
    elif density == 'voronoi':
        # Calculate golden angles (hard-coded)
        tau = (np.sqrt(5) + 1) / 2
        golden_angle = 180 / tau
        angles = np.vstack(list(range(spokes))).astype(np.float64)
        angles = angles[:, 0]
        angles = angles*golden_angle + 90
        # Full shape ramp
        v_ramp = np.zeros((ns, spokes, nt), dtype=np.float32)
        for k in range(nt):
            # This variable d marks whether a spoke exists or not
            d = np.sum(np.abs(ksf[:, :, k]), axis=0)
            angles_t = angles[d > 0]
            v_ramp0 = get_v_ramp(ns, angles_t.shape[0], angles_t)
            # Put back into the full shape
            s = 0
            for j in range(spokes):
                if d[j] > 0:
                    v_ramp[:, j, k] = v_ramp0[:, s]
                    s = s + 1

    cine = np.zeros((nx, ny, nt), dtype=np.complex64)
    for k in range(nt):
        recon_t = np.zeros((nx, ny, len(nufft_list)), dtype=np.complex64)
        for i in range(len(nufft_list)):
            if density == 'voronoi':
                ramp = v_ramp[:, :, k]
            elif density == 'absk':
                ramp = get_ramp(ns, 1)
                ramp = np.reshape(ramp, (ns, 1))
                ramp = np.tile(ramp, (1, spokes))
            # Reconstruct a subset of spokes (of a cardiac phase)
            # Current section of k-space
            ks_curr = ksf[:, i*max_spokes:(i+1)*max_spokes, k]
            ramp = ramp[:, i*max_spokes:(i+1)*max_spokes]
            n_spokes = ks_curr.shape[1]  # Size of spokes dimension
            dummy = np.multiply(ks_curr, ramp)  # Density compensation
            dummy = np.reshape(dummy, (ns*n_spokes))
            dummy = nufft_list[i].adjoint(dummy)  # Inverse
            # Normalize
            dummy = dummy*(ns/nx)*np.sqrt(nx*ny)
            recon_t[:, :, i] = dummy

        recon_t = np.sum(recon_t, axis=2) * fs / cp_count[k]
        cine[:, :, k] = recon_t
    return cine


def simulate_acq(ks, hr, tr, spokes, nt, ns):
    """Simulates the acquisition of k-space under the given heart rate signal.

    Interpolates the k-space in time depending on the variable heart rate.

    Arguments:
        ks: The full k-space (all spokes at every cardiac phase).
            Shape [ns, spokes, nt].
        hr: Vector. Simulated heart rate signal.
        tr: MRI repetition time. Resolution of the heart rate signal.

    Returns:
        kspace: Simulated k-space.
    """
    # Simulate acquired k-space with variable heart rate
    kspace = np.zeros((ns, spokes), dtype=np.complex64)
    # Number of beats passed
    ib = 0
    for a in range(spokes):
        # Portion of a heart beat that passed
        db = hr[a] / 60 / 1000 * tr
        # Total number of beats passed
        ib = ib + db
        # Portion of a beat that passed since the last completed beat
        mb = ib % 1
        ind = mb * nt  # Convert to index of cine
        ind0 = math.floor(ind)  # Index of the frame before
        ind1 = ind0 + 1  # Index of the following frame
        # Linearly interpolate that spoke between two cardiac phases
        kspace[:, a] = ((ind1-ind)*ks[:, a, ind0] +
                        (ind-ind0)*ks[:, a, (ind1 % nt)])
    return kspace


def calculate_full_kspace(img, nufft_list, nop, nx, ny, nt, ns, spokes):
    """Calculate the full k-space for each frame.

    Every spoke in the acquisition is calculated for each frame.

    Args:
        img: Input image.
        nufft_list: List of planned PyNUFFT objects.

    Returns:
        ks: The full k-space (all spokes at every cardiac phase).
            Shape [ns, spokes, nt].
    """
    ks = np.zeros((ns, spokes, nt), dtype=np.complex64)
    for k in range(nt):
        ks_t = []  # List to hold the transform of each section
        for i in range(nop):
            # Forward transform to get each set of spokes
            dummy = nufft_list[i].forward(img[:, :, k]) / np.sqrt(nx*ny)
            dummy = np.reshape(dummy, (ns, -1))
            ks_t.append(dummy)
        ks_t = np.concatenate(ks_t, axis=1)
        ks[:, :, k] = ks_t
    return ks


def simulate_rr(base_rr, std, delta, spokes, tr):
    '''Simulates a fetal RR signal using a bounded random walk.

    The length of each step is normally distributed.

    Arguments:
        base_rr: Baseline RR interval (ms).
        std: Standard deviation of the RR intervals (ms).
        delta: Strength of the bias bounding the walk around the baseline
            value. Should be a value between 0 and 1.
        spokes: Total number of spokes to acquire.
        tr: MRI repetition time.

    Returns:
        rr: Vector. Simulated RR intervals (ms).
        rr_t: Vector. Simulated RR intervals as a function of time. Duration is
            the total acquisition time.
        tp: Time points of each spoke (ms) (ungated).
        trigger_times: Trigger times (Times of R-waves) (ms).
    '''
    # Total acquisition time (ms)
    acq_time = spokes*tr

    # List to hold sequence of RR intervals
    rr = []
    # First RR value
    rr.append(base_rr + np.random.normal(0, std))
    # Holds elapsed time in the simulation (ms)
    elapsed_time = rr[0]
    # Calculate the trigger times (ms) (the beginning of the RR interval)
    trigger_times = [0, rr[0]]
    # Index for current beat
    b = 1
    while elapsed_time < acq_time:  # Ensures the heart beats longer than acq
        # Mean of the normally distributed step length
        mean = base_rr - rr[b - 1]
        # Random walk and append to list of RR intervals
        rr.append(rr[b - 1] + np.random.normal(delta*mean, std))
        elapsed_time = elapsed_time + rr[b]
        trigger_times.append(elapsed_time)
        b = b + 1

    rr = np.vstack(rr).astype(np.float32)  # Convert to column vector
    trigger_times = np.vstack(trigger_times).astype(np.float32)

    # Time (ms)
    tp = np.linspace(tr, acq_time, num=spokes)
    # Initialize the RR signal as a function of time
    rr_t = np.zeros(spokes, dtype=np.float32)

    timer = 0
    n = 0  # Heart beat index
    rr_sum = rr[0]
    for a in range(spokes):
        timer = timer + tr
        if timer > rr_sum:  # If passing to next RR interval
            ind1 = timer - tr
            ind2 = timer
            rr_t[a] = (rr_sum-ind1)/tr*rr[n] + (ind2-rr_sum)/tr*rr[n+1]
            rr_sum = rr_sum + rr[n+1]  # Time when next beat ends
            n = n + 1  # Move to next beat
        else:
            rr_t[a] = rr[n]

    return rr, rr_t, tp, trigger_times


def main():
    return


if __name__ == '__main__':
    main()
