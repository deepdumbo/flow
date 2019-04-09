'''Functions for calculating density compensation and trajectory for NUFFT.'''

import math
from math import pi

import numpy as np

from pynufft import NUFFT_cpu


def get_ramp(ns, na):
    '''Creates the ramp filter for reconstruction of radial data.

    The density compensation is just 'absolute k'.

    Arguments:
        ns: Integer, number of k-space samples per radial profile.
        na: Integer, number of radial profiles per time frame.

    Returns:
        ramp: 1D numpy array of length ns*na and dtype float32.
    '''
    # End-points
    ind1 = math.floor(ns/2)
    ind2 = math.ceil(ns/2 - 1)
    # Create ramp filter
    ramp = np.linspace(-ind1, ind2, num=ns)
    ramp = np.absolute(ramp)
    # Scale
    ramp = pi / na * ramp
    ramp[math.floor(ns/2)] = pi / (4*na)  # Set the value of the mid-point
    # Normalize
    ramp = ramp / np.max(ramp)
    # Repeat the same weights for each angle
    ramp = np.tile(np.expand_dims(ramp, axis=1), (1, na))
    # Reshape to vector (row-major order)
    ramp = np.reshape(ramp, (ns*na)).astype(np.float32)
    return ramp


def get_v_ramp(ns, na, angles):
    '''Creates the ramp filter for reconstruction of radial data.

    Voronoi density compensation for radial. Depends on the angles provided.

    Arguments:
        ns: Integer, number of k-space samples per radial profile.
        na: Integer, number of radial profiles per time frame.
        angles: Vector of angles, in degrees, corresponding to the angular
            rotation of each spoke. 1D numpy array of length na.

    Returns:
        v_ramp: 2D numpy array of shape [ns, na] and dtype float32.
    '''
    if na < 3:
        raise Warning('Too few spokes for Voronoi correction.')

    # End-points
    ind1 = math.floor(ns/2)
    ind2 = math.ceil(ns/2 - 1)
    # Create ramp filter
    v_ramp = np.linspace(-ind1, ind2, num=ns)
    v_ramp = np.absolute(v_ramp)
    # Repeat for each angle
    v_ramp = np.tile(np.expand_dims(v_ramp, axis=1), (1, na))

    # Convert angles to radians
    angles = np.radians(angles)
    # Convert to relative angle between 0 to pi
    angles = angles % pi
    # Sort the angles
    angles_sorted = np.sort(angles)
    # Find arguments that would sort the angles
    inds = np.argsort(angles)

    # Voronoi correction for each spoke (in the sorted order)
    vor = np.zeros(na, dtype=np.float32)
    # Calculate the weighting for the first sorted spoke
    vor[0] = (angles_sorted[1] - (angles_sorted[-1] - pi)) / 2
    # Calculate the weighting for the last sorted spoke
    vor[-1] = ((angles_sorted[0] + pi) - angles_sorted[-2]) / 2
    # Loop over the rest of the spokes
    for i in range(1, na-1):
        vor[i] = (angles_sorted[i+1] - angles_sorted[i-1]) / 2

    # Sort it back
    dphi = np.zeros_like(vor)
    dphi[inds] = vor
    # Correct
    v_ramp = np.multiply(v_ramp, np.expand_dims(dphi, 0))
    # Set the value of the mid-point
    v_ramp[math.floor(ns/2), :] = pi / (4*na)
    # Normalize
    themax = math.floor(ns/2) * pi / na  # To be the same as the abs k ramp
    v_ramp = v_ramp / themax
    # v_ramp = v_ramp / np.max(v_ramp)  # Causes blinking in time
    v_ramp = np.float32(v_ramp)
    return v_ramp


def get_traj(ns, na, theta_init, gan=1):
    '''Creates the coordinates for the golden-angle radial k-space trajectory.

    The angle theta is measured with respect to the kx axis. The coordinates
    are normalized to range from -pi to pi, as required by PyNUFFT. The first
    vector stores the kx coordinate and the second vector stores the ky
    coordinate.

    Arguments:
        ns: Integer, number of k-space samples per radial profile.
        na: Integer, total number of radial profiles across time.
        theta_init: Starting angular position for the trajectory, in degrees.
        gan: Nth golden angle (to switch to using tiny golden angles).

    Returns:
        radial_traj: 2D numpy array of shape [ns*na, 2] and dtype float32.
    '''
    # Golden angle (S. Wundrak et al. 2016)
    tau = (np.sqrt(5) + 1) / 2
    golden_angle = 180 / (tau + gan - 1)
    # End-points
    ind1 = math.floor(ns/2)
    ind2 = math.ceil(ns/2 - 1)
    # First make trajectory at theta = 0
    kx = np.linspace(-ind1, ind2, num=ns)
    ky = np.zeros(ns)

    # Full trajectory
    radial_traj = np.zeros((ns, na, 2), dtype=np.float32)
    for j in range(na):
        angle = np.radians(j * golden_angle + theta_init)
        costheta = math.cos(angle)
        sintheta = math.sin(angle)
        for i in range(ns):
            # Rotated kx coordinate
            radial_traj[i, j, 0] = costheta*kx[i] - sintheta*ky[i]
            # Rotated ky coordinate
            radial_traj[i, j, 1] = sintheta*kx[i] + costheta*ky[i]

    # Normalize to range from -pi to pi
    radial_traj = radial_traj / ind1 * pi
    # Reshape to two vectors (Row-major order)
    radial_traj = np.reshape(radial_traj, (ns*na, 2))
    return radial_traj


def create_nufft_list(radial_traj, max_spokes, nx, ny, ns):
    '''Creates a list of PyNUFFT operators.

    For transforming k-spaces with a large number of spokes (~2000+). Divides
    the transformation into multiple operators to prevent memory overload.

    Arguments:
        radial_traj: The coordinates of the golden-angle radial k-space
            trajectory. Shape [ns, spokes, 2].
        max_spokes: Maximum number of spokes in an NUFFT operator.

    Returns:
        nufft_list: List of planned PyNUFFT objects. Each object performs the
            transform for a subset of spokes. E.g. first operator is for spokes
            1-500, second operator for spokes 501-1000, etc.
        nop: Number of operators.
    '''
    print('Creating PyNUFFT operators...', end=' ')
    spokes = radial_traj.shape[1]
    nop = math.ceil(spokes/max_spokes)  # Divide into this many operators
    nufft_list = []
    for i in range(nop):
        traj = radial_traj[:, i*max_spokes:(i+1)*max_spokes, :]
        n_spokes = traj.shape[1]  # Number of spokes in the subset
        traj = np.reshape(traj, (ns*n_spokes, 2))  # Reshape
        nufft_obj = NUFFT_cpu()  # Create NUFFT object
        nufft_obj.plan(traj, (nx, ny), (ns, ns), (6, 6))  # Plan
        nufft_list.append(nufft_obj)
    print('Created %d operators for a total of %d spokes.\n' % (nop, spokes))
    return nufft_list, nop


def main():
    print('Testing the functions in the nufft_functions module.')
    nx = 128
    ny = 128
    ns = 256
    spokes = 3000
    # Create the coordinates of the golden-angle radial k-space trajectory
    radial_traj = get_traj(ns, spokes, theta_init=90)
    radial_traj = np.reshape(radial_traj, (ns, spokes, 2))
    # Create list of NUFFT operators
    max_spokes = 500
    nufft_list, nop = create_nufft_list(radial_traj, max_spokes, nx, ny, ns)
    # Make ramp filter
    get_ramp(ns, 1)
    print('Testing complete.\n')
    return


if __name__ == '__main__':
    main()
