""" Functions for hodogram analysis"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def hodogram(comp1, comp2, compz, title="", ndt=0.001, azimuth=None, incidence=None):
    """
    Plot an hodogram from 3C data

    :param comp1: (numpy.ndarray) for horizontal component 1
    :param comp2: (numpy.ndarray) for horizontal component 2
    :param compz: (numpy.ndarray) for vertical component
    :param title: Figure title
    :param ndt: Sampling interval delta in second
    :param azimuth: Azimuth angle solution to superimpose on hodogram
    :param incidence: Incidence angle solution to superimpose on hodogram
    :return:
    """

    fig, axs = plt.subplots(2, 2)
    # fig.set_size_inches(8, 8)
    axs[0][0].plot(comp1, comp2)
    axs[0][0].set_xlabel('horizontal 1')
    axs[0][0].set_ylabel('horizontal 2')

    axs[0][1].plot(comp1, compz)
    axs[0][1].set_xlabel('horizontal 1')
    axs[0][1].set_ylabel('vertical')

    axs[1][0].plot(comp2, compz)
    axs[1][0].set_xlabel('horizontal 2')
    axs[1][0].set_ylabel('vertical')

    # Set axes limits
    mmax = np.max(np.abs(np.array([comp1, comp2, compz])))
    axs[0][0].set_xlim(-mmax, mmax)
    axs[0][0].set_ylim(-mmax, mmax)
    axs[0][1].set_xlim(-mmax, mmax)
    axs[0][1].set_ylim(-mmax, mmax)
    axs[1][0].set_xlim(-mmax, mmax)
    axs[1][0].set_ylim(-mmax, mmax)

    # Add azimuth and incidence angle to hodogram
    if azimuth is not None:
        azi_rad = np.deg2rad(azimuth)
        x = mmax * np.sin(azi_rad)
        y = mmax * np.cos(azi_rad)
        axs[0][0].plot([x, -x], [y, -y], marker=None, linestyle="-", color="red", linewidth=2)
    if incidence is not None:
        inc_rad = np.deg2rad(incidence)
        x = mmax * np.sin(inc_rad)
        y = -mmax * np.cos(inc_rad)
        axs[0][1].plot([x, -x], [y, -y], marker=None, linestyle="-", color="red", linewidth=2)
        axs[1][0].plot([x, -x], [y, -y], marker=None, linestyle="-", color="red", linewidth=2)

    t = np.arange(0, len(comp1) * ndt, ndt)
    axs[1][1].plot(t, comp1, marker=None, color='blue')
    axs[1][1].plot(t, comp2, marker=None, color='green')
    axs[1][1].plot(t, compz, marker=None, color='red')
    if azimuth is not None and incidence is not None:
        plt.suptitle("Azimuth = %f, Incidence = %f" % (azimuth, incidence))
    else:
        plt.suptitle(title)


def polarization_svd(datax, datay, dataz):
    """
    Get azimuth and incidence angle from 3C input data (numpy.ndarray)
    :param datax: Horizontal component along "X"/Easting
    :param datay: Horizontal component along "Y"/Northing
    :param dataz: Vertical component
    :return:
    """
    covmat = np.zeros([3, 3])
    covmat[0][0] = np.cov(datax, rowvar=False)
    covmat[0][1] = covmat[1][0] = np.cov(datax, datay, rowvar=False)[0, 1]
    covmat[0][2] = covmat[2][0] = np.cov(datax, dataz, rowvar=False)[0, 1]
    covmat[1][1] = np.cov(datay, rowvar=False)
    covmat[1][2] = covmat[2][1] = np.cov(dataz, datay, rowvar=False)[0, 1]
    covmat[2][2] = np.cov(dataz, rowvar=False)
    eigenvec, eigenval, unit_vec = (np.linalg.svd(covmat))
    azimuth = math.degrees(np.arctan2(eigenvec[0, 0] * np.sign(eigenvec[2, 0]), eigenvec[1, 0] * np.sign(eigenvec[2, 0])))
    incidence = math.degrees(np.arccos(eigenvec[2, 0]))
    return azimuth, incidence


def dip_azimuth2nez_base_vector(dip, azimuth):
    """
    Helper function converting a vector described with azimuth and dip of unit
    length to a vector in the NEZ (North, East, Vertical) base.

    The definition of azimuth and dip is according to the SEED reference
    manual.
    """
    dip = np.deg2rad(dip)
    azimuth = np.deg2rad(azimuth)

    return np.array([np.cos(azimuth) * np.cos(dip),
                     np.sin(azimuth) * np.cos(dip),
                     -np.sin(dip)])


def rotate_to_zne(comp1, comp2, comp3, inc, baz):
    rotmat_inc = R.from_euler('y', inc, degrees=True).as_matrix()  # inclination correction
    rotmat_baz = R.from_euler('z', baz, degrees=True).as_matrix()  # backazimuth correction
    rotmat = np.dot(rotmat_baz, rotmat_inc)  # combined rotation matrix

    out = np.dot(rotmat, np.array([comp1, comp2, comp3]))
    compN = out[0, :]
    compE = out[1, :]
    compZ = out[2, :]

    return compN, compE, compZ
