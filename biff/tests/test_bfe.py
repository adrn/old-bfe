# coding: utf-8

""" Test ...  """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import logging

# Third-party
import numpy as np
from astropy import log as logger

# Project
from ..bfe import acceleration, value

logger.setLevel(logging.DEBUG)

this_path = os.path.abspath(__file__)
data_path = os.path.abspath(os.path.join(this_path, "../../../fortran/"))

def test_compare_to_fortran():

    nmax = 6
    lmax = 4

    # compute from fortran
    accpot = np.loadtxt(os.path.join(data_path, "pot-acc.dat"))
    scf_acc = accpot[:,:3]
    scf_pot = accpot[:,3]

    # read in coefficients from file:
    _sin_c,_cos_c = np.loadtxt(os.path.join(data_path, "coeff.dat"), skiprows=1).T

    sin_c = np.zeros((nmax+1,lmax+1,lmax+1))
    cos_c = np.zeros((nmax+1,lmax+1,lmax+1))
    i = 0
    for n in range(nmax+1):
        for l in range(lmax+1):
            for m in range(l+1):
                sin_c[n,l,m] = _sin_c[i]
                cos_c[n,l,m] = _cos_c[i]
                i += 1

    xyz = np.loadtxt(os.path.join(data_path, "positions.dat"), skiprows=1)

    # compute value of potential
    pot = np.zeros(len(xyz))
    value(xyz, pot, sin_c, cos_c, nmax=nmax, lmax=lmax)
    print((pot - scf_pot)/scf_pot)

    # compute acceleration
    acc = np.zeros((len(xyz),3))
    acceleration(xyz, acc, sin_c, cos_c, nmax=nmax, lmax=lmax)
    diff = ((acc - scf_acc)/scf_acc)
    print(diff[:,0])
    print(diff[:,1])
    print(diff[:,2])
