# coding: utf-8

""" Test ...  """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Standard library
import os
import logging

# Third-party
import numpy as np
import astropy.units as u
from astropy.constants import G as _G
from astropy import log as logger
from scipy.misc import derivative

# Project
from ..bfe import acceleration, value

logger.setLevel(logging.DEBUG)

this_path = os.path.abspath(__file__)
data_path = os.path.abspath(os.path.join(this_path, "../../../fortran/"))

def test_dumb():

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

    xyz = np.array([[8.,0.8,0.8]])
    # pot = np.zeros(len(xyz))
    # value(xyz, pot, sin_c, cos_c, nmax=nmax, lmax=lmax)

    acc = np.zeros_like(xyz)
    acceleration(xyz, acc, sin_c, cos_c, nmax=nmax, lmax=lmax)

# def test_hernquist():
#     nmax = 2
#     lmax = 2

#     sin_coeff = np.zeros((nmax+1,lmax+1,lmax+1))
#     cos_coeff = np.zeros((nmax+1,lmax+1,lmax+1))
#     cos_coeff[0,0,0] = 1.

#     r = np.random.uniform(-10,10,size=(100,3))
#     a = np.zeros_like(r)
#     p = np.zeros(len(r))

#     hpot = gp.HernquistPotential(m=1., c=1., units=galactic)
#     print(hpot.acceleration(r))
#     print(hpot.value(r))

#     bfe.acceleration(r, a, sin_coeff, cos_coeff, nmax, lmax)
#     bfe.value(r, p, sin_coeff, cos_coeff, nmax, lmax)
#     print(a * hpot.G)
#     print(p * hpot.G)

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

def test_numerical_gradient_vs_gradient():
    cos_coeff = np.array([[[1.509, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [-2.606, 0.0, 0.665, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [6.406, 0.0, -0.66, 0.0, 0.044, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [-5.859, 0.0, 0.984, 0.0, -0.03, 0.0, 0.001]], [[-0.086, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [-0.221, 0.0, 0.129, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [1.295, 0.0, -0.14, 0.0, -0.012, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], [[-0.033, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [-0.001, 0.0, 0.006, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], [[-0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
    sin_coeff = np.zeros_like(cos_coeff)
    nmax = sin_coeff.shape[0]-1
    lmax = sin_coeff.shape[1]-1

    def partial_derivative(func, point, ix=0, **kwargs):
        xyz = np.array(point)
        pot = np.zeros_like(xyz)
        xyz = np.ascontiguousarray(xyz[None])
        def wraps(a):
            xyz[:,ix] = a
            func(xyz, pot, sin_coeff, cos_coeff, nmax, lmax)
            return pot[0]
        return derivative(wraps, point[ix], **kwargs)

    dx = 1E-6
    max_x = 3.

    grid = np.linspace(-max_x,max_x,8)
    grid = grid[grid != 0.]
    xyz = np.ascontiguousarray(np.vstack(map(np.ravel, np.meshgrid(grid,grid,grid))).T)

    num_grad = np.zeros_like(xyz)
    for i in range(xyz.shape[0]):
        num_grad[i] = np.array([partial_derivative(value, xyz[i], ix=ix, n=1, dx=dx, order=5) for ix in range(3)]).T

    grad = np.zeros_like(xyz)
    acceleration(xyz, grad, sin_coeff, cos_coeff, nmax, lmax)
    grad *= -1.

    np.testing.assert_allclose(num_grad, grad, rtol=1E-5)

def test_against_gary():
    cos_coeff = np.array([[[1.509, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [-2.606, 0.0, 0.665, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [6.406, 0.0, -0.66, 0.0, 0.044, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [-5.859, 0.0, 0.984, 0.0, -0.03, 0.0, 0.001]], [[-0.086, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [-0.221, 0.0, 0.129, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [1.295, 0.0, -0.14, 0.0, -0.012, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], [[-0.033, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [-0.001, 0.0, 0.006, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], [[-0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
    sin_coeff = np.zeros_like(cos_coeff)
    nmax = sin_coeff.shape[0]-1
    lmax = sin_coeff.shape[1]-1

    max_x = 10.
    grid = np.linspace(-max_x,max_x,8)
    grid = grid[grid != 0.]
    xyz = np.ascontiguousarray(np.vstack(map(np.ravel, np.meshgrid(grid,grid,grid))).T)
    # xyz = np.array([[1.,1.,1.]])

    grad = np.zeros_like(xyz)
    acceleration(xyz, grad, sin_coeff, cos_coeff, nmax, lmax)
    grad *= -1.

    val = np.zeros(len(grad))
    value(xyz, val, sin_coeff, cos_coeff, nmax, lmax)

    # gary
    import gary.potential as gp
    from gary.units import galactic
    G = _G.decompose(galactic).value
    potential = gp.SCFPotential(m=1/G, r_s=1.,
                                sin_coeff=sin_coeff, cos_coeff=cos_coeff,
                                units=galactic)
    gary_val = potential.value(xyz)
    gary_grad = potential.gradient(xyz)

    np.testing.assert_allclose(gary_val, val, rtol=1E-5)
    np.testing.assert_allclose(gary_grad, grad, rtol=1E-5)
