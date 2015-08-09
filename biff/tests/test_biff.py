# coding: utf-8

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

# Third-party
import numpy as np
from astropy import log as logger
import gary.potential as gp
from gary.units import galactic
from scipy.misc import derivative

# Project
from biff import bfe

# def main():
#     nmax = 6
#     lmax = 4

#     # read full coeff file in...
#     derp = np.loadtxt("fortran/coeff.dat", skiprows=1)

#     sin_coeff = np.zeros((nmax+1,lmax+1,lmax+1))
#     cos_coeff = np.zeros((nmax+1,lmax+1,lmax+1))
#     i = 0
#     # for n in range(nmax+1):
#     #     for l in range(lmax+1):
#     #         for m in range(l+1):
#     #             sin_coeff[n,l,m] = derp[i][0]
#     #             cos_coeff[n,l,m] = derp[i][1]
#     #             i += 1
#     cos_coeff[0,0,0] = -1.

#     # r = np.random.uniform(-1000,1000,size=(100,3))
#     r = np.array([[1100., 0, 0],
#                   [100.,0,0]])
#     a = np.zeros_like(r)
#     p = np.zeros(len(r))

#     hpot = gp.HernquistPotential(m=1., c=1., units=galactic)
#     print(hpot.acceleration(r))
#     print(hpot.value(r))

#     bfe.acceleration(r, a, sin_coeff, cos_coeff, nmax, lmax)
#     bfe.value(r, p, sin_coeff, cos_coeff, nmax, lmax)
#     print(a * hpot.G)
#     print(p * hpot.G)


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

def test_numerical_gradient_vs_gradient():
    dx = 1E-6
    max_x = 3.

    grid = np.linspace(-max_x,max_x,8)
    grid = grid[grid != 0.]
    xyz = np.ascontiguousarray(np.vstack(map(np.ravel, np.meshgrid(grid,grid,grid))).T)

    num_grad = np.zeros_like(xyz)
    for i in range(xyz.shape[0]):
        num_grad[i] = np.array([partial_derivative(bfe.value, xyz[i], ix=ix, n=1, dx=dx, order=5) for ix in range(3)]).T

    grad = np.zeros_like(xyz)
    bfe.acceleration(xyz, grad, sin_coeff, cos_coeff, nmax, lmax)
    grad *= -1.

    np.testing.assert_allclose(num_grad, grad, rtol=1E-5)
