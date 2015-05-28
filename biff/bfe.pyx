# coding: utf-8

""" Basis Function Expansion in Cython """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

import numpy as np
cimport numpy as np

cdef extern from "math.h":
    double exp(double x) nogil

cdef extern from "gsl/gsl_sf_gamma.h":
    double gsl_sf_lngamma(double x) nogil
    double gsl_sf_fact(unsigned int n) nogil

__all__ = ['acceleration']

cpdef acceleration(double[::1] r, double[::1] a,
                   int nmax, int lmax):

    cdef unsigned int n,l

    # ----------------------------------------------------------------
    # This stuff was all in a "firstc" or "first calculation" check
    #   in Fortran. We may not need to compute this every time...but
    #   for now, just do it™
    cdef double arggam, deltam0

    anltilde = np.zeros((nmax+1, lmax+1))
    twoalpha = np.zeros(lmax+1)
    coeflm = np.zeros((lmax+1,lmax+1))
    dblfact = np.zeros(lmax+1)
    dblfact[0] = 1.

    for l in range(2,lmax+1):
        dblfact[l] = dblfact[l-1] * (2.*l - 1.)

    for n in range(nmax+1):
        for l in range(lmax+1):
            knl = 0.5*n*(n+4.*l+3.)+(l+1.)*(2.*l+1.)
            anltilde[n,l] = -2.**(8.*l+6.)*gsl_sf_fact(n)*(n+2.*l+1.5)
            arggam = 2.*l+1.5
            anltilde[n,l] *= (exp(gsl_sf_lngamma(arggam)))**2
            anltilde[n,l] /= (4.*np.pi*knl*gsl_sf_fact(n+4*l+2))

    for l in range(lmax+1):
        twoalpha[l] = 2.0*(2.*l+1.5)

        for m in range(l+1):
            deltam0 = 2.
            if m == 0:
                deltam0 = 1.
            coeflm[l,m] = (2.*l+1.)*deltam0 * gsl_sf_fact(l-m)/gsl_sf_fact(l+m)


    for n in range(1,nmax+1):
        c3[n] = 1./(n+1)
        for l in range(lmax+1):
            c1[n,l] = 2.0*n + twoalpha[l]
            c2[n,l] = n-1.0 + twoalpha[l]
    # ----------------------------------------------------------------

