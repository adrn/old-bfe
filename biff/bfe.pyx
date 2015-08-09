# coding: utf-8
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: profile=False

""" Basis Function Expansion in Cython """

from __future__ import division, print_function

__author__ = "adrn <adrn@astro.columbia.edu>"

import numpy as np
cimport numpy as np
from libc.math cimport M_PI

cdef extern from "math.h":
    double exp(double x) nogil
    double sqrt(double x) nogil
    double atan2(double y, double x) nogil
    double cos(double x) nogil
    double sin(double x) nogil

cdef extern from "gsl/gsl_sf_gamma.h":
    double gsl_sf_lngamma(double x) nogil
    double gsl_sf_fact(unsigned int n) nogil

__all__ = ['value', 'acceleration', 'compute_coefficients']

cpdef _compute_helpers(double[::1] twoalpha, double[::1] dblfact,
                       double[:,::1] c1, double[:,::1] c2,
                       double[::1] c3, int nmax, int lmax):
    cdef int l, n

    dblfact[1] = 1.
    for l in range(2,lmax+1):
        dblfact[l] = dblfact[l-1] * (2.*l - 1.)

    for l in range(lmax+1):
        twoalpha[l] = 2.0*(2.*l + 1.5)

    for n in range(1,nmax+1):
        c3[n] = 1./(n+1)
        for l in range(lmax+1):
            c1[n,l] = 2.0*n + twoalpha[l]
            c2[n,l] = n-1.0 + twoalpha[l]

cpdef value(double[:,::1] xyz, double[::1] pot,
            double[:,:,::1] sin_coeff, double[:,:,::1] cos_coeff,
            int nmax, int lmax):

    cdef int n,l,m,i
    cdef:
        int norbits = xyz.shape[0]
        double phinltil,costh,un,xi,phi,r,unm1,plm1m,plm2m
        double clm,dlm,temp3

    # initialize empty arrays
    cdef:
        double[::1] cosmphi = np.zeros(lmax+1)
        double[::1] sinmphi = np.zeros(lmax+1)
        double[:,::1] ultrasp = np.zeros((nmax+1,lmax+1))
        double[:,::1] plm = np.zeros((lmax+1,lmax+1))

    # ----------------------------------------------------------------
    # This stuff was all in a "firstc" or "first calculation" check
    #   in Fortran. We may not need to compute this every time...but
    #   for now, just do it™
    cdef:
        double[::1] twoalpha = np.zeros(lmax+1)
        double[::1] dblfact = np.zeros(lmax+1)
        double[:,::1] c1 = np.zeros((nmax+1, lmax+1))
        double[:,::1] c2 = np.zeros((nmax+1,lmax+1))
        double[::1] c3 = np.zeros(nmax+1)
    _compute_helpers(twoalpha, dblfact, c1, c2, c3, nmax, lmax)
    # ----------------------------------------------------------------

    for i in range(norbits):
        r = sqrt(xyz[i,0]*xyz[i,0] + xyz[i,1]*xyz[i,1] + xyz[i,2]*xyz[i,2])
        costh = xyz[i,2]/r
        phi = atan2(xyz[i,1], xyz[i,0])
        xi = (r-1.)/(r+1.)

        for m in range(lmax+1):
            cosmphi[m] = cos(m*phi)
            sinmphi[m] = sin(m*phi)

        pot[i] = 0.
        for l in range(lmax+1):
            ultrasp[0,l] = 1.0
            ultrasp[1,l] = twoalpha[l]*xi

            un = ultrasp[1,l]
            unm1 = 1.0
            for n in range(1,nmax):
                ultrasp[n+1,l] = (c1[n,l]*xi*un - c2[n,l]*unm1) * c3[n]
                unm1 = un
                un = ultrasp[n+1,l]

        for m in range(lmax+1):
            plm[m,m] = 1.0
            if m > 0:
                plm[m,m] = (-1.)**m * dblfact[m] * sqrt(1.-costh*costh)**m

            plm1m = plm[m,m]
            plm2m = 0.0
            for l in range(m+1,lmax+1):
                plm[l,m] = (costh*(2.*l-1.)*plm1m-(l+m-1.)*plm2m)/(l-m)
                plm2m = plm1m
                plm1m = plm[l,m]

        for l in range(0,lmax+1):
            temp3 = 0.0
            for m in range(l+1):
                clm = 0.0
                dlm = 0.0
                for n in range(nmax+1):
                    clm += ultrasp[n,l] * cos_coeff[n,l,m]
                    dlm += ultrasp[n,l] * sin_coeff[n,l,m]

                temp3 += plm[l,m] * (clm*cosmphi[m] + dlm*sinmphi[m])

            phinltil = r**l / ((1.+r)**(2*l+1))
            pot[i] += -temp3*phinltil

cpdef acceleration(double[:,::1] xyz, double[:,::1] acc,
                   double[:,:,::1] sin_coeff, double[:,:,::1] cos_coeff,
                   int nmax, int lmax):

    cdef int n,l,m,i
    cdef:
        int norbits = xyz.shape[0]
        double ar,aphi,ath
        double phinltil,costh,un,xi,phi,r,unm1,plm1m,plm2m
        double clm,dlm,elm,flm
        double temp3,temp4,temp5,temp6

    norbits = xyz.shape[0]

    # initialize empty arrays
    cdef:
        double[::1] cosmphi = np.zeros(lmax+1)
        double[::1] sinmphi = np.zeros(lmax+1)
        double[:,::1] ultrasp = np.zeros((nmax+1,lmax+1))
        double[:,::1] ultrasp1 = np.zeros((nmax+1,lmax+1))
        double[:,::1] plm = np.zeros((lmax+1,lmax+1))
        double[:,::1] dplm = np.zeros((lmax+1,lmax+1))

    # ----------------------------------------------------------------
    # This stuff was all in a "firstc" or "first calculation" check
    #   in Fortran. We may not need to compute this every time...but
    #   for now, just do it™
    cdef:
        double[::1] twoalpha = np.zeros(lmax+1)
        double[::1] dblfact = np.zeros(lmax+1)
        double[:,::1] c1 = np.zeros((nmax+1,lmax+1))
        double[:,::1] c2 = np.zeros((nmax+1,lmax+1))
        double[::1] c3 = np.zeros(nmax+1)
    _compute_helpers(twoalpha, dblfact, c1, c2, c3, nmax, lmax)
    # ----------------------------------------------------------------

    for i in range(norbits):
        r = sqrt(xyz[i,0]*xyz[i,0] + xyz[i,1]*xyz[i,1] + xyz[i,2]*xyz[i,2])
        costh = xyz[i,2]/r
        phi = atan2(xyz[i,1], xyz[i,0])
        xi = (r-1.)/(r+1.)

        for m in range(lmax+1):
            cosmphi[m] = cos(m*phi)
            sinmphi[m] = sin(m*phi)

        ar = 0.
        ath = 0.
        aphi = 0.
        for l in range(lmax+1):
            ultrasp[0,l] = 1.0
            ultrasp[1,l] = twoalpha[l]*xi
            ultrasp1[0,l] = 0.0
            ultrasp1[1,l] = 1.0

            un = ultrasp[1,l]
            unm1 = 1.0
            for n in range(1,nmax):
                ultrasp[n+1,l] = (c1[n,l]*xi*un - c2[n,l]*unm1) * c3[n]
                unm1 = un
                un = ultrasp[n+1,l]
                ultrasp1[n+1,l] = ((twoalpha[l] + (n+1)-1.)*unm1-(n+1)*xi*ultrasp[n+1,l]) / \
                                   (twoalpha[l]*(1.-xi*xi))

        for m in range(lmax+1):
            plm[m,m] = 1.0
            if m > 0:
                plm[m,m] = (-1.)**m * dblfact[m] * sqrt(1.-costh*costh)**m

            plm1m = plm[m,m]
            plm2m = 0.0

            for l in range(m+1,lmax+1):
                plm[l,m] = (costh*(2.*l-1.)*plm1m-(l+m-1.)*plm2m)/(l-m)
                plm2m = plm1m
                plm1m = plm[l,m]

        dplm[0,0] = 0.0
        for l in range(1,lmax+1):
            for m in range(0,l+1):
                if l == m:
                    dplm[l,m] = l*costh*plm[l,m] / (costh*costh-1.0)
                else:
                    dplm[l,m] = ((l*costh*plm[l,m]-(l+m)*plm[l-1,m]) /
                                 (costh*costh-1.0))

        for l in range(0,lmax+1):
            temp3 = 0.0
            temp4 = 0.0
            temp5 = 0.0
            temp6 = 0.0

            for m in range(l+1):
                clm = 0.0
                dlm = 0.0
                elm = 0.0
                flm = 0.0

                for n in range(nmax+1):
                    clm += ultrasp[n,l] * cos_coeff[n,l,m]
                    dlm += ultrasp[n,l] * sin_coeff[n,l,m]
                    elm += ultrasp1[n,l] * cos_coeff[n,l,m]
                    flm += ultrasp1[n,l] * sin_coeff[n,l,m]

                temp3 += plm[l,m] * (clm*cosmphi[m] + dlm*sinmphi[m])
                temp4 -= plm[l,m] * (elm*cosmphi[m] + flm*sinmphi[m])
                temp5 -= dplm[l,m] * (clm*cosmphi[m] + dlm*sinmphi[m])
                temp6 -= m*plm[l,m] * (dlm*cosmphi[m] - clm*sinmphi[m])

            phinltil = r**l / ((1.+r)**(2*l+1))
            ar += phinltil*(-temp3*(l/r-(2.*l+1.)/(1.+r)) + \
                            temp4*4.*(2.*l+1.5)/(1.+r)**2)
            ath += temp5*phinltil
            aphi += temp6*phinltil

        cosp = cos(phi)
        sinp = sin(phi)

        sinth = sqrt(1.-costh*costh)
        ath = -sinth*ath/r
        aphi = aphi/(r*sinth)
        acc[i,0] = -(sinth*cosp*ar + costh*cosp*ath - sinp*aphi)
        acc[i,1] = -(sinth*sinp*ar + costh*sinp*ath + cosp*aphi)
        acc[i,2] = -(costh*ar - sinth*ath)

cpdef compute_coefficients(double[:,::1] xyz, double[::1] mass,
                           double[:,:,::1] sin_coeff, double[:,:,::1] cos_coeff,
                           int nmax, int lmax):

    cdef int n,l,m,k
    cdef:
        int norbits = xyz.shape[0]
        double costh,un,xi,phi,r,unm1,plm1m,plm2m
        double clm,dlm
        double temp3,temp4,temp5,ttemp5

    # initialize empty arrays
    cdef:
        double[::1] cosmphi = np.zeros(lmax+1)
        double[::1] sinmphi = np.zeros(lmax+1)
        double[:,::1] ultrasp = np.zeros((nmax+1,lmax+1))
        double[:,::1] ultrasp1 = np.zeros((nmax+1,lmax+1))
        double[:,::1] ultraspt = np.zeros((nmax+1,lmax+1))
        double[:,::1] anltilde = np.zeros((nmax+1,lmax+1))
        double[:,::1] plm = np.zeros((lmax+1,lmax+1))
        double[:,::1] coeflm = np.zeros((lmax+1,lmax+1))

    # ----------------------------------------------------------------
    # This stuff was all in a "firstc" or "first calculation" check
    #   in Fortran. We may not need to compute this every time...but
    #   for now, just do it™
    cdef:
        double[::1] twoalpha = np.zeros(lmax+1)
        double[::1] dblfact = np.zeros(lmax+1)
        double[:,::1] c1 = np.zeros((nmax, lmax+1))
        double[:,::1] c2 = np.zeros((nmax,lmax+1))
        double[::1] c3 = np.zeros(nmax)
    _compute_helpers(twoalpha, dblfact, c1, c2, c3, nmax, lmax)
    # ----------------------------------------------------------------

    for n in range(nmax+1):
        for l in range(lmax+1):
            knl = 0.5*n*(n + 4.*l + 3.) + (l + 1.)*(2.*l + 1.)
            anltilde[n,l] = -2.**(8.*l+6.)*gsl_sf_fact(n)*(n+2.*l+1.5)
            arggam = 2.*l + 1.5
            anltilde[n,l] *= (exp(gsl_sf_lngamma(arggam)))**2
            anltilde[n,l] /= (4.*M_PI*knl*gsl_sf_fact(n+4*l+2))

    for l in range(lmax+1):
        twoalpha[l] = 2.*(2.*l + 1.5)

        for m in range(0,l+1):
            deltam0 = 1. if m == 0 else 2.
            coeflm[l,m] = (2.*l+1.) * deltam0 * gsl_sf_fact(l-m)/gsl_sf_fact(l+m)

    for k in range(norbits):
        r = sqrt(xyz[k,0]*xyz[k,0] + xyz[k,1]*xyz[k,1] + xyz[k,2]*xyz[k,2])
        costh = xyz[k,2]/r
        phi = atan2(xyz[k,1], xyz[k,0])
        xi = (r-1.)/(r+1.)

        for m in range(lmax+1):
            cosmphi[m] = cos(m*phi)
            sinmphi[m] = sin(m*phi)

        ar = 0.
        ath = 0.
        aphi = 0.
        for l in range(lmax+1):
            ultrasp[0,l] = 1.0
            ultrasp[1,l] = twoalpha[l]*xi

            un = ultrasp[1,l]
            unm1 = 1.0
            for n in range(1,nmax):
                ultrasp[n+1,l] = (c1[n-1,l]*xi*un - c2[n-1,l]*unm1) * c3[n-1]
                unm1 = un
                un = ultrasp[n+1,l]

            for n in range(nmax+1):
               ultraspt[n,l] = ultrasp[n,l]*anltilde[n,l]

        for m in range(lmax+1):
            plm[m,m] = 1.0
            if m > 0:
                plm[m,m] = (-1.)**m * dblfact[m] * sqrt(1.-costh*costh)**m

            plm1m = plm[m,m]
            plm2m = 0.0

            for l in range(m+1,lmax+1):
                plm[l,m] = (costh*(2.*l-1.)*plm1m - (l+m-1.)*plm2m) / (l-m)
                plm2m = plm1m
                plm1m = plm[l,m]

        for l in range(lmax+1):
            temp5 = r**l / ((1.+r)**(2*l+1)) * mass[k]
            for m in range(0,l+1):
                ttemp5 = temp5 * plm[l,m] * coeflm[l,m]
                temp3 = ttemp5 * sinmphi[m]
                temp4 = ttemp5 * cosmphi[m]

                for n in range(nmax+1):
                    sin_coeff[n,l,m] += temp3*ultraspt[n,l]
                    cos_coeff[n,l,m] += temp4*ultraspt[n,l]
