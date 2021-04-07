"""
Routines to calculate the static structure factor
for a set of particles.

Authors:
    Wen Yan and Michael O'Brien (2021)
    Biophysical Modeling Group
    Center for Computational Biology
    Flatiron Institute
"""

import numpy as np
from scipy.integrate import simps
from scipy.special import jv
from .point_cloud import rdf
from time import time


def structure_factor(points, boxsize, return_rdf=False,
                     qmin=None, qmax=None, dq=None, bench=False, **kwargs):
    """
    Calculate the isotropic static structure factor from
    a set of particles radial distribution function.

    Arguments
    ---------
    points : np.ndarray [ndim, N]
        Particle locations, where ndim is number
        of dimensions and N is number of particles.
    boxsize : float or list of floats
        Size of the rectangular domain over which
        to apply periodic boundary conditions.
        See scipy.spatial.cKDTree documentation.

    Keywords
    --------
    return_rdf : bool
        Return the rdf used to calculate structure factor.
    qmin : float
        Minimum wavenumber for Sq
    qmax : float
        Maximum wavenumber for Sq
    dq : float
        Wavenumber step size

    **kwargs passed to rdf.

    Returns
    -------
    Sq : np.ndarray
        The static structure factor
    q : np.ndarray
        Dimensional wavenumbers discretized by 2pi/L, where
        L is the maximum dimension of boxsize
    """
    ndim, N = points.shape
    boxsize = boxsize if type(boxsize) is list else ndim*[boxsize]

    if ndim not in [2, 3]:
        raise ValueError("Dimension of space must be 2 or 3")

    # Generate wavenumbers
    dq = (2*np.pi / max(boxsize)) if dq is None else dq
    qmin = dq if qmin is None else qmin
    qmax = 100*dq if qmax is None else qmax
    q = np.arange(qmin, qmax+dq, dq)

    if bench:
        t0 = time()

    # Calculate g(r) and density for integration
    gr, r = rdf(points, boxsize, **kwargs)
    rho = N/np.prod(boxsize)

    def S(q):
        '''
        Integrand for isotropic structure factor

        See https://en.wikipedia.org/wiki/Radial_distribution_function.
        '''
        if ndim == 3:
            f = np.sin(q*r)*r*(gr-1)
            return 1+4*np.pi*rho*simps(f, r)/(q)
        else:
            f = jv(0, q*r)*r*(gr-1)
            return 1+2*np.pi*rho*simps(f, r)

    # Integrate for all q
    Sq = []
    for j in range(len(q)):
        Sq.append(S(q[j]))
    Sq = np.array(Sq)

    if bench:
        print(f"Time: {time() - t0:.04f} s")

    args = (Sq, q) if not return_rdf else (Sq, q, gr, r)

    return args
