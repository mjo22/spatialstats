"""
Routines to calculate the static structure factor
for a set of particles.
"""

import numpy as np
from scipy.integrate import simps
from .point_cloud import rdf


def structure_factor(points, boxsize, return_rdf=False,
                     qmin=None, qmax=None, npts=None, **kwargs):
    """
    Calculate the isotropic static structure factor from
    a set of particles radial distribution function.

    Arguments
    ---------
    points : np.ndarray [ndim, N]
        Particle locations, where ndim is number
        of dimensions and N is number of particles.
    q : np.ndarray
        Wavenumbers over which to calculate Sq.
        This should be dimensional.
    boxsize : float or list of floats
        Size of the rectangular domain over which
        to apply periodic boundary conditions.
        See scipy.spatial.cKDTree documentation.

    Keywords
    --------
    return_rdf : bool
        Return the rdf used to calculate structure factor.

    **kwargs passed to rdf.

    Returns
    -------
    Sq : np.ndarray
        The static structure factor
    q : np.ndarray
        Dimensional wavenumbers
    """
    ndim, N = points.shape
    boxsize = boxsize if type(boxsize) is list else ndim*[boxsize]

    # Generate wavenumbers
    qmin = 1./max(boxsize) if qmin is None else qmin
    qmax = 100./max(boxsize) if qmax is None else qmax
    npts = qmax-qmin+1 if npts is None else npts
    q = np.linspace(qmin, qmax, npts)

    # Calculate g(r) and density for integration
    gr, r = rdf(points, boxsize, **kwargs)
    rho = N/np.prod(boxsize)

    def S(q):
        '''
        Integrand for isotropic structure factor
        
        See https://en.wikipedia.org/wiki/Radial_distribution_function.
        '''
        f = np.sin(2*np.pi*q*r)*r*(gr-1)
        return 1+2**(ndim-1)*np.pi*rho*simps(f, r)/(2*np.pi*q)

    # Integrate for all q
    Sq = []
    for j in range(len(q)):
        Sq.append(S(q[j]))
    Sq = np.array(Sq)

    args = (Sq, q) if not return_rdf else (Sq, q, gr, r)

    return args
