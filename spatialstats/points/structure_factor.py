"""
.. moduleauthor:: Michael O'Brien <michaelobrien@g.harvard.edu>
.. moduleauthor:: Wen Yan

"""

import numpy as np
from scipy.integrate import simps
from scipy.special import jv
from .point_cloud import rdf


def structure_factor(points, boxsize, qmin=None, qmax=None, dq=None, **kwargs):
    """
    Calculate the isotropic static structure factor from
    the radial distribution function of a set of particles.

    kwargs passed to spatialstats.points.point_cloud.rdf.

    Parameters
    ---------
    points : np.ndarray, shape (ndim, N)
        Particle locations, where ndim is number
        of dimensions and N is number of particles.
    boxsize : float or list of floats
        Size of the rectangular domain over which
        to apply periodic boundary conditions.
        See scipy.spatial.cKDTree documentation.
    qmin : float, optional
        Minimum wavenumber for Sq.
    qmax : float, optional
        Maximum wavenumber for Sq.
    dq : float, optional
        Wavenumber step size.

    Returns
    -------
    Sq : np.ndarray
        The static structure factor.
    q : np.ndarray
        Dimensional wavenumbers discretized by 2pi/L, where
        L is the maximum dimension of boxsize.
    gr : np.ndarray
        Radial distribution function g(r).
        See spatialstats.points.point_cloud.rdf.
    r : np.ndarray
        Radius r.
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

    return Sq, q, gr, r
