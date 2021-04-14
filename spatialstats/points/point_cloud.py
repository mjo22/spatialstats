"""
Routines to calculate the radial distribution function g(r)
and isotropic structure factor S(q).

See https://en.wikipedia.org/wiki/Radial_distribution_function
to learn more.

.. moduleauthor:: Michael O'Brien <michaelobrien@g.harvard.edu>
.. moduleauthor:: Wen Yan

"""

import numpy as np
import scipy.spatial as ss
import numba as nb
from scipy.integrate import simps
from scipy.special import jv
from time import time


def structure_factor(gr, r, N, boxsize,
                     qmin=None, qmax=None, dq=None, **kwargs):
    """
    Calculate the isotropic structure factor S(q) from
    the radial distribution function g(r) of a set of particles.

    Parameters
    ----------
    gr : `np.ndarray`
        The radial distribution function outputted
        from `spatialstats.points.point_cloud.rdf`.
    r : `np.ndarray`
        The domain of the radial distribution function
        outputted from `spatialstats.points.point_cloud.rdf`.
    N : `int`
        The number of particles.
    boxsize : `list` of `float`
        The rectangular domain over which
        to apply periodic boundary conditions.
        See `scipy.spatial.cKDTree` documentation.
    qmin : `float`, optional
        Minimum wavenumber for S(q). Default is ``dq``.
    qmax : `float`, optional
        Maximum wavenumber for S(q). Default is ``100*dq``
    dq : `float`, optional
        Wavenumber step size. Default is ``2*np.pi/L``, where
        ``L = max(boxsize)``.

    Returns
    -------
    Sq : `np.ndarray`
        The static structure factor.
    q : `np.ndarray`
        Dimensional wavenumbers discretized by ``2*np.pi/L``, where
        ``L = max(boxsize)``.
    """
    ndim = len(boxsize)

    if ndim not in [2, 3]:
        raise ValueError("Dimension of space must be 2 or 3")

    # Generate wavenumbers
    dq = (2*np.pi / max(boxsize)) if dq is None else dq
    qmin = dq if qmin is None else qmin
    qmax = 100*dq if qmax is None else qmax
    q = np.arange(qmin, qmax+dq, dq)

    def S(q):
        '''Integrand for isotropic structure factor'''
        rho = N/np.prod(boxsize)
        if ndim == 3:
            f = np.sin(q*r)*r*(gr-1)
            return 1+4*np.pi*rho*simps(f, r)/q
        else:
            f = jv(0, q*r)*r*(gr-1)
            return 1+2*np.pi*rho*simps(f, r)

    # Integrate for all q
    Sq = []
    for j in range(len(q)):
        Sq.append(S(q[j]))
    Sq = np.array(Sq)

    return Sq, q


def rdf(points, boxsize, rmin=None, rmax=None, npts=100, bench=False):
    """
    Calculate the radial distribution function g(r)
    for a group of particles in 2D or 3D.

    Works for periodic boundary conditions on rectangular
    domains.

    Parameters
    ---------
    points : `np.ndarray`, shape `(ndim, N)`
        Particle locations, where ndim is number
        of dimensions and N is number of particles.
    boxsize : `list` of `float`
        The rectangular domain over which
        to apply periodic boundary conditions.
        See `scipy.spatial.cKDTree` documentation.
    rmin : `float`, optional
        Minimum r value in g(r).
    rmax : `float`, optional
        Cutoff radius for KDTree search and
        maximum r value in g(r). Default is
        maximum distance between any pair of
        particles.
    npts : `int`, optional
        Number points between [`rmin`, `rmax`] in
        g(r).
    bench : `bool`, optional
        Print message for time of calculation.
    Returns
    -------
    gr : `np.ndarray`
        Radial distribution function g(r).
    r : `np.ndarray`
        Radius r.
    """
    ndim, N = points.shape
    points = points.T
    rmax = min(boxsize)/2 if rmax is None else rmax

    if ndim not in [2, 3]:
        raise ValueError("Dimension of space must be 2 or 3")

    if bench:
        t0 = time()

    # Periodic boundary conditions
    _impose_pbc(points, boxsize)

    # Get point pairs and their displacement vectors
    pairs = _get_pairs(points, boxsize, rmax)
    rjk = _get_displacements(points, pairs, boxsize, rmax)

    # Get g(r)
    r, gr = _gen_rdf(rjk, N, N/(np.prod(boxsize)),
                     rmin, rmax, npts)

    if bench:
        print(f"Time: {time() - t0:.04f} s")

    return gr, r


def _gen_rdf(rvec, npar, density, rmin, rmax, nbins):
    '''Generate radial distribution function'''
    ndim = rvec.shape[1]
    rnorm = np.linalg.norm(rvec, axis=1)
    rmin = 0 if rmin is None else rmin
    rmax = np.max(rnorm) if rmax is None else rmax
    count, bins = np.histogram(rnorm, np.linspace(rmin, rmax, nbins))
    # Scale with vol and density
    vol = np.zeros(count.shape)
    if ndim == 2:    # area = pi(r1^2-r0^2)
        for i in range(nbins-1):
            vol[i] = np.pi*(bins[i+1]**2-bins[i]**2)
    elif ndim == 3:  # area = 4pi/3(r1^3-r0^3)
        for i in range(nbins-1):
            vol[i] = (4.0/3.0)*np.pi*(bins[i+1]**3-bins[i]**3)
    rdf = count/(npar*vol*density)
    r = 0.5*(bins[:-1]+bins[1:])
    return r, rdf


@nb.njit(cache=True)
def _get_displacements(coords, pairs, boxsize, rmax):
    '''Get displacements between pairs'''
    npairs, itpairs = len(pairs), iter(pairs)
    rvec = np.zeros((npairs, coords.shape[1]))
    for index in range(npairs):
        pair = next(itpairs)
        id0, id1 = pair
        pos0, pos1 = coords[id0], coords[id1]
        vec01 = pos1-pos0
        if np.linalg.norm(vec01) < rmax:
            rvec[index] = vec01
        else:  # fix periodic image
            image = _closest_image(pos0, pos1, boxsize)
            rvec[index] = image-pos0
    return rvec


def _get_pairs(coords, boxsize, rmax):
    '''Get coordinate pairs within distance rmax'''
    tree = ss.cKDTree(coords, boxsize=boxsize)
    boxsize = np.array(boxsize)
    pairs = tree.query_pairs(r=rmax)  # this returns only pairs (i<j)
    pairs2 = set()
    for p in pairs:
        pairs2.add((p[1], p[0]))
    pairs.update(pairs2)
    return pairs


@nb.njit(cache=True)
def _impose_pbc(coords, boxsize):
    '''Impose periodic boundary conditions for KDTree'''
    dim = len(boxsize)
    for j in range(len(coords)):
        p = coords[j]
        for i in range(dim):
            while p[i] < 0:
                p[i] = p[i]+boxsize[i]
            while p[i] > boxsize[i]:
                p[i] = p[i]-boxsize[i]


@nb.njit(cache=True)
def _closest_point(target, points):
    '''Get closest points to target in 2D and 3D'''
    target = np.array(target)
    points = np.array(points)
    distance = []
    for p in points:
        distance.append(np.linalg.norm(p-target))
    distance = np.array(distance)
    ind = np.argmin(distance)
    return points[ind], ind


@nb.njit(cache=True)
def _closest_point1d(target, points):
    '''Get closest points to target in 1D'''
    distance = []
    for p in points:
        distance.append(np.abs(p-target))
    distance = np.array(distance)
    ind = np.argmin(distance)
    return points[ind], ind


@nb.njit(cache=True)
def _closest_image(target, source, boxsize):
    '''Get closest periodic image to target'''
    dim = target.shape[0]
    assert source.shape[0] == dim
    image = np.zeros(dim)
    for i in range(dim):
        pts = [source[i], source[i]-boxsize[i], source[i]+boxsize[i]]
        pos, ind = _closest_point1d(target[i], pts)
        image[i] = pos
    return image
