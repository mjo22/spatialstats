"""
Routines to calculate the radial distribution function :math:`g(r)`
and isotropic structure factor :math:`S(q)`.

See `here <https://en.wikipedia.org/wiki/Radial_distribution_function>`_
to learn more.

Adapted from https://github.com/wenyan4work/point_cloud.

.. moduleauthor:: Michael O'Brien <michaelobrien@g.harvard.edu>

"""

import numpy as np
import scipy.spatial as ss
import numba as nb
from scipy.integrate import simps
from scipy.special import jv
from time import time


def structure_factor(gr, r, N, boxsize, q=None, **kwargs):
    """
    Calculate the isotropic structure factor :math:`S(q)` from
    the radial distribution function :math:`g(r)` of a set of :math:`N` 
    particle positions in a 2D or 3D periodic box with volume :math:`V`.

    The structure factor in 3D is computed as

    .. math::

        S(q) = 1 + 4\\pi \\rho \\frac{1}{q} \int dr \ r \ \\textrm{sin}(qr) [g(r) - 1]

    and in 2D

    .. math::

        S(q) = 1 + 2\\pi \\rho \int dr \ r \ J_{0}(qr) [g(r) - 1],

    where :math:`\\rho = N/V` and :math:`J_{0}` is the 0th bessel function of the first kind.

    Parameters
    ----------
    gr : `np.ndarray`
        The radial distribution function :math:`g(r)`
        from :ref:`spatialstats.points.rdf<rdf>`.
    r : `np.ndarray`
        The domain of :math:`g(r)`
        from :ref:`spatialstats.points.rdf<rdf>`.
    N : `int`
        The number of particles :math:`N`.
    boxsize : `list` of `float`
        The rectangular domain over which
        to apply periodic boundary conditions.
        See ``scipy.spatial.cKDTree``.
    q : `np.ndarray`, optional
        Dimensional wavenumber bins :math:`q`.
        If ``None``, ``q = np.arange(dq, 200*dq, dq)``
        with ``dq = 2*np.pi / max(boxsize)``.

    Returns
    -------
    Sq : `np.ndarray`
        The structure factor :math:`S(q)`.
    q : `np.ndarray`
        Wavenumber bins :math:`q`.
    """
    ndim = len(boxsize)

    if ndim not in [2, 3]:
        raise ValueError("Dimension of space must be 2 or 3")

    # Generate wavenumbers
    if q is None:
        dq = (2*np.pi / max(boxsize))
        q = np.arange(dq, 200*dq, dq)

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
    .. _rdf:

    Calculate the radial distribution function :math:`g(r)`
    for a set of :math:`N` particle positions in a 2D or 3D periodic box.

    Parameters
    ---------
    points : `np.ndarray`, shape `(N, ndim)`
        Particle positions in ``ndim`` dimensions
        and ``N`` particles. Passed to
        ``scipy.spatial.cKDTree``.
    boxsize : `list` of `float`
        The rectangular domain over which
        to apply periodic boundary conditions.
        Passed to ``scipy.spatial.cKDTree``.
    rmin : `float`, optional
        Minimum :math:`r` value in :math:`g(r)`.
    rmax : `float`, optional
        Cutoff radius for KDTree search and
        maximum :math:`r` value in :math:`g(r)`.
        Default is maximum distance between any pair of
        particles.
    npts : `int`, optional
        Number points in domain :math:`r`.
    bench : `bool`, optional
        Print message for time of calculation.
    Returns
    -------
    gr : `np.ndarray`
        Radial distribution function :math:`g(r)`.
    r : `np.ndarray`
        Radius :math:`r`.
    """
    N, ndim = points.shape
    rmax = min(boxsize)/2 if rmax is None else rmax
    boxsize = np.array(boxsize)

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


@nb.njit(parallel=True, cache=True)
def _get_displacements(coords, pairs, boxsize, rmax):
    '''Get displacements between pairs'''
    npairs = pairs.shape[0]
    rvec = np.zeros((npairs, coords.shape[1]))
    for index in nb.prange(npairs):
        pair = pairs[index]
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
    # Get unique pairs (i<j)
    temp = np.array(list(tree.query_pairs(r=rmax)), dtype=np.int)
    # Get rest of the pairs (i>=j)
    npairs = len(temp)
    pairs = np.zeros(shape=(2*npairs, 2), dtype=np.int)
    pairs[:npairs, :] = temp
    pairs[npairs:, :] = temp[:, [1, 0]]
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


if __name__ == "__main__":

    from matplotlib import pyplot as plt

    N = 2000
    boxsize = [100, 100]
    data = np.random.rand(N, 2)*100
    rmax = boxsize[0]/4

    gr, r = rdf(data, boxsize, rmax=rmax, npts=200)

    Sq, q = structure_factor(gr, r, N, boxsize, qmin=0, qmax=100, dq=.5)

    fig, axes = plt.subplots(ncols=2)
    axes[0].plot(r, gr)
    axes[0].set_xlabel("$r$")
    axes[0].set_ylabel("$g(r)$")

    axes[1].plot(q, Sq)
    axes[1].set_xlabel("$q$")
    axes[1].set_ylabel("$S(q)$")

    plt.show()
