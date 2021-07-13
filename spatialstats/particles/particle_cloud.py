"""
Routines to calculate spatial distribution functions for point
and rod-like particles. For point-like particles, can reduce to
the usual :math:`g(r)` and isotropic structure factor :math:`S(q)`.

See `here <https://en.wikipedia.org/wiki/Radial_distribution_function>`
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
        The radial distribution function :math:`g(r)` from
        :ref:`spatialstats.particles.sdf<sdf>`.
    r : `np.ndarray`
        The domain of :math:`g(r)` from
        :ref:`spatialstats.particles.sdf<sdf>`.
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


def sdf(positions, boxsize, orientations=None,
        rmin=None, rmax=None, nr=100, nphi=100, bench=False):
    """
    .. _sdf:

    Calculate the spatial distribution function :math:`g(r, \\phi)`
    for a set of :math:`N` point-like or rod-like particles
    :math:`\\mathbf{r}_i` in a 2D or 3D periodic box. :math:`r` and
    :math:`\\phi` are defined as the usual spherical coordinates
    for displacement vectors between particle pairs
    :math:`\\mathbf{r}_{ij} = \\mathbf{r}_i - \\mathbf{r}_j`.

    If particles orientations :math:`\\mathbf{p}_i` are included,
    instead define :math:`\\phi` as the angle in the
    coordinate system with :math:`\\mathbf{p}_i` pointed in the
    :math:`+z` direction.

    .. note::
        Reduces to the 1D distribution function :math:`g(r)`
        when ``nphi = None``.

    Parameters
    ---------
    positions : `np.ndarray`, shape `(N, ndim)`
        Particle positions :math:`\\mathbf{r}_i`
        in ``ndim`` dimensions for ``N`` particles.
        Passed to ``scipy.spatial.cKDTree``.
    boxsize : `list` of `float`
        The rectangular domain over which
        to apply periodic boundary conditions.
        Passed to ``scipy.spatial.cKDTree``.
    orientations : `np.ndarray`, shape `(N, ndim)`
        Particle orientation vectors :math:`\\mathbf{p}_i`.
        Vectors should be unitary, but they will be
        normalized automatically.
    rmin : `float`, optional
        Minimum :math:`r` value in :math:`g(r, \\phi)`.
    rmax : `float`, optional
        Cutoff radius for KDTree search and
        maximum :math:`r` value in :math:`g(r, \\phi)`.
        Default is maximum distance between any pair of
        particles.
    nr : `int`, optional
        Number of points to bin in :math:`r`.
    nphi : `int`, optional
        Number of points to bin in :math:`\\phi`
    bench : `bool`, optional
        Print message for time of calculation.
    Returns
    -------
    g : `np.ndarray`, shape `(nr,)` or `(nr, nphi)`
        Radial distribution function :math:`g(r)`
        or :math:`g(r, \\phi)`.
    r : `np.ndarray`, shape `(nr,)`
        Left edges of radial bins :math:`r`.
    phi : `np.ndarray`, shape `(nphi,)`, optional
        Left edges of angular bins :math:`\\phi`
    """
    N, ndim = positions.shape
    boxsize = np.array(boxsize)

    if ndim not in [2, 3]:
        raise ValueError("Dimension of space must be 2 or 3")

    if orientations is not None:
        if orientations.shape != (N, ndim):
            msg = f"Shape of orientations array must match positions array {(N, ndim)}"
            raise ValueError(msg)
    else:
        orientations = np.array([])

    # Binning keyword args
    rmin = 0 if rmin is None else rmin
    rmax = max(boxsize)/2 if rmax is None else rmax
    nphi = 1 if nphi is None else nphi

    if bench:
        t0 = time()

    # Periodic boundary conditions
    _impose_pbc(positions, boxsize)

    # Get particle pairs and their displacement vectors
    pairs = _get_pairs(positions, boxsize, rmax)
    r, phi = _get_displacements(positions, orientations,
                                pairs, boxsize, rmax, nphi)

    # Get g(r, phi)
    r_n = np.linspace(rmin, rmax, nr+1)
    phi_m = np.pi*np.linspace(0, 1, nphi+1)
    g = _get_distribution(r, phi, N, boxsize, r_n, phi_m)

    del r, phi

    if bench:
        print(f"Time: {time() - t0:.04f} s")

    return g, r_n[:-1], phi_m[:-1]


def _get_distribution(r, phi, N, boxsize, r_n, phi_m):
    '''Generate spatial distribution function'''
    ndim = boxsize.size
    density = N/(np.prod(boxsize))
    if phi.size == r.size:
        # Get g(r, phi)
        count, rbins, pbins = np.histogram2d(r, phi, bins=[r_n, phi_m])
        fac = (phi_m[1] - phi_m[0]) / np.pi
    else:
        # Get g(r)
        count, rbins = np.histogram(r, r_n)
        fac = 1
    # Scale with bin volume and density
    vol = np.zeros(count.shape)
    if ndim == 2:    # area = pi(r1^2-r0^2)
        for i in range(r_n.size-1):
            vol[i] = fac*np.pi*(rbins[i+1]**2-rbins[i]**2)
    elif ndim == 3:  # area = 4pi/3(r1^3-r0^3)
        for i in range(r_n.size-1):
            vol[i] = fac*(4.0/3.0)*np.pi*(rbins[i+1]**3-rbins[i]**3)
    g = count/(N*vol*density)
    return g


@nb.njit(parallel=True, cache=True)
def _get_displacements(r, p, pairs, boxsize, rmax, nphi):
    '''Get displacements between pairs'''
    rotate = True if nphi > 1 else False
    ndim = r.shape[-1]
    npairs = pairs.shape[0]
    rnorm = np.zeros(npairs)
    phi = np.zeros(npairs) if nphi > 1 else np.array([0.])
    for index in nb.prange(npairs):
        pair = pairs[index]
        i, j = pair
        r_i, r_j = r[i], r[j]
        r_ij = r_j - r_i
        if np.linalg.norm(r_ij) >= rmax:
            # Fix periodic image
            image = _closest_image(r_i, r_j, boxsize)
            r_ij = image - r_i
        if rotate:
            # Rotate particle head to +z direction
            p_i = p[i] / np.linalg.norm(p[i])
            angle = np.arccos(p_i[-1])
            c, s = np.cos(angle), np.sin(angle)
            if ndim == 2:
                R = np.array(((c, -s), (s, c)))
            else:
                R = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
            r_ij = R @ r_ij
        rnorm[index] = np.linalg.norm(r_ij)
        if nphi > 1:
            phi[index] = np.arctan2(r_ij[1], r_ij[0])
    return rnorm, phi


def _get_pairs(coords, boxsize, rmax):
    '''Get coordinate pairs within distance rmax'''
    tree = ss.cKDTree(coords, boxsize=boxsize)
    # Get unique pairs (i<j)
    temp = tree.query_pairs(r=rmax, output_type="ndarray")
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
def _closest_point(target, positions):
    '''Get closest positions to target in 2D and 3D'''
    target = np.array(target)
    positions = np.array(positions)
    distance = []
    for p in positions:
        distance.append(np.linalg.norm(p-target))
    distance = np.array(distance)
    ind = np.argmin(distance)
    return positions[ind], ind


@nb.njit(cache=True)
def _closest_point1d(target, positions):
    '''Get closest positions to target in 1D'''
    distance = []
    for p in positions:
        distance.append(np.abs(p-target))
    distance = np.array(distance)
    ind = np.argmin(distance)
    return positions[ind], ind


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
    pos = np.random.rand(N, 2)*100
    orient = np.ones((N, 2))
    rmax = boxsize[0]/8

    g, r, phi = sdf(pos, boxsize, rmax=rmax,
                    orientations=orient,
                    nr=150, nphi=None)

    if g.ndim == 1:
        S, q = structure_factor(g, r, N, boxsize, qmin=0, qmax=100, dq=.5)
        fig, axes = plt.subplots(ncols=2)
        axes[0].plot(r, g)
        axes[0].set_xlabel("$r$")
        axes[0].set_ylabel("$g(r)$")
        axes[1].plot(q, S)
        axes[1].set_xlabel("$q$")
        axes[1].set_ylabel("$S(q)$")
        plt.show()
    else:
        fig, ax = plt.subplots()
        ax.imshow(g, origin="lower")
        plt.show()
