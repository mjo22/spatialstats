"""
Routines to calculate the radial distribution function
for a set of particles.

Authors:
    Wen Yan and Michael O'Brien (2021)
    Biophysical Modeling Group
    Center for Computational Biology
    Flatiron Institute
"""

import numpy as np
import scipy.spatial as ss
import numba as nb
from time import time


def rdf(points, boxsize, rmax=None, bench=False, **kwargs):
    """
    Calculate the radial distribution function
    for a group of particles in 2D or 3D.

    Works for periodic boundary conditions on rectangular
    domains.

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
    rmax : float
        Cutoff radius for KDTree search

    **kwargs passed to gen_rdf.

    Returns
    -------
    gr : np.ndarray
        Radial distribution function g(r)
    r : np.ndarray
        Radius r
    """
    ndim, N = points.shape
    points = points.T
    boxsize = boxsize if type(boxsize) is list else ndim*[boxsize]
    rmax = min(boxsize)/2 if rmax is None else rmax

    if ndim not in [2, 3]:
        raise ValueError("Dimension of space must be 2 or 3")

    if bench:
        t0 = time()

    # Periodic boundary conditions
    impose_pbc(points, boxsize)

    # Get point pairs and their displacement vectors
    pairs = get_pairs(points, boxsize, rmax)
    rjk = get_displacements(points, pairs, boxsize, rmax)

    # Get g(r)
    r, gr = gen_rdf(rjk, N, N/(np.prod(boxsize)),
                    rmax=rmax, **kwargs)

    if bench:
        print(f"Time: {time() - t0:.04f} s")

    return gr, r


def gen_rdf(rvec, npar, density, rmin=None, rmax=None, nbins=100):
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
def get_displacements(coords, pairs, boxsize, rmax):
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
            image = closest_image(pos0, pos1, boxsize)
            rvec[index] = image-pos0
    return rvec


def get_pairs(coords, boxsize, rmax):
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
def impose_pbc(coords, boxsize):
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
def closest_point(target, points):
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
def closest_point1d(target, points):
    '''Get closest points to target in 1D'''
    distance = []
    for p in points:
        distance.append(np.abs(p-target))
    distance = np.array(distance)
    ind = np.argmin(distance)
    return points[ind], ind


@nb.njit(cache=True)
def closest_image(target, source, boxsize):
    '''Get closest periodic image to target'''
    dim = target.shape[0]
    assert source.shape[0] == dim
    image = np.zeros(dim)
    for i in range(dim):
        pts = [source[i], source[i]-boxsize[i], source[i]+boxsize[i]]
        pos, ind = closest_point1d(target[i], pts)
        image[i] = pos
    return image
