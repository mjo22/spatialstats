"""
Routines for averaging 2D and 3D images
in polar and spherical coordinates
"""

import numpy as np


def uniqueavg(data, coords):
    """
    Average data by choosing bins as all unique
    values in coordinate system

    Parameters
    ----------
    data : np.ndarray
        Data to average
    coords : np.ndarray
        Coordinate system to average over. Must
        have same shape as data

    Returns
    -------
    avg : np.ndarray
        Averaged data over all unique values in coords
    bins : np.ndarray
        All unique values in coords
    """
    bins = np.append(np.unique(coords.flat), coords.max())
    count, bins = np.histogram(coords.flat, bins)
    weight, bins = np.histogram(coords.flat, bins, weights=data.flat)
    avg = weight / count
    bins = 0.5 * (bins[:-1] + bins[1:])

    return avg, bins


def radialavg(data, center=None, dphi=None,
              bounds=(0, 2*np.pi), portion=False,
              stdev=False, avg=True, weight=None):
    """Avg data over bins of radial contours

    Adapted from aziavg.py in https://github.com/davidgrier/dgmath/

    Arguments
    ---------
    data : ndarray
        two-dimensional data array.

    Keywords
    --------
    center : [xc, yc]
        Coordinates of the point around which to compute average
        Default: geometric center
    dphi : float
        Angular division of bins on which to group angles when
        averaging.
    bounds : tuple int
        angle range of average [radians]
        Default: (0, 2*pi)
    portion : bool
        Choose whether to scale data according to position in bin.
        Good for small number of bins and a lot of data.
    stdev : bool
        Choose whether to return error bars for each bin.
        Only works for portion equal to False
    avg : bool
        Choose whether to average data along radial contours
        or compute sum

    weight : ndarray
        relative weighting of each pixel in data.
        Default: uniform weighting

    Returns
    -------
    result : ndarray
        One-dimensional radial average
    phin : ndarray
        The bins of angles
    stdev : ndarray, optional
        Standard deviations from the average.
    """
    y, x = np.indices(data.shape)

    if center is None:
        xc = (x.max() - x.min())/2.
        yc = (y.max() - y.min())/2.
    else:
        xc, yc = center

    a = data
    if weight is not None:
        a *= weight

    if dphi is None:
        dphi = (bounds[1] - bounds[0]) / 20

    # angle with respect to center
    phi = np.arctan2(y - yc, x - xc)
    phi[np.where(phi < 0)] += 2*np.pi

    # bin by angle with respect to center
    phin = np.arange(bounds[0], bounds[1], dphi)
    ndx = np.digitize(phi.flat, phin)
    N = phin.size

    if portion:
        # apportion data according to position in bin
        fh1 = ((phi.flat - phin[ndx-1]) / dphi).reshape(phi.shape)
        fl1 = 1 - fh1
        ah1 = fh1 * a
        al1 = fh1 * a
        # double count values that exist on the border
        # of 2 bins to smooth out binning of angles
        mdx = ndx.copy()
        fh2 = fh1.flat.copy()
        fl2 = fl1.flat.copy()
        # get location of borders
        idx = np.where(fh1.flat <= 0.)
        jdx = np.where(fh1.flat >= 1.)
        # shift border locations to previous/next bin
        mdx[idx] = mdx[idx] - 1
        mdx[jdx] = mdx[jdx] + 1
        mdx[mdx == N+1] = 1
        mdx[mdx == 0] = N
        # an angle that was at the top of a bin is
        # now at the bottom of next bin (and vice versa)
        fh2[idx] = 1.
        fl2[jdx] = 0.
        # reshape and weigh data
        fh2 = fh2.reshape(fh1.shape)
        fl2 = fh2.reshape(fl1.shape)
        ah2 = fh2 * a
        al2 = fl2 * a

    # bin up data according angle
    acc = np.zeros(phin.size)
    count = np.zeros(phin.size)
    std = np.zeros(phin.size)
    for n in range(1, N+1):
        w = np.where(ndx == n)
        if not portion:
            sample = a.flat[w]
            acc[n-1] += sample.sum()
            count[n-1] += sample.size
            std[n-1] = np.std(sample)
        else:
            v = np.where(mdx == n)
            acc[n-1] += al1.flat[w].sum() + al2.flat[v].sum()
            count[n-1] += fl1.flat[w].sum() + fl2.flat[v].sum()
            if n != N:
                count[n] = fh1.flat[w].sum() + fh2.flat[v].sum()
                acc[n] = ah1.flat[w].sum() + ah2.flat[v].sum()

    nz = np.where(count > 0)
    result = acc[nz]/count[nz] if avg else acc[nz]
    phin = phin[nz]
    std = std[nz]

    if not stdev:
        return result, phin
    else:
        return result, phin, std


def aziavg(data, center=None, rad=None,
           portion=False, stdev=False,
           avg=True, weight=None):
    '''Average data over bins of azimuthal contours

    Extension of aziavg.py in https://github.com/davidgrier/dgmath/

    Arguments
    ---------
    data : ndarray
        two-dimensional data array.

    Keywords
    --------
    center : [xc, yc]
        Coordinates of the point around which to compute average
        Default: geometric center
    rad : ndarray or int
        array of bins to use in average or
        maximum radius of average [pixels]
        Default: half of the minimum dimension of the data
    portion : bool
        Choose whether to portion data by position in bin.
        Good for small number of bins compared to dimension
        of data.
    stdev : bool
        Choose whether to return error bars for each bin.
        Only works for portion equal to False
    avg : bool
        Choose whether to average data along azimuthal contours
        or to compute sum
    weight : ndarray
        relative weighting of each pixel in data.
        Default: uniform weighting

    Returns
    -------
    result : ndarray
        One-dimensional azimuthal average
    rn : ndarray
        Radial bins of average
    stdev : ndarray, optional
        Standard deviations from the average.
    '''
    y, x = np.indices(data.shape)

    if center is None:
        xc = (x.max() - x.min())/2.
        yc = (y.max() - y.min())/2.
    else:
        xc, yc = center

    if rad is None:
        rad = np.min([xc, x.max() - xc - 1, yc, y.max() - yc - 1])
        rad = np.floor(rad).astype(int) + 1

    a = data
    if weight is not None:
        a *= weight

    # distance to center
    r = np.hypot(x - xc, y - yc)

    # bin by distance to center
    if type(rad) is not np.ndarray:
        rn = np.arange(rad)
        dr = 1
    else:
        rn = rad
        dr = (rn[-1] - rn[0]) / rn.size
    ndx = np.digitize(r.flat, rn)

    if portion:
        # apportion data according to position in bin
        fh = ((r.flat - rn[ndx-1]) / dr).reshape(r.shape)
        fl = 1 - fh
        ah = fh * a
        al = fl * a

    # bin up data according to distance
    N = rn.size
    acc = np.zeros(N)
    count = np.zeros(N)
    std = np.zeros(N)
    for n in range(1, N):
        w = np.where(ndx == n)
        if not portion:
            sample = a.flat[w]
            acc[n-1] = sample.sum()
            count[n-1] = sample.size
            std[n-1] = np.std(sample)
        else:
            acc[n-1] += al.flat[w].sum()
            count[n-1] += fl.flat[w].sum()
            if n != N:
                acc[n] = ah.flat[w].sum()
                count[n] = fh.flat[w].sum()

    nz = np.where(count > 0)
    result = acc[nz]/count[nz] if avg else acc[nz]
    rn = rn[nz]
    std = std[nz]

    if not stdev:
        return result, rn
    else:
        return result, rn, std


def shellavg(data, center=None, rad=None,
             portion=False, stdev=False,
             avg=True, weight=None):
    """Average data over bins of spherical shells

    Adapted from aziavg.py in https://github.com/davidgrier/dgmath/

    Arguments
    ---------
    data : ndarray
        three-dimensional data array.

    Keywords
    --------
    center : [xc, yc, zc]
        Coordinates of the point around which to compute average
        Default: geometric center
    rad : np.ndarray or int
        array of bins to use in average or
        maximum radius of average [pixels]
        Default: half of the minimum dimension of the data
    portion : bool
        Choose whether to scale data according to position in bin.
        Good for small number of bins and a lot of data.
    stdev : bool
        Choose whether to return error bars for each bin.
        Only works for portion equal to False
    avg : bool
        Choose whether to average data along shells or compute sum
    weight : ndarray
        relative weighting of each pixel in data.
        Default: uniform weighting

    Returns
    -------
    result : ndarray
        One-dimensional angular average
    rn : ndarray
        Radial bins of average
    stdev : ndarray, optional
        Standard deviations from the average.
    """
    z, y, x = np.indices(data.shape)

    if center is None:
        xc = (x.max() - x.min())/2.
        yc = (y.max() - y.min())/2.
        zc = (z.max() - z.min())/2.
    else:
        xc, yc, zc = center

    if rad is None:
        rad = np.min([xc, x.max() - xc - 1,
                      yc, y.max() - yc - 1,
                      zc, z.max() - zc - 1])
        rad = np.floor(rad).astype(int) + 1

    a = data
    if weight is not None:
        a *= weight

    # distance to center
    r = np.sqrt((x-xc)**2 + (y-yc)**2 + (z-zc)**2)

    # bin by distance to center
    if type(rad) is not np.ndarray:
        rn = np.arange(rad)
        dr = 1
    else:
        rn = rad
        dr = (rad[-1] - rad[0]) / rad.size
    ndx = np.digitize(r.flat, rn)

    if portion:
        # apportion data according to position in bin
        fh = ((r.flat - rn[ndx-1]) / dr).reshape(r.shape)
        fl = 1 - fh
        ah = fh * a
        al = fl * a

    # bin up data according to distance
    N = rn.size
    acc = np.zeros(N)
    count = np.zeros(N)
    std = np.zeros(N)
    for n in range(1, N):
        w = np.where(ndx == n)
        if not portion:
            sample = a.flat[w]
            acc[n-1] = sample.sum()
            count[n-1] = sample.size
            std[n-1] = np.std(sample)
        else:
            acc[n-1] += al.flat[w].sum()
            count[n-1] += fl.flat[w].sum()
            if n != N:
                acc[n] = ah.flat[w].sum()
                count[n] = fh.flat[w].sum()

    nz = np.where(count > 0)
    result = acc[nz]/count[nz] if avg else acc[nz]
    rn = rn[nz]
    std = std[nz]

    if not stdev:
        return result, rn
    else:
        return result, rn, std


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    test = 'aziavg'

    if test == 'aziavg':
        dim = 24
        y, x = np.indices([dim, dim])
        yc, xc = 0, 0
        image = np.hypot(x-xc, y-yc) + np.random.random(x.shape)
        rad = np.arange(0, dim, .1)
        profile, rn, std = aziavg(image, center=[xc, yc], rad=rad, stdev=True)
        print(rn.size, profile.size, std.size)
        plt.errorbar(rn, profile, yerr=std, marker='.', capsize=1, alpha=.5)
        plt.show()
    elif test == 'radialavg':
        dim = 24
        y, x = np.indices([dim, dim])
        xc, yc = (0, 0)
        image = np.hypot(y-yc, x-xc)
        profile, phin = radialavg(image, center=[xc, yc],
                                  dphi=np.pi/48, bounds=(0, np.pi/2))
        plt.scatter(phin, profile)
        plt.show()
    elif test == 'shellavg':
        dim = 24
        z, y, x = np.indices([dim, dim, dim])
        zc, yc, xc = (0, 0, 0)
        image = np.sqrt((x-xc)**2 + (y-yc)**2 + (z-zc)**2)
        rad = np.arange(0, dim, .1)
        profile, rn = shellavg(image,
                               center=[xc, yc, zc],
                               rad=rad)
        plt.scatter(rn, profile)
        plt.show()
