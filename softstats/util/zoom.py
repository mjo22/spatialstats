"""
Zoom into a region of a grid with a specified coordinate system.
"""

import numpy as np


def grid_zoom(data, coords=None, region=None, skip=1):
    '''
    Zoom into a specific region of a vector field.
    Useful for visualization.'''
    ndim = data.shape[0]
    u = data
    if type(skip) is int:
        skip = ndim*[skip]
    if coords is not None:
        if type(coords) is np.ndarray:
            coords = ndim*[coords]
    else:
        coords = ndim*[np.arange(0, u[0].shape)]
    grid = np.meshgrid(*coords, indexing='ij')
    if region is not None:
        if type(region) is tuple:
            region = ndim*[region]
        regnew = []
        for i in range(ndim):
            n, l = grid[0].shape[i], coords[i].max()-coords[i].min()
            regnew.append((int(n/l*region[i][0]), int(n/l*region[i][1])))
        region = regnew
        inds = [np.arange(r[0], r[1]+1) for r in region]
        unew = []
        gridnew = []
        if ndim == 2:
            for i in range(ndim):
                unew.append(u[i][inds[0]][:, inds[1]])
                gridnew.append(grid[i][inds[0]][:, inds[1]])
        if ndim == 3:
            for i in range(ndim):
                unew.append(u[i][inds[0]][:, inds[1]][:, :, inds[2]])
                gridnew.append(grid[i][inds[0]][:, inds[1]][:, :, inds[2]])
        u, grid = unew, gridnew
    xargs, uargs = [], []
    for i in range(ndim):
        if ndim == 2:
            xarg = grid[i][::skip[0], ::skip[1]]
            uarg = u[i][::skip[0], ::skip[1]]
        elif ndim == 3:
            xarg = grid[i][::skip[0], ::skip[1], ::skip[2]]
            uarg = u[i][::skip[0], ::skip[1], ::skip[2]]
        xargs.append(xarg)
        uargs.append(uarg)
    return uargs, xargs
