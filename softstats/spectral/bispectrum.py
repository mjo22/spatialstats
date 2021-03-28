"""
Bisprectrum calculation using numba acceleration

This implementation works on 2D and 3D rectangular domains for real
or complex valued data. It includes code for exact bispectrum
calculations and bispectrum calculations using uniform sampling.

See https://turbustat.readthedocs.io/en/latest/tutorials/statistics/bispectrum_example.html for more details.

Author:
    Michael J. O'Brien (2020)
    Biophysical Modeling Group
    Center for Computational Biology
    Flatiron Institute
"""

import numpy as np
import numba as nb
import pyfftw
from time import time


def bispectrum(data, kmin=None, kmax=None,
               nsamples=None, mean_subtract=False,
               compute_fft=True, use_pyfftw=False,
               bench=True, **kwargs):
    """
    Compute bispectrum in 2D or 3D.

    Arguments
    ---------
    data : np.ndarray
        Real or complex valued 2D or 3D scalar data.
        The shape should be (d1, d2) or (d1, d2, d3)
        where di is the ith dimension of the image.

    Keywords
    --------
    kmin : int
        Minimum wavenumber in bispectrum calculation.
    kmax : int
        Maximum wavenumber in bispectrum calculation.
    mean_subtract : bool
        Subtract mean off of image data to highlight
        non-linearities in bicoherence.
    compute_fft : bool
        If False, do not take the FFT of the input data.

    **kwargs passed to fftn (defined below)

    Returns
    -------
    bispectrum : np.ndarray
        Complex-valued 2D image
    bicoherence : np.ndarray
        Real-valued normalized bispectrum
    kn : np.ndarray
        Wavenumbers along axis of bispectrum
    """

    shape, ndim = nb.typed.List(data.shape), data.ndim
    norm = float(data.size)**3

    if ndim not in [2, 3]:
        raise ValueError("Image must be a 2D or 3D")

    # Geometry of output image
    kmax = int(max(shape)/2) if kmax is None else int(kmax)
    kmin = 1 if kmin is None else int(kmin)
    kn = np.arange(kmin, kmax+1, 1, dtype=int)
    dim = kn.size

    if bench:
        t0 = time()

    # FFT
    if compute_fft:
        temp = data - data.mean() if mean_subtract else data
        if use_pyfftw:
            fft = fftn(temp, **kwargs)
        else:
            fft = np.fft.fftn(temp, **kwargs)
    else:
        fft = data

    del temp

    # Get binned radial coordinates of FFT
    kv = np.meshgrid(*([np.fft.fftfreq(Ni).astype(np.float32)*Ni
                        for Ni in shape]), indexing="ij")
    kr = np.zeros_like(kv[0])
    for i in range(ndim):
        kr[...] += kv[i]**2
    kr[...] = np.sqrt(kr)

    kcoords = nb.typed.List()
    for i in range(ndim):
        temp = kv[i].astype(np.int16).ravel()
        kcoords.append(temp)

    del kv, temp

    kbins = np.arange(int(np.ceil(kr.max())))
    kbinned = (np.digitize(kr.ravel(), kbins)-1).astype(np.int16)

    del kr

    # Enumerate indices in each bin
    kind = nb.typed.List()
    for ki in kn:
        temp = np.where(kbinned == ki)[0].astype(np.int64)
        kind.append(temp)

    del kbinned

    if nsamples is None:
        nsamples = np.iinfo(np.int64).max
    if np.issubdtype(type(nsamples), np.integer):
        nsamples = np.full((dim, dim), nsamples, dtype=np.int64)

    # Run main loop
    compute_pixel = compute_pixel3D if ndim == 3 else compute_pixel2D
    bispec, binorm = compute_bispectrum(kind, kcoords, fft, nsamples,
                                        ndim, dim, shape, compute_pixel)

    bicoh = np.abs(bispec) / binorm
    bispec /= norm

    if bench:
        print(f"Time: {time() - t0:.04f} s")

    return bispec, bicoh, kn


def fftn(image, overwrite_input=False, threads=-1, **kwargs):
    """
    Calculate N-dimensional fft of image with pyfftw.
    See pyfftw.builders.fftn for kwargs documentation.

    Parameters
    ----------
    image : np.ndarray
        Real or complex valued 2D or 3D image

    Keywords
    --------
    overwrite_input : bool
        Specify whether input data can be destroyed.
        This is useful for reducing memory usage.
        See pyfftw.builders.fftn for more.
    threads : int
        Number of threads for pyfftw to use. Default
        is number of cores.

    **kwargs passed to pyfftw.builders.fftn

    Returns
    -------
    fft : np.ndarray
        The fft. Will be the shape of the input image
        or the user specified shape.
    """
    a = pyfftw.empty_aligned(image.shape, dtype=np.complex128)
    f = pyfftw.builders.fftn(a, overwrite_input=overwrite_input,
                             threads=threads, **kwargs)
    a[...] = image
    fft = f()

    del a

    return fft


@nb.njit(parallel=True)
def compute_bispectrum(kind, kcoords, fft, nsamples,
                       ndim, dim, shape, compute_pixel):
    bispec = np.zeros((dim, dim), dtype=np.complex128)
    binorm = np.zeros((dim, dim), dtype=np.float64)
    for i in range(dim):
        k1ind = kind[i]
        nk1 = k1ind.size
        for j in range(i+1):
            k2ind = kind[j]
            nk2 = k2ind.size
            nsamp = nsamples[i, j]
            if nsamp < nk1*nk2:
                samp = np.random.randint(0, nk1*nk2, size=nsamp)
                count = nsamp
            else:
                samp = np.arange(nk1*nk2)
                count = nk1*nk2
            bispecbuf = np.zeros(count, dtype=np.complex128)
            binormbuf = np.zeros(count)
            countbuf = np.zeros(count, dtype=np.int8)
            compute_pixel(k1ind, k2ind, kcoords, fft,
                          nk1, nk2, shape, samp, count,
                          bispecbuf, binormbuf, countbuf)
            value = bispecbuf.sum() / countbuf.sum()
            norm = binormbuf.sum() / countbuf.sum()
            bispec[i, j], bispec[j, i] = value, value
            binorm[i, j], binorm[j, i] = norm, norm
    return bispec, binorm


@nb.njit(parallel=True)
def compute_pixel3D(k1ind, k2ind, kcoords, fft, nk1, nk2, shape,
                    samp, count, bispecbuf, binormbuf, countbuf):
    kx, ky, kz = kcoords[0], kcoords[1], kcoords[2]
    Nx, Ny, Nz = shape[0], shape[1], shape[2]
    for idx in nb.prange(count):
        n, m = k1ind[samp[idx] % nk1], k2ind[samp[idx] // nk1]
        k1x, k1y, k1z = kx[n], ky[n], kz[n]
        k2x, k2y, k2z = kx[m], ky[m], kz[n]
        k3x, k3y, k3z = k1x+k2x, k1y+k2y, k1z+k2z
        if np.abs(k3x) > Nx//2 or np.abs(k3y) > Ny//2 or np.abs(k3z) > Nz//2:
            continue
        sample = fft[k1x, k1y, k1z]*fft[k2x, k2y, k2z]*np.conj(fft[k3x, k3y, k3z])
        bispecbuf[idx] = sample
        binormbuf[idx] = np.abs(sample)
        countbuf[idx] = 1


@nb.njit(parallel=True)
def compute_pixel2D(k1ind, k2ind, kcoords, fft, nk1, nk2, shape,
                    samp, count, bispecbuf, binormbuf, countbuf):
    kx, ky = kcoords[0], kcoords[1]
    Nx, Ny = shape[0], shape[1]
    for idx in nb.prange(count):
        n, m = k1ind[samp[idx] % nk1], k2ind[samp[idx] // nk1]
        k1x, k1y = kx[n], ky[n]
        k2x, k2y = kx[m], ky[m]
        k3x, k3y = k1x+k2x, k1y+k2y
        if np.abs(k3x) > Nx//2 or np.abs(k3y) > Ny//2:
            continue
        sample = fft[k1x, k1y]*fft[k2x, k2y]*np.conj(fft[k3x, k3y])
        bispecbuf[idx] = sample
        binormbuf[idx] = np.abs(sample)
        countbuf[idx] = 1


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from astropy.io import fits

    # Open file
    N = 128
    data = np.random.normal(size=N**2).reshape((N, N))+1

    # Calculate
    import os
    #fn = os.path.expanduser("~/dev/pybispec/dens.fits.gz")
    fn = "/mnt/home/mobrien/ceph/driving/b1p.05_r512_f7/data/dn_b1p.05_512_f7_500.fits"
    data = fits.open(fn)[0].data.astype(np.float32).sum(axis=0)
    kmin, kmax = 0, 32
    bispec, bicoh, kn = bispectrum(data, nsamples=None, kmin=kmin, kmax=kmax,
                                   mean_subtract=False,
                                   bench=True)
    print(bispec.mean(), bicoh.mean())
    print(bicoh.max())

    # Plot
    cmap = 'plasma'
    labels = [r"$B(k_1, k_2)$", "$b(k_1, k_2)$"]
    data = [np.log10(np.abs(bispec)), bicoh]
    fig, axes = plt.subplots(ncols=2)
    for i in range(2):
        ax = axes[i]
        #ax.set_xscale("log")
        #ax.set_yscale("log")
        im = ax.imshow(data[i], origin="lower",
                       interpolation="nearest",
                       cmap=cmap,
                       extent=[kmin, kmax, kmin, kmax])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label(labels[i])
        if i == 0:
            ax.contour(data[i], colors='k', extent=[kmin, kmax, kmin, kmax])
        ax.set_xlabel(r"$k_1$")
        ax.set_ylabel(r"$k_2$")

    plt.tight_layout()

    plt.show()
