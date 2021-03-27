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
from astropy.utils.console import ProgressBar


def bispectrum(data, kmin=None, kmax=None,
               nsamples=np.inf, mean_subtract=False,
               compute_fft=True, use_pyfftw=False, **kwargs):
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

    N, ndim = max(data.shape), data.ndim
    norm = float(data.size)**3

    if ndim not in [2, 3]:
        raise ValueError("Image must be a 2D or 3D")

    # Geometry of output image
    kmax = int(N/2) if kmax is None else int(kmax)
    kmin = 1 if kmin is None else int(kmin)
    kn = np.arange(kmin, kmax+1, 1, dtype=int)

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
    kcoords = np.meshgrid(*(ndim*[np.fft.fftfreq(N).astype(np.float32)*N]))
    if ndim == 2:
        kx, ky = [kv.ravel() for kv in kcoords]
        kr = np.sqrt(kx**2 + ky**2)
        kx, ky = [kv.astype(np.int16) for kv in [kx, ky]]
    else:
        kx, ky, kz = [kv.ravel() for kv in kcoords]
        kr = np.sqrt(kx**2 + ky**2 + kz**2)
        kx, ky, kz = [kv.astype(np.int16) for kv in [kx, ky, kz]]
    kr = np.digitize(kr, np.arange(int(np.ceil(kr.max()))))-1
    kr = kr.astype(np.int16)

    del kcoords

    if nsamples is None:
        nsamples = np.iinfo(np.int64).max
    if np.issubdtype(type(nsamples), np.integer):
        nsamples = np.full((kn.size, kn.size), nsamples, dtype=np.int64)

    # Run main loop
    if ndim == 2:
        bispec, binorm = _bispectrum2D(kr, kx, ky, kn, fft, N, nsamples)
    else:
        bispec, binorm = _bispectrum3D(kr, kx, ky, kz, kn, fft, N, nsamples)

    bicoh = np.abs(bispec) / binorm
    bispec /= norm

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


@nb.njit(cache=True, parallel=True)
def _bispectrum3D(kr, kx, ky, kz, kn, fft, N, nsamples):
    dim = kn.size
    bispec = np.zeros((dim, dim), dtype=np.complex128)
    binorm = np.zeros((dim, dim), dtype=np.float64)
    for i in range(dim):
        k1ind = np.where(kr == kn[i])[0]
        nk1 = k1ind.size
        for j in range(i+1):
            k2ind = np.where(kr == kn[j])[0]
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
            for idx in nb.prange(count):
                n, m = k1ind[samp[idx] % nk1], k2ind[samp[idx] // nk1]
                k1x, k1y, k1z = kx[n], ky[n], kz[n]
                k2x, k2y, k2z = kx[m], ky[m], kz[n]
                k1samp = fft[k1x, k1y, k1z]
                k2samp = fft[k2x, k2y, k1z]
                k3x, k3y, k3z = k1x+k2x, k1y+k2y, k1z+k2z
                if np.abs(k3x) > N//2 or np.abs(k3y) > N//2:
                    count -= 1
                else:
                    k3samp = np.conj(fft[k3x, k3y, k3z])
                    sample = k1samp*k2samp*k3samp
                    bispecbuf[idx] = sample
                    binormbuf[idx] = np.abs(sample)
            value = bispecbuf.sum() / count
            norm = binormbuf.sum() / count
            bispec[i, j], bispec[j, i] = value, value
            binorm[i, j], binorm[j, i] = norm, norm
    return bispec, binorm


@nb.njit(cache=True, parallel=True)
def _bispectrum2D(kr, kx, ky, kn, fft, N, nsamples):
    dim = kn.size
    bispec = np.zeros((dim, dim), dtype=np.complex128)
    binorm = np.zeros((dim, dim), dtype=np.float64)
    for i in range(dim):
        k1ind = np.where(kr == kn[i])[0]
        nk1 = k1ind.size
        for j in range(i+1):
            k2ind = np.where(kr == kn[j])[0]
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
            for idx in nb.prange(count):
                n, m = k1ind[samp[idx] % nk1], k2ind[samp[idx] // nk1]
                k1x, k1y = kx[n], ky[n]
                k2x, k2y = kx[m], ky[m]
                k1samp = fft[k1x, k1y]
                k2samp = fft[k2x, k2y]
                k3x, k3y = k1x+k2x, k1y+k2y
                if np.abs(k3x) > N//2 or np.abs(k3y) > N//2:
                    count -= 1
                else:
                    k3samp = np.conj(fft[k3x, k3y])
                    sample = k1samp*k2samp*k3samp
                    bispecbuf[idx] = sample
                    binormbuf[idx] = np.abs(sample)
            value = bispecbuf.sum() / count
            norm = binormbuf.sum() / count
            bispec[i, j] = value
            bispec[j, i] = value
            binorm[i, j] = norm
            binorm[j, i] = norm
    return bispec, binorm


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from astropy.io import fits

    # Open file
    N = 128
    data = np.random.normal(size=N**2).reshape((N, N))

    # Calculate
    fn = "./dens.fits.gz"
    data = fits.open(fn)[0].data#.sum(axis=0)
    bispec, bicoh, kn = bispectrum(data, nsamples=None, kmin=1, kmax=32,
                                   use_pyfftw=True)
    print(bispec.mean(), bicoh.mean())
    print(bicoh.max())

    # Plot
    cmap = 'plasma'
    labels = [r"$B(k_1, k_2)$", "$b(k_1, k_2)$"]
    data = [np.log10(np.abs(bispec)), bicoh]
    fig, axes = plt.subplots(ncols=2)
    for i in range(2):
        ax = axes[i]
        im = ax.imshow(data[i], origin="lower",
                       interpolation="nearest", cmap=cmap)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label(labels[i])
        if i == 0:
            ax.contour(data[i], colors='k')
        ax.set_xlabel(r"$k_1$")
        ax.set_ylabel(r"$k_2$")

    plt.tight_layout()

    plt.show()
