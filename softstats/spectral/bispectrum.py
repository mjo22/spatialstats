"""
Bisprectrum calculation using numba acceleration

This implementation works on 2D and 3D rectangular domains for real
or complex valued data. It uniformly samples over all
triangles in fourier space for a given two wavenumbers.

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


def bispectrum(data, vector=False, nsamples=100000,
               mean_subtract=False, seed=None, nchunks=None,
               npts=None, kmin=None, kmax=None,
               bench=True, progress=False, use_pyfftw=False, **kwargs):
    """
    Compute the bispectrum of 2D or 3D data with
    numba acceleration in double-precision.

    Parameters
    ----------
    data : np.ndarray
        Real or complex valued 2D or 3D vector or scalar data.
        If vector data, the shape should be (n, d1, d2) or 
        (n, d1, d2, d3) where n is the number of vector components
        and di is the ith dimension of the image.

    Keywords
    --------
    vector : bool
        Specify whether you have passed vector or scalar data
    nsamples : int
        The number of triangles to sample for each
        pixel. For sufficiently large nsamples and
        small nchunks, the user may receive an OutOfMemoryError.
    mean_subtract : bool
        Subtract mean off of image data to highlight
        non-linearities in bicoherence
    seed : bool
        Random number seed
    nchunks : int
        The number of iterations to calculate the
        bispectrum. For sufficiently large nsamples and
        small nchunks, the user may receive an OutOfMemoryError.
    npts : int
        Number of wavenumbers in bispectrum calculation
    kmin : int
        Minimum wavenumber in bispectrum calculation
    kmax : int
        Maximum wavenumber in bispectrum calculation
    bench : bool
        Return compute times of calculation
    progress : bool
        Display progress bar of calculation

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
    float, complex = np.float64, np.complex128

    if vector:
        N, ndim = max(data[0].shape), data[0].ndim
        ncomp = data.shape[0]
        shape = data[0].shape
        norm = float(data[0].size)**3
        kernel = _kernel3Dvec if ndim == 3 else _kernel2Dvec
    else:
        N, ndim = max(data.shape), data.ndim
        ncomp = 1
        shape = data.shape
        norm = float(data.size)**3
        kernel = _kernel3D if ndim == 3 else _kernel2D

    if ndim not in [2, 3]:
        raise ValueError("Image must be a 2D or 3D")

    if bench:
        t0 = time()

    # Geometry of output image
    kmax = int(N/2) if kmax is None else int(kmax)
    kmin = 0 if kmin is None else int(kmin)
    dim = kmax-kmin if npts is None else npts
    nchunks = dim if nchunks is None else nchunks
    kn = np.linspace(kmin, kmax, dim, dtype=float, endpoint=False)

    if kmax > N//2:
        raise ValueError(f"kmax should not exceed {N//2}")

    # Take fft using fftn defined below
    fft = []
    for i in range(ncomp):
        comp = np.empty(shape, dtype=complex)
        temp = data[i] if vector else data
        comp[...] = temp
        # Subtract mean of data to highlight non-linearities
        if mean_subtract:
            comp[...] -= comp.mean()
        if use_pyfftw:
            fftcomp = fftn(comp, **kwargs)
        else:
            fftcomp = np.fft.fftn(comp, **kwargs)
        fft.append(fftcomp)
        del temp

    del comp

    # Array of pixel indices
    npix = dim**2
    bind = np.indices((npix,), dtype=np.int32)[0]

    # Random samples. Uses same samples for all spheres in k-space
    if seed is not None:
        np.random.seed(seed=seed)

    # Package args for dispatch
    args = (kn, dim, np.int32(nsamples), *fft, *shape)

    def dispatch(bind, buf, out, cbuf, count):

        npix = np.int32(bind.size)

        birebuf, biimbuf, binormbuf = buf[0], buf[1], buf[2]
        bire, biim, binorm = out[0], out[1], out[2]

        kernel(birebuf, biimbuf, binormbuf, cbuf, bind, npix, *args)

        bire[:] = np.sum(birebuf, axis=1)
        biim[:] = np.sum(biimbuf, axis=1)
        binorm[:] = np.sum(binormbuf, axis=1)
        count[:] = np.sum(cbuf, axis=1)

    # Chunk size
    chunk = dim**2//nchunks
    if dim**2 % nchunks != 0:
        msg = f"nchunks {nchunks} must divide squared image dimension {dim}"
        raise ValueError(msg)

    # Catch memory errors. Larger datasets require less parallelization.
    try:
        buf = np.zeros((3, chunk, nsamples), dtype=float)
        cbuf = np.zeros((chunk, nsamples), dtype=np.int32)
    except MemoryError as err:
        msg = f"Out of memory allocating buffers of shape {(chunk, nsamples)}. "
        msg += "Try increasing nchunks."
        raise ValueError(msg) from err

    if progress:
        bar = ProgressBar(nchunks)

    # Calculate chunk pixels at a time
    result = np.zeros((3, npix), dtype=float)
    out = np.zeros((3, chunk), dtype=float)
    count = np.zeros(chunk, dtype=np.int32)
    bire, biim, biconorm = result[0], result[1], result[2]
    for i in range(nchunks):
        start, stop = i*chunk, (i+1)*chunk
        ind = bind[start:stop]
        dispatch(ind, buf, out, cbuf, count)
        bire[start:stop] = out[0]/count
        biim[start:stop] = out[1]/count
        biconorm[start:stop] = out[2]/count
        if progress:
            bar.update(i+1)

    # Finalize and transfer to cpu
    bispectrum = bire + 1.j*biim
    bicoh = (np.abs(bispectrum) / biconorm).reshape((dim, dim))
    bispec = bispectrum.reshape((dim, dim)) / (norm*nsamples)

    # Release memory
    del fft, bispectrum
    del buf, cbuf, result, count, out, bire, biim, bind, biconorm

    if bench:
        print(f"\nTime: {time() - t0:.04f}")

    return bispec, bicoh, kn


def fftn(image, overwrite_input=True, threads=-1, **kwargs):
    """
    Calculate N-dimensional fft of image with pyfftw.
    See pyfftw.builders.fftn for kwargs documentation.

    Parameters
    ----------
    image : np.ndarray
        Real or complex valued 2D or 3D image

    Keywords
    --------
    dtype : str
        Specify precision for pyfftw buffer.
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
    if image.dtype in [np.complex64, np.complex128]:
        dtype = 'complex128'
        fftn = pyfftw.builders.fftn
    elif image.dtype in [np.float32, np.float64]:
        dtype = 'float64'
        fftn = pyfftw.builders.rfftn
    else:
        raise ValueError(f"{data.dtype} is unrecognized data type.")

    a = pyfftw.empty_aligned(image.shape, dtype=dtype)
    f = fftn(a, threads=threads, overwrite_input=overwrite_input, **kwargs)
    a[...] = image
    fft = f()

    del a, fftn

    return fft


@nb.njit(cache=True, parallel=True)
def _kernel3D(birebuf, biimbuf, binormbuf, count, bind, npixels,
              kn, dim, nsamples, fft, N1, N2, N3):
    for idx in nb.prange(npixels*nsamples):

        l = idx // nsamples
        k = idx % nsamples

        bidx = bind[l]
        i = bidx // dim
        j = bidx % dim

        k1mag = kn[i]
        k2mag = kn[j]

        phi1 = np.random.uniform(0., 2*np.pi)
        phi2 = np.random.uniform(0., 2*np.pi)
        costheta1 = np.random.uniform(-1., 1.)
        costheta2 = np.random.uniform(-1., 1.)

        cosphi1 = np.cos(phi1)
        sinphi1 = np.sin(phi1)
        cosphi2 = np.cos(phi2)
        sinphi2 = np.sin(phi2)
        sintheta1 = np.sqrt(1 - costheta1**2)
        sintheta2 = np.sqrt(1 - costheta2**2)

        k1x = k1mag * cosphi1 * sintheta1
        k1y = k1mag * sinphi1 * sintheta1
        k1z = k1mag * costheta1
        k2x = k2mag * cosphi2 * sintheta2
        k2y = k2mag * sinphi2 * sintheta2
        k2z = k2mag * costheta2
        k3x = k1x + k2x
        k3y = k1y + k2y
        k3z = k1z + k2z

        k1i = np.int32(np.rint(k1x))
        k1j = np.int32(np.rint(k1y))
        k1k = np.int32(np.rint(k1z))
        k2i = np.int32(np.rint(k2x))
        k2j = np.int32(np.rint(k2y))
        k2k = np.int32(np.rint(k2z))
        k3i = np.int32(np.rint(k3x))
        k3j = np.int32(np.rint(k3y))
        k3k = np.int32(np.rint(k3z))

        if k3i > N1//2 or k3j > N2//2 or k3k > N3//2:
            continue

        sample = fft[k1i, k1j, k1k] * fft[k2i, k2j, k2k] \
            * np.conj(fft[k3i, k3j, k3k])
        mod = np.abs(sample)

        birebuf[l, k] = np.real(sample)
        biimbuf[l, k] = np.imag(sample)
        binormbuf[l, k] = mod
        count[l, k] = 1


@nb.njit(cache=True, parallel=True)
def _kernel2D(birebuf, biimbuf, binormbuf, count, bind, npixels,
              kn, dim, nsamples, fft, N1, N2):
    for idx in nb.prange(npixels*nsamples):
        l = idx // nsamples
        k = idx % nsamples

        bidx = bind[l]
        i = bidx // dim
        j = bidx % dim

        k1mag = kn[i]
        k2mag = kn[j]

        phi1 = np.random.uniform(0., 2*np.pi)
        phi2 = np.random.uniform(0., 2*np.pi)

        cosphi1 = np.cos(phi1)
        sinphi1 = np.sin(phi1)
        cosphi2 = np.cos(phi2)
        sinphi2 = np.sin(phi2)

        k1x = k1mag * cosphi1
        k1y = k1mag * sinphi1
        k2x = k2mag * cosphi2
        k2y = k2mag * sinphi2
        k3x = k1x + k2x
        k3y = k1y + k2y

        k1i = np.int32(np.rint(k1x))
        k2i = np.int32(np.rint(k2x))
        k3i = np.int32(np.rint(k3x))
        k1j = np.int32(np.rint(k1y))
        k2j = np.int32(np.rint(k2y))
        k3j = np.int32(np.rint(k3y))

        if k3i > N1//2 or k3j > N2//2:
            continue

        sample = fft[k1i, k1j] * fft[k2i, k2j] \
            * np.conj(fft[k3i, k3j])
        mod = np.abs(sample)

        birebuf[l, k] = np.real(sample)
        biimbuf[l, k] = np.imag(sample)
        binormbuf[l, k] = mod
        count[l, k] = 1


@nb.njit(cache=True, parallel=True)
def _kernel3Dvec(birebuf, biimbuf, binormbuf, count, bind, npixels,
                 kn, dim, nsamples, fftx, ffty, fftz, N1, N2, N3):
    for idx in nb.prange(npixels*nsamples):

        l = idx // nsamples
        k = idx % nsamples

        bidx = bind[l]
        i = bidx // dim
        j = bidx % dim

        k1mag = kn[i]
        k2mag = kn[j]

        phi1 = np.random.uniform(0., 2*np.pi)
        phi2 = np.random.uniform(0., 2*np.pi)
        costheta1 = np.random.uniform(-1., 1.)
        costheta2 = np.random.uniform(-1., 1.)

        cosphi1 = np.cos(phi1)
        sinphi1 = np.sin(phi1)
        cosphi2 = np.cos(phi2)
        sinphi2 = np.sin(phi2)
        sintheta1 = np.sqrt(1 - costheta1**2)
        sintheta2 = np.sqrt(1 - costheta2**2)

        k1x = k1mag * cosphi1 * sintheta1
        k1y = k1mag * sinphi1 * sintheta1
        k1z = k1mag * costheta1
        k2x = k2mag * cosphi2 * sintheta2
        k2y = k2mag * sinphi2 * sintheta2
        k2z = k2mag * costheta2
        k3x = k1x + k2x
        k3y = k1y + k2y
        k3z = k1z + k2z

        k1i = np.int32(np.rint(k1x))
        k1j = np.int32(np.rint(k1y))
        k1k = np.int32(np.rint(k1z))
        k2i = np.int32(np.rint(k2x))
        k2j = np.int32(np.rint(k2y))
        k2k = np.int32(np.rint(k2z))
        k3i = np.int32(np.rint(k3x))
        k3j = np.int32(np.rint(k3y))
        k3k = np.int32(np.rint(k3z))

        if k3i > N1//2 or k3j > N2//2 or k3k > N3//2:
            continue

        sample = fftx[k1i, k1j, k1k] * fftx[k2i, k2j, k2k] \
            * np.conj(fftx[k3i, k3j, k3k])
        sample += ffty[k1i, k1j, k1k] * ffty[k2i, k2j, k2k] \
            * np.conj(ffty[k3i, k3j, k3k])
        sample += fftz[k1i, k1j, k1k] * fftz[k2i, k2j, k2k] \
            * np.conj(fftz[k3i, k3j, k3k])
        mod = np.abs(sample)

        birebuf[l, k] = np.real(sample)
        biimbuf[l, k] = np.imag(sample)
        binormbuf[l, k] = mod
        count[l, k] = 1


@nb.njit(cache=True, parallel=True)
def _kernel2Dvec(birebuf, biimbuf, binormbuf, count, bind, npixels,
                 kn, dim, nsamples, fftx, ffty, N1, N2):
    for idx in nb.prange(npixels*nsamples):
        l = idx // nsamples
        k = idx % nsamples

        bidx = bind[l]
        i = bidx // dim
        j = bidx % dim

        k1mag = kn[i]
        k2mag = kn[j]

        phi1 = np.random.uniform(0., 2*np.pi)
        phi2 = np.random.uniform(0., 2*np.pi)

        cosphi1 = np.cos(phi1)
        sinphi1 = np.sin(phi1)
        cosphi2 = np.cos(phi2)
        sinphi2 = np.sin(phi2)

        k1x = k1mag * cosphi1
        k1y = k1mag * sinphi1
        k2x = k2mag * cosphi2
        k2y = k2mag * sinphi2
        k3x = k1x + k2x
        k3y = k1y + k2y

        k1i = np.int32(np.rint(k1x))
        k1j = np.int32(np.rint(k1y))
        k2i = np.int32(np.rint(k2x))
        k2j = np.int32(np.rint(k2y))
        k3i = np.int32(np.rint(k3x))
        k3j = np.int32(np.rint(k3y))

        if k3i > N1//2 or k3j > N2//2:
            continue

        sample = fftx[k1i, k1j] * fftx[k2i, k2j] \
            * np.conj(fftx[k3i, k3j])
        sample += ffty[k1i, k1j] * ffty[k2i, k2j] \
            * np.conj(ffty[k3i, k3j])
        mod = np.abs(sample)

        birebuf[l, k] = np.real(sample)
        biimbuf[l, k] = np.imag(sample)
        binormbuf[l, k] = mod
        count[l, k] = 1


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from astropy.io import fits

    # Open file
    hdul = fits.open('dens.fits.gz')
    image = hdul[0].data.astype(np.float64)  # [:512, :512, :512]

    # Calculate
    bispec, bicoh, kn = bispec(image, nsamples=int(1e6),
                               progress=True, mean_subtract=True)
    print(bispec.mean(), bicoh.mean())

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
