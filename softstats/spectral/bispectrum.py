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


def bispectrum(*args, nsamples=100000, **kwargs):
    """
    Bispectrum wrapper that chooses between exact
    and sampling implementations.
    """
    if nsamples is None:
        result = bispectrum_exact(*args, **kwargs)
    else:
        result = bispectrum_sampled(*args, nsamples=nsamples, **kwargs)
    return result


def bispectrum_sampled(data, vector=False, nsamples=100000,
                       mean_subtract=False, seed=None, chunks=None,
                       npts=None, kmin=None, kmax=None,
                       compute_fft=True, use_pyfftw=False,
                       bench=False, progress=False, **kwargs):
    """
    Compute the bispectrum of 2D or 3D data using uniform sampling.

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
        large chunks, the user may receive an OutOfMemoryError.
    mean_subtract : bool
        Subtract mean off of image data to highlight
        non-linearities in bicoherence
    seed : bool
        Random number seed
    chunks : int
        Determines the number of iterations to calculate the
        bispectrum. For sufficiently large nsamples and
        large chunks, the user may receive an MemoryError.
    npts : int
        Number of wavenumbers in bispectrum calculation
    kmin : int
        Minimum wavenumber in bispectrum calculation
    kmax : int
        Maximum wavenumber in bispectrum calculation
    compute_fft : bool
        If False, do not take the FFT of the input data.
    use_pyfftw : bool
        If True, use function fftn (below) to calculate
        FFTs using pyfftw. If False, use numpy implementation.
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
        temp = data[0]
        N, ndim = max(temp.shape), temp.ndim
        ncomp = data.shape[0]
        shape = temp.shape
        norm = float(temp.size)**3
        kernel = _bispectrumVec3D if ndim == 3 else _bispectrumVec2D
    else:
        N, ndim = max(data.shape), data.ndim
        ncomp = 1
        shape = data.shape
        norm = float(data.size)**3
        kernel = _bispectrum3D if ndim == 3 else _bispectrum2D

    if ndim not in [2, 3]:
        raise ValueError("Image must be a 2D or 3D")

    if bench:
        t0 = time()

    # Geometry of output image
    kmax = int(N/2) if kmax is None else int(kmax)
    kmin = 1 if kmin is None else int(kmin)
    dim = kmax-kmin+1 if npts is None else npts
    chunks = dim if chunks is None else chunks
    kn = np.linspace(kmin, kmax, dim, dtype=float, endpoint=True)

    if kmax > N//2:
        raise ValueError(f"kmax should not exceed {N//2}")

    # Take fft using fftn defined below
    fft = []
    for i in range(ncomp):
        comp = np.empty(shape, dtype=complex)
        temp = data[i] if vector else data
        comp[...] = temp
        if compute_fft:
            # Subtract mean of data to highlight non-linearities
            if mean_subtract:
                comp[...] -= comp.mean()
            if use_pyfftw:
                fftcomp = fftn(comp, **kwargs)
            else:
                fftcomp = np.fft.fftn(comp, **kwargs)
        else:
            fftcomp = comp
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

        buf[...] = 0.
        cbuf[...] = 0.
        out[...] = 0.

        npix = np.int32(bind.size)

        birebuf, biimbuf, binormbuf = buf[0], buf[1], buf[2]
        bire, biim, binorm = out[0], out[1], out[2]

        kernel(birebuf, biimbuf, binormbuf, cbuf, bind, npix, *args)

        bire[:] = np.sum(birebuf, axis=1)
        biim[:] = np.sum(biimbuf, axis=1)
        binorm[:] = np.sum(binormbuf, axis=1)
        count[:] = np.sum(cbuf, axis=1)

    # Chunk size
    nchunks = dim**2//chunks
    if dim**2 % chunks != 0:
        msg = f"chunks {chunks} must divide squared image dimension {dim}"
        raise ValueError(msg)

    # Catch memory errors. Larger datasets require less parallelization.
    try:
        buf = np.zeros((3, chunks, nsamples), dtype=float)
        cbuf = np.zeros((chunks, nsamples), dtype=np.int32)
    except MemoryError as err:
        msg = f"Out of memory allocating buffers of shape {(chunks, nsamples)}. "
        msg += "Try decreasing chunks."
        raise ValueError(msg) from err

    if progress:
        bar = ProgressBar(nchunks)

    # Calculate chunks pixels at a time
    result = np.zeros((3, npix), dtype=float)
    out = np.zeros((3, chunks), dtype=float)
    count = np.zeros(chunks, dtype=np.int32)
    bire, biim, biconorm = result[0], result[1], result[2]
    for i in range(nchunks):
        start, stop = i*chunks, (i+1)*chunks
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
        if progress:
            print()
        print(f"Time: {time() - t0:.04f} s")

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
def _bispectrum3D(birebuf, biimbuf, binormbuf, count, bind, npixels,
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

        if np.abs(k3i) > N1//2 or np.abs(k3j) > N2//2 or np.abs(k3k) > N3//2:
            continue

        sample = fft[k1i, k1j, k1k] * fft[k2i, k2j, k2k] \
            * np.conj(fft[k3i, k3j, k3k])
        mod = np.abs(sample)

        birebuf[l, k] = np.real(sample)
        biimbuf[l, k] = np.imag(sample)
        binormbuf[l, k] = mod
        count[l, k] = 1


@nb.njit(cache=True, parallel=True)
def _bispectrum2D(birebuf, biimbuf, binormbuf, count, bind, npixels,
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

        if np.abs(k3i) > N1//2 or np.abs(k3j) > N2//2:
            continue

        sample = fft[k1i, k1j] * fft[k2i, k2j] \
            * np.conj(fft[k3i, k3j])
        mod = np.abs(sample)

        birebuf[l, k] = np.real(sample)
        biimbuf[l, k] = np.imag(sample)
        binormbuf[l, k] = mod
        count[l, k] = 1


@nb.njit(cache=True, parallel=True)
def _bispectrumVec3D(birebuf, biimbuf, binormbuf, count, bind, npixels,
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

        if np.abs(k3i) > N1//2 or np.abs(k3j) > N2//2 or np.abs(k3k) > N3//2:
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
def _bispectrumVec2D(birebuf, biimbuf, binormbuf, count, bind, npixels,
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

        if np.abs(k3i) > N1//2 or np.abs(k3j) > N2//2:
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


def bispectrum_exact(data, kmin=None, kmax=None,
                     mean_subtract=False, compute_fft=True, **kwargs):
    """
    Compute exact bispectrum in 2D or 3D.

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
        fft = np.fft.fftn(temp, **kwargs)
    else:
        fft = data

    # Get binned radial coordinates of FFT
    kcoords = np.meshgrid(*(ndim*[np.fft.fftfreq(N)*N]))
    if ndim == 2:
        kx, ky = [kv.astype(int) for kv in kcoords]
        kr = np.sqrt(kx**2 + ky**2)
    else:
        kx, ky, kz = [kv.astype(int) for kv in kcoords]
        kr = np.sqrt(kx**2 + ky**2 + kz**2)
    kr = np.digitize(kr, np.arange(int(np.ceil(kr.max()))))-1

    # Run main loop
    if ndim == 2:
        bispec, binorm = _bispectrumExact2D(kr, kx, ky, kn, fft, N)
    else:
        bispec, binorm = _bispectrumExact3D(kr, kx, ky, kz, kn, fft, N)

    bicoh = np.abs(bispec) / binorm
    bispec /= norm

    return bispec, bicoh, kn


@nb.njit(cache=True)
def _bispectrumExact2D(kr, kx, ky, kn, fft, N):
    bispec = np.zeros((kn.size, kn.size), dtype=np.complex128)
    binorm = np.zeros((kn.size, kn.size), dtype=np.float64)
    for i in range(kn.size):
        k1ind = np.where(kr.ravel() == kn[i])
        k1samples = fft.ravel()[k1ind]
        nk1 = k1samples.size
        for j in range(kn.size):
            k2ind = np.where(kr.ravel() == kn[j])
            k2samples = fft.ravel()[k2ind]
            nk2 = k2samples.size
            norm = nk1*nk2
            for n in range(nk1):
                ndx = k1ind[0][n]
                k1x, k1y = kx.ravel()[ndx], ky.ravel()[ndx]
                k1samp = fft[k1x, k1y]
                for m in range(nk2):
                    mdx = k2ind[0][m]
                    k2x, k2y = kx.ravel()[mdx], ky.ravel()[mdx]
                    k2samp = fft[k2x, k2y]
                    k3x, k3y = k1x+k2x, k1y+k2y
                    if np.abs(k3x) > N//2 or np.abs(k3y) > N//2:
                        norm -= 1
                    else:
                        k3samp = np.conj(fft[k3x, k3y])
                        sample = k1samp*k2samp*k3samp
                        bispec[i, j] += sample
                        binorm[i, j] += np.abs(sample)
            bispec[i, j] /= norm
            binorm[i, j] /= norm
    return bispec, binorm


@nb.njit(cache=True)
def _bispectrumExact3D(kr, kx, ky, kz, kn, fft, N):
    bispec = np.zeros((kn.size, kn.size), dtype=np.complex128)
    binorm = np.zeros((kn.size, kn.size), dtype=np.float64)
    for i in range(kn.size):
        k1ind = np.where(kr.flat == kn[i])
        k1samples = fft.flat[k1ind]
        nk1 = k1samples.size
        for j in range(kn.size):
            k2ind = np.where(kr.flat == kn[j])
            k2samples = fft.flat[k2ind]
            nk2 = k2samples.size
            norm = nk1*nk2
            for n in range(nk1):
                ndx = k1ind[0][n]
                k1x, k1y, k1z = kx.flat[ndx], ky.flat[ndx], kz.flat[ndx]
                k1samp = fft[k1x, k1y, k1z]
                for m in range(nk2):
                    mdx = k2ind[0][m]
                    k2x, k2y, k2z = kx.flat[mdx], ky.flat[mdx], kz.flat[mdx]
                    k2samp = fft[k2x, k2y, k2z]
                    k3x, k3y, k3z = k1x+k2x, k1y+k2y, k1z+k2z
                    if np.abs(k3x) > N//2 or np.abs(k3y) > N//2  \
                       or np.abs(k3z) > N//2:
                        norm -= 1
                    else:
                        k3samp = np.conj(fft[k3x, k3y, k3z])
                        sample = k1samp*k2samp*k3samp
                        bispec[i, j] += sample
                        binorm[i, j] += np.abs(sample)
            bispec[i, j] /= norm
            binorm[i, j] /= norm
    return bispec, binorm


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from astropy.io import fits

    # Open file
    N = 128
    data = np.random.normal(size=N**2).reshape((N, N))

    # Calculate
    bispec, bicoh, kn = bispectrum(data, nsamples=int(1e4), chunks=64,
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
