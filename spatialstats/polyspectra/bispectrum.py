"""
Calculating the bispectrum using numba acceleration.

This implementation works on 2D and 3D rectangular domains for real
or complex valued data. It can compute the bispectrum exactly or
using uniform sampling.

.. moduleauthor:: Michael O'Brien <michaelobrien@g.harvard.edu>

"""

import numpy as np
import numba as nb
from time import time


def bispectrum(data, kmin=None, kmax=None,
               nsamples=None, sample_thresh=None,
               exclude=False, mean_subtract=False,
               compute_fft=True, full=False,
               use_pyfftw=False,
               bench=False, progress=False, **kwargs):
    """
    Compute the bispectrum of 2D or 3D real or complex valued data.

    kwargs are passed to np.fft.fftn, np.fft.rfftn,
    pyfftw.builders.fftn, or pyfftw.builders.rfftn.

    Parameters
    ----------
    data : np.ndarray
        Real or complex valued 2D or 3D scalar data.
        The shape should be (d1, d2) or (d1, d2, d3)
        where di is the ith dimension of the image.
    kmin : int, optional
        Minimum wavenumber in bispectrum calculation.
    kmax : int, optional
        Maximum wavenumber in bispectrum calculation.
    nsamples : int, float or np.ndarray, shape (kmax-kmin+1, kmax-kmin+1), optional
        Number of sample triangles or fraction of total
        possible triangles. This may be an array that
        specifies for a given k1, k2. If None, calculate
        the bispectrum exactly.
    sample_thresh : int, optional
        When the size of the sample space is greater than
        this number, start to use sampling instead of exact
        calculation. If None, switch to exact calculation
        when nsamples is less than the size of the sample space.
    exclude : bool, optional
        If True, exclude k1, k2 such that k1 + k2 is greater
        than the Nyquist frequency. Excluded points will be
        set to nan.
    mean_subtract : bool, optional
        Subtract mean off of image data to highlight
        non-linearities in bicoherence.
    compute_fft : bool, optional
        If False, do not take the FFT of the input data.
    full : bool, optional
        Return the full output of calculation. Namely,
        return the optional sampling diagnostics.
    use_pyfftw : bool, optional
        If True, use pyfftw to compute the FFTs.
    bench : bool, optional
        If True, print calculation time.
    progress : bool, optional
        Print progress bar of calculation.

    Returns
    -------
    bispec : np.ndarray, shape (kmax-kmin+1, kmax-kmin+1)
        Real or complex-valued bispectrum.
        Will be real-valued if the input data is real.
    bicoh : np.ndarray, shape (kmax-kmin+1, kmax-kmin+1)
        Real-valued bicoherence.
    kn : np.ndarray
        Wavenumbers along axis of bispectrum.
    omega : np.ndarray, shape (kmax-kmin+1, kmax-kmin+1), optional
        Number of possible triangles in the sample space.
    counts : np.ndarray, shape (kmax-kmin+1, kmax-kmin+1), optional
        Number of evaluations in the bispectrum sum.
    """

    shape, ndim = nb.typed.List(data.shape), data.ndim

    if ndim not in [2, 3]:
        raise ValueError("Data must be 2D or 3D.")

    # Geometry of output image
    kmax = int(max(shape)/2) if kmax is None else int(kmax)
    kmin = 1 if kmin is None else int(kmin)
    kn = np.arange(kmin, kmax+1, 1, dtype=int)
    dim = kn.size

    if bench:
        t0 = time()

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

    # FFT
    if compute_fft:
        temp = data - data.mean() if mean_subtract else data
        if use_pyfftw:
            fft = _fftn(temp, **kwargs)
        else:
            fft = np.fft.fftn(temp, **kwargs)
        del temp
    else:
        fft = data

    if sample_thresh is None:
        sample_thresh = np.iinfo(np.int64).max
    if nsamples is None:
        nsamples = np.iinfo(np.int64).max
        sample_thresh = np.iinfo(np.int64).max

    if np.issubdtype(type(nsamples), np.integer):
        nsamples = np.full((dim, dim), nsamples, dtype=np.int_)
    elif np.issubdtype(type(nsamples), np.floating):
        nsamples = np.full((dim, dim), nsamples)
    elif type(nsamples) is np.ndarray:
        if np.issubdtype(nsamples.dtype, np.integer):
            nsamples = nsamples.astype(np.int_)

    # Run main loop
    compute_point = _compute_point3D if ndim == 3 else _compute_point2D
    args = (kind, kn, kcoords, fft, nsamples, sample_thresh,
            ndim, dim, shape, progress, exclude, compute_point)
    bispec, binorm, omega, counts = _compute_bispectrum(*args)

    if np.issubdtype(data.dtype, np.floating):
        bispec = bispec.real

    bicoh = np.abs(bispec) / binorm
    bispec *= (omega / counts)

    if bench:
        print(f"Time: {time() - t0:.04f} s")

    if not full:
        return bispec, bicoh, kn
    else:
        return bispec, bicoh, kn, omega, counts


def _fftn(image, overwrite_input=False, threads=-1, **kwargs):
    """
    Calculate N-dimensional fft of image with pyfftw.
    See pyfftw.builders.fftn for kwargs documentation.

    Parameters
    ----------
    image : np.ndarray
        Real or complex valued 2D or 3D image
    overwrite_input : bool, optional
        Specify whether input data can be destroyed.
        This is useful for reducing memory usage.
        See pyfftw.builders.fftn for more.
    threads : int, optional
        Number of threads for pyfftw to use. Default
        is number of cores.

    **kwargs passed to pyfftw.builders.fftn

    Returns
    -------
    fft : np.ndarray
        The fft. Will be the shape of the input image
        or the user specified shape.
    """
    import pyfftw

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


@nb.njit(parallel=True)
def _compute_bispectrum(kind, kn, kcoords, fft, nsamples, sample_thresh,
                        ndim, dim, shape, progress, exclude, compute_point):
    knyq = max(shape) // 2
    bispec = np.full((dim, dim), np.nan, dtype=np.complex128)
    binorm = np.full((dim, dim), np.nan, dtype=np.float64)
    omega = np.zeros((dim, dim), dtype=np.int64)
    counts = np.zeros((dim, dim), dtype=np.int64)
    for i in range(dim):
        k1 = kn[i]
        k1ind = kind[i]
        nk1 = k1ind.size
        for j in range(i+1):
            k2 = kn[j]
            if exclude and k1 + k2 > knyq:
                continue
            k2ind = kind[j]
            nk2 = k2ind.size
            nsamp = nsamples[i, j]
            nsamp = int(nsamp) if type(nsamp) is np.int64 \
                else max(int(nsamp*nk1*nk2), 1)
            if nsamp < nk1*nk2 or nsamp > sample_thresh:
                samp = np.random.randint(0, nk1*nk2, size=nsamp)
                count = nsamp
            else:
                samp = np.arange(nk1*nk2)
                count = nk1*nk2
            bispecbuf = np.zeros(count, dtype=np.complex128)
            binormbuf = np.zeros(count, dtype=np.float64)
            countbuf = np.zeros(count, dtype=np.int16)
            compute_point(k1ind, k2ind, kcoords, fft,
                          nk1, nk2, shape, samp, count,
                          bispecbuf, binormbuf, countbuf)
            N = countbuf.sum()
            value = bispecbuf.sum()
            norm = binormbuf.sum()
            bispec[i, j], bispec[j, i] = value, value
            binorm[i, j], binorm[j, i] = norm, norm
            omega[i, j], omega[j, i] = nk1*nk2, nk1*nk2
            counts[i, j], counts[j, i] = N, N
        if progress:
            with nb.objmode():
                _printProgressBar(i, dim-1)
    return bispec, binorm, omega, counts


@nb.njit(parallel=True, cache=True)
def _compute_point3D(k1ind, k2ind, kcoords, fft, nk1, nk2, shape,
                     samp, count, bispecbuf, binormbuf, countbuf):
    kx, ky, kz = kcoords[0], kcoords[1], kcoords[2]
    Nx, Ny, Nz = shape[0], shape[1], shape[2]
    for idx in nb.prange(count):
        n, m = k1ind[samp[idx] % nk1], k2ind[samp[idx] // nk1]
        k1x, k1y, k1z = kx[n], ky[n], kz[n]
        k2x, k2y, k2z = kx[m], ky[m], kz[m]
        k3x, k3y, k3z = k1x+k2x, k1y+k2y, k1z+k2z
        if np.abs(k3x) > Nx//2 or np.abs(k3y) > Ny//2 or np.abs(k3z) > Nz//2:
            continue
        sample = fft[k1x, k1y, k1z]*fft[k2x, k2y, k2z]*np.conj(fft[k3x, k3y, k3z])
        bispecbuf[idx] = sample
        binormbuf[idx] = np.abs(sample)
        countbuf[idx] = 1


@nb.njit(parallel=True, cache=True)
def _compute_point2D(k1ind, k2ind, kcoords, fft, nk1, nk2, shape,
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


@nb.jit(forceobj=True, cache=True)
def _printProgressBar(iteration, total, prefix='', suffix='', decimals=1,
                      length=50, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar

    Adapted from
    https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    """
    prefix = '(%d/%d)' % (iteration, total) if prefix == '' else prefix
    percent = str("%."+str(decimals)+"f") % (100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    prog = '\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)
    print(prog, end=printEnd, flush=True)
    if iteration == total:
        print()


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    N = 512
    np.random.seed(1234)
    data = np.random.normal(size=N**2).reshape((N, N))+1

    kmin, kmax = 1, 100
    bispec, bicoh, kn, omega, counts = bispectrum(data, nsamples=None,
                                                  kmin=kmin, kmax=kmax,
                                                  progress=True,
                                                  mean_subtract=True,
                                                  full=True, bench=True)
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
                       interpolation="nearest",
                       cmap=cmap,
                       extent=[kmin, kmax, kmin, kmax])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label(labels[i])
        #if i == 0:
        #    ax.contour(data[i], colors='k', extent=[kmin, kmax, kmin, kmax])
        ax.set_xlabel(r"$k_1$")
        ax.set_ylabel(r"$k_2$")

    plt.tight_layout()

    plt.show()
