"""
Routines to calculate power spectrum on scalar and vector image data
using CuPy acceleration.

This implementation works on 1D, 2D, and 3D rectangular domains for real
or complex valued data.

Author:
    Michael O'Brien (2020)
    Biophysical Modeling Group
    Center for Computational Biology
    Flatiron Institute
"""


import numpy as np
from time import time
import cupy as cp
from cupyx.scipy import fft as cufft


def powerspectrum(data, vector=False, real=True, average=False,
                  kmin=None, kmax=None, npts=None,
                  compute_fft=True, compute_sqr=True, bench=False,
                  **kwargs):
    """
    Returns the radially averaged power spectrum
    of real signal on 2 or 3 dimensional scalar or
    vector data using CuPy acceleration.

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
        Specify whether user has passed scalar or 
        vector data.
    kmin : float or int
        Minimum k in powerspectrum bins. If None,
        use 1.
    kmax : float or int
        Maximum k in powerspectrum bins. If None,
        use highest mode from FFT.
    npts : int
        Number of modes between [kmin, kmax]
    average : bool
        If True, average over values in a given
        bin. If False, add values. 
    bench : bool
        Print message for time of calculation

    **kwargs are passed to cufftn (defined below)

    Returns
    -------
    spectrum : np.ndarray
        Radially averaged power spectrum
    kn : np.ndarray
        Corresponding bins for spectrum. Same
        size as spectrum.
    """
    if bench:
        t0 = time()

    if vector:
        shape = data[0].shape
        ndim = data[0].ndim
        ncomp = data.shape[0]
        N = max(data[0].shape)
    else:
        shape = data.shape
        ndim = data.ndim
        ncomp = 1
        N = max(data.shape)

    if real:
        dtype = np.float64
    else:
        dtype = np.complex128

    if ndim not in [1, 2, 3]:
        raise ValueError("Dimension of image must be 2 or 3.")

    # Get memory pools
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()

    # Compute power spectral density with memory efficiency
    density = None
    comp = cp.empty(shape, dtype=dtype)
    norm = np.float64(comp.size)
    for i in range(ncomp):
        temp = cp.asarray(data[i]) if vector else cp.asarray(data)
        comp[...] = temp
        if compute_fft:
            fft = cufftn(comp, **kwargs)
        else:
            fft = comp
        if density is None:
            fftshape = fft.shape
            density = cp.zeros(fft.shape)
        if compute_sqr:
            density[...] += mod_squared(fft)
        else:
            density[...] += np.real(fft)
        del fft, temp
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

    # Normalize FFT
    fac = 2. if real else 1.
    if compute_sqr:
        density[...] *= fac/norm**2
    else:
        density[...] *= fac/norm

    # Get radial coordinates
    kr = cp.asarray(kmag_sampling(fftshape, real=real).astype(np.float32))

    # Flatten arrays
    kr = kr.ravel()
    density = density.ravel()

    # Get minimum and maximum k for binning if not given
    if kmin is None:
        kmin = 1
    if kmax is None:
        kmax = int(N/2)
    if npts is None:
        npts = kmax - kmin

    # Generate bins
    kn = cp.linspace(kmin, kmax, npts, endpoint=False)  # Left edges of bins
    dk = kn[1] - kn[0]
    kn += dk/2  # Convert kn to bin centers.

    # Radially average power spectral density
    if ndim == 1:
        fac = 2*np.pi
    elif ndim == 2:
        fac = 4*np.pi
    elif ndim == 3:
        fac = 4./3.*np.pi
    spectrum = cp.zeros_like(kn)
    for i, ki in enumerate(kn):
        ii = cp.where(np.logical_and(kr >= ki-dk/2, kr < ki+dk/2))
        if average:
            dv = fac*cp.pi*((ki+dk/2)**ndim-(ki-dk/2)**ndim)
            spectrum[i] = dv*cp.mean(density[ii])
        else:
            spectrum[i] = cp.sum(density[ii])

    spectrum = cp.asnumpy(spectrum)
    kn = cp.asnumpy(kn)

    del density, kr
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()

    if bench:
        print(f"Time: {time() - t0:.04f} s")

    return spectrum, kn


def cufftn(data, overwrite_input=False, **kwargs):
    """
    Calculate the N-dimensional fft of an image
    with memory efficiency

    Parameters
    ----------
    data : cupy.ndarray
        Real or complex valued 2D or 3D image

    Keywords
    --------
    complex : cupy.dtype
        Specify single or double precision fft
    overwrite_input : bool
        Specify whether input data can be destroyed.
        This is useful if low on memory.
        See cupyx.scipy.fft.fftn for more.

    **kwargs passes to cupyx.scipy.fft.fftn

    Returns
    -------
    fft : cupy.ndarray
        The fft. Will be the shape of the input image
        or the user specified shape.
    """
    # Get memory pools
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()

    # Real vs. Complex data
    if data.dtype in [cp.float32, cp.float64]:
        value_type = 'R2C'
        fftn = cufft.rfftn
    elif data.dtype in [cp.complex64, cp.complex128]:
        value_type = 'C2C'
        fftn = cufft.fftn
    else:
        raise ValueError(f"{data.dtype} is unrecognized data type.")

    # Get plan for computing fft
    plan = cufft.get_fft_plan(data, value_type=value_type)

    # Compute fft
    with plan:
        fft = fftn(data, overwrite_x=overwrite_input, **kwargs)

    # Release memory
    del plan
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()

    return fft


@cp.fuse(kernel_name='mod_squared')
def mod_squared(a):
    return cp.real(a*cp.conj(a))


def kmag_sampling(shape, real=True):
    """
    Generates the |k| coordinate system.

    Parameters
    ----------
    shape : tuple
        Shape of 1D, 2D, or 3D FFT

    Returns
    -------
    kmag : np.ndarray
        Samples of k vector magnitudes on coordinate
        system of size shape
    """
    if real:
        freq = np.fft.rfftfreq
        s = list(shape)
        s[-1] = (s[-1]-1)*2
        shape = s
    else:
        freq = np.fft.fftfreq
    ndim = len(shape)
    kmag = np.zeros(shape)
    ksqr = []
    for i in range(ndim):
        ni = shape[i]
        sample = freq(ni) if i == ndim - 1 else np.fft.fftfreq(ni)
        if real:
            sample = np.abs(sample)
        k1d = sample * ni
        ksqr.append(k1d * k1d)

    if ndim == 1:
        ksqr = ksqr[0]
    elif ndim == 2:
        ksqr = np.add.outer(ksqr[0], ksqr[1])
    elif ndim == 3:
        ksqr = np.add.outer(np.add.outer(ksqr[0], ksqr[1]), ksqr[2])

    kmag = np.sqrt(ksqr)

    return kmag


if __name__ == '__main__':
    import pyFC
    from matplotlib import pyplot as plt

    dim = 100
    fc = pyFC.LogNormalFractalCube(
        ni=dim, nj=dim, nk=dim, kmin=10, mean=1, beta=-5/3)
    fc.gen_cube()
    data = fc.cube

    # psdFC = fc.iso_power_spec()
    psd, kn = powerspectrum(data, real=True, bench=True, average=True)

    print(psd.mean())

    def zero_log10(s):
        '''
        Takes logarithm of an array while retaining the zeros
        '''
        sp = np.where(s > 0., s, 1)
        return np.log10(sp)

    log_psd = zero_log10(psd)
    log_kn = zero_log10(kn)
    idxs = np.where(log_kn >= np.log10(fc.kmin))
    m, b = np.polyfit(log_kn[idxs], log_psd[idxs], 1)

    plt.plot(log_kn, log_psd,
             label=rf'PSD, $\beta = {fc.beta}$', color='g')
    plt.plot(log_kn[idxs], m*log_kn[idxs]+b,
             label=rf'Fit, $\beta = {m}$', color='k')
    plt.ylabel(r"$\log{P(k)}$")
    plt.xlabel(r"$\log{k}$")
    plt.legend(loc='upper right')

    plt.show()
