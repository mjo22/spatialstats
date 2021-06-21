"""
Implementation using CuPy acceleration.

.. moduleauthor:: Michael O'Brien <michaelobrien@g.harvard.edu>

"""

import numpy as np
from time import time
import cupy as cp
from cupyx.scipy import fft as cufft


def powerspectrum(*u, average=True, diagnostics=False,
                  kmin=None, kmax=None, npts=None,
                  compute_fft=True, compute_sqr=True,
                  double=True, bench=False, **kwargs):
    """
    See the documentation for the :ref:`CPU version<powerspectrum>`.

    Parameters
    ----------
    u : `np.ndarray`
        Scalar or vector field.
        If vector data, pass arguments as ``u1, u2, ..., un``
        where ``ui`` is the ith vector component.
        Each ``ui`` can be 1D, 2D, or 3D, and all must have the
        same ``ui.shape`` and ``ui.dtype``.
    average : `bool`, optional
        If ``True``, average over values in a given
        bin and multiply by the bin volume.
        If ``False``, compute the sum.
    diagnostics : `bool`, optional
        Return the standard deviation and number of points
        in a particular radial bin.
    kmin : `int` or `float`, optional
        Minimum wavenumber in power spectrum bins.
        If ``None``, ``kmin = 1``.
    kmax : `int` or `float`, optional
        Maximum wavenumber in power spectrum bins.
        If ``None``, ``kmax = max(u.shape)//2``.
    npts : `int`, optional
        Number of modes between ``kmin`` and ``kmax``,
        inclusive.
        If ``None``, ``npts = kmax-kmin+1``.
    compute_fft : `bool`, optional
        If ``False``, do not take the FFT of the input data.
        FFTs should not be passed with the zero-frequency
        component in the center.
    compute_sqr : `bool`, optional
        If ``False``, sum the real part of the FFT. This can be
        useful for purely real FFTs, where the sign of the
        FFT is useful information. If ``True``, take the square
        as usual.
    double : `bool`, optional
        If ``False``, calculate FFTs in single precision.
        Useful for saving memory.
    bench : `bool`, optional
        Print message for time of calculation.
    kwargs
        Additional keyword arguments passed to
        ``cupyx.scipy.fft.fftn`` or ``cupyx.scipy.fft.rfftn``.

    Returns
    -------
    spectrum : `np.ndarray`, shape `(npts,)`
        Radially averaged power spectrum :math:`P(k)`.
    kn : `np.ndarray`, shape `(npts,)`
        Left edges of radial bins :math:`k`.
    counts : `np.ndarray`, shape `(npts,)`, optional
        Number of points :math:`N_k` in each bin.
    vol : `np.ndarray`, shape `(npts,)`, optional
        Volume :math:`V_k` of each bin.
    stdev : `np.ndarray`, shape `(npts,)`, optional
        Standard deviation multiplied with :math:`V_k`
        in each bin.
    """
    if bench:
        t0 = time()

    shape = u[0].shape
    ndim = u[0].ndim
    ncomp = len(u)
    N = max(u[0].shape)

    if np.issubdtype(u[0].dtype, np.floating):
        real = True
        dtype = cp.float64 if double else cp.float32
    else:
        real = False
        dtype = cp.complex128 if double else cp.complex64

    if ndim not in [1, 2, 3]:
        raise ValueError("Dimension of image must be 1, 2, or 3.")

    # Get memory pools
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()

    # Compute pqower spectral density with memory efficiency
    density = None
    comp = cp.empty(shape, dtype=dtype)
    for i in range(ncomp):
        temp = cp.asarray(u[i], dtype=dtype)
        comp[...] = temp
        del temp
        if compute_fft:
            fft = _cufftn(comp, **kwargs)
        else:
            fft = comp
        if density is None:
            fftshape = fft.shape
            density = cp.zeros(fft.shape)
        if compute_sqr:
            density[...] += _mod_squared(fft)
        else:
            density[...] += cp.real(fft)
        del fft
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

    # Need to double count if using rfftn
    if real and compute_fft:
        density[...] *= 2

    # Get radial coordinates
    kr = cp.asarray(_kmag_sampling(fftshape, real=real).astype(np.float32))

    # Flatten arrays
    kr = kr.ravel()
    density = density.ravel()

    # Get minimum and maximum k for binning if not given
    if kmin is None:
        kmin = 1
    if kmax is None:
        kmax = int(N/2)
    if npts is None:
        npts = kmax-kmin+1

    # Generate bins
    kn = cp.linspace(kmin, kmax, npts, endpoint=True)  # Left edges of bins
    dk = kn[1] - kn[0]

    # Radially average power spectral density
    if ndim == 1:
        fac = 2*np.pi
    elif ndim == 2:
        fac = 4*np.pi
    elif ndim == 3:
        fac = 4./3.*np.pi
    spectrum = cp.zeros_like(kn)
    stdev = cp.zeros_like(kn)
    vol = cp.zeros_like(kn)
    counts = cp.zeros(kn.shape, dtype=np.int64)
    for i, ki in enumerate(kn):
        ii = cp.where(cp.logical_and(kr >= ki, kr < ki+dk))
        samples = density[ii]
        vk = fac*cp.pi*((ki+dk)**ndim-(ki)**ndim)
        if average:
            spectrum[i] = vk*cp.mean(samples)
        else:
            spectrum[i] = cp.sum(samples)
        if diagnostics:
            Nk = samples.size
            stdev[i] = vk * cp.std(samples, ddof=1)
            vol[i] = vk
            counts[i] = Nk

    del density, kr
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()

    if bench:
        print(f"Time: {time() - t0:.04f} s")

    result = [spectrum.get(), kn.get()]
    if diagnostics:
        result.extend([counts.get(), vol.get(), stdev.get()])

    return tuple(result)


def _cufftn(data, overwrite_input=False, **kwargs):
    """
    Calculate the N-dimensional fft of an image
    with memory efficiency
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
def _mod_squared(a):
    return cp.real(a*cp.conj(a))


def _kmag_sampling(shape, real=True):
    """
    Generates the |k| coordinate system.
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

    psd, kn, stdev, vol, N = powerspectrum(data, diagnostics=True)

    print(psd.mean())

    def zero_log10(s):
        """
        Takes logarithm of an array while retaining the zeros
        """
        sp = np.where(s > 0., s, 1)
        return np.log10(sp)

    log_psd = zero_log10(psd)
    log_kn = zero_log10(kn)
    idxs = np.where(log_kn >= np.log10(fc.kmin))
    m, b = np.polyfit(log_kn[idxs], log_psd[idxs], 1)

    plt.errorbar(kn, psd,
                 label=rf'PSD, $\beta = {fc.beta}$', color='g')
    plt.plot(log_kn[idxs], m*log_kn[idxs]+b,
             label=rf'Fit, $\beta = {m}$', color='k')
    plt.ylabel(r"$\log{P(k)}$")
    plt.xlabel(r"$\log{k}$")
    plt.legend(loc='upper right')

    plt.show()
