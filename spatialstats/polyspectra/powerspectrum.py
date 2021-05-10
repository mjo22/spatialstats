"""
Power spectrum CPU implementation.

.. moduleauthor:: Michael O'Brien <michaelobrien@g.harvard.edu>

"""

import numpy as np
from time import time


def powerspectrum(*U, average=True, diagnostics=False,
                  kmin=None, kmax=None, npts=None,
                  compute_fft=True, compute_sqr=True,
                  use_pyfftw=False, bench=False, **kwargs):
    """
    Returns the 1D radially averaged power spectrum :math:`P(k)`
    of a real scalar or vector field :math:`U`. Assuming statistical
    homogeneity and isotropy, this is defined as

    .. math::
        P(k) = \int\limits_{|\mathbf{k}| \in [k, \ k+\Delta k)} 
               d\mathbf{k} \ |\hat{U}(\mathbf{k})|^2,

    where :math:`\hat{U}` is the FFT of :math:`U`, :math:`\mathbf{k}`
    is a wavevector, :math:`k` is a scalar wavenumber, and :math:`\Delta k`
    is the radial spacing between bins.

    We approximate this integral as

    .. math::

        P(k) = \frac{V_k}{N_k} \sum\limits_{|\mathbf{k}| \in [k, k+\Delta k)} |\hat{U}(\mathbf{k})|^2,

    where :math:`V_k` is the volume of the :math:`k`th bin and :math:`N_k`
    is the number of points in the bin.

    Parameters
    ----------
    U : `np.ndarray`
        Scalar or vector field.
        If passing vector data, pass arguments as ``U1, U2, ..., Un``
        where ``Ui`` is the ith vector component.
        Each ``Ui`` can be 1D, 2D, or 3D, and all must have the
        same ``Ui.shape`` and ``Ui.dtype``.
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
        If ``None``, ``kmax = max(U.shape)//2``.
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
    use_pyfftw : `bool`, optional
        If ``True``, use ``pyfftw`` instead of ``np.fft``
        to compute FFTs.
    bench : `bool`, optional
        Print message for time of calculation.
    kwargs
        Additional keyword arguments passed to
        ``np.fft.fftn``, ``np.fft.rfftn``, ``pyfftw.builders.fftn``,
        or ``pyfftw.builders.rfftn``.

    Returns
    -------
    spectrum : `np.ndarray`, shape `(npts,)`
        Radially averaged power spectrum :math:`P(k)`.
    kn : `np.ndarray`, shape `(npts,)`
        Left edges :math:`k` of radial bins.
    std : `np.ndarray`, shape `(npts,)`, optional
        Sample standard deviation in each bin.
    counts : `np.ndarray`, shape `(npts,)`, optional
        Number of points in each bin.
    """
    if bench:
        t0 = time()

    ndim = U[0].ndim
    ncomp = len(U)
    N = max(U[0].shape)

    real = True if np.issubdtype(U[0].dtype, np.floating) else False

    if ndim not in [1, 2, 3]:
        raise ValueError("Dimension of image must be 1, 2, or 3.")

    # Compute FFT
    density = None
    for i in range(ncomp):
        comp = U[i]
        if compute_fft:
            # Compute fft of a component
            if use_pyfftw:
                fft = _fftn(comp, **kwargs)
            else:
                if real:
                    fft = np.fft.rfftn(comp, **kwargs)
                else:
                    fft = np.fft.fftn(comp, **kwargs)
        else:
            fft = comp
        # Compute amplitudes
        if density is None:
            fftshape = fft.shape
            density = np.zeros(fftshape)
        if compute_sqr:
            density[...] += np.real(fft*np.conj(fft))
        else:
            density[...] += np.real(fft)
        del fft

    # Need to double count if using rfftn
    if real:
        density[...] *= 2.

    # Compute radial coordinates
    kr = _kmag_sampling(fftshape, real=real)

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
    kn = np.linspace(kmin, kmax, npts, endpoint=True)  # Left edges of bins
    dk = kn[1] - kn[0]

    # Radially average power spectral density
    if ndim == 1:
        fac = 2*np.pi
    elif ndim == 2:
        fac = 4*np.pi
    elif ndim == 3:
        fac = 4./3.*np.pi
    spectrum = np.zeros_like(kn)
    std = np.zeros_like(kn)
    counts = np.zeros(kn.shape, dtype=np.int64)
    for i, ki in enumerate(kn):
        ii = np.where(np.logical_and(kr >= ki, kr < ki+dk))
        samples = density[ii]
        if average:
            dv = fac*np.pi*((ki+dk)**ndim-(ki)**ndim)
            spectrum[i] = dv*np.mean(samples)
        else:
            spectrum[i] = np.sum(samples)
        counts[i] = samples.size
        std[i] = np.std(samples)

    del density, kr

    if bench:
        print(f"Time: {time() - t0:.04f} s")

    result = [spectrum, kn]
    if diagnostics:
        result.extend([std, counts])

    return tuple(result)


def _fftn(image, overwrite_input=False, threads=-1, **kwargs):
    """
    Calculate N-dimensional fft of image with pyfftw.
    See pyfftw.builders.fftn for kwargs documentation.
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


def _kmag_sampling(shape, real=True):
    """
    Samples the |k| coordinate system.
    """
    if real:
        freq = np.fft.rfftfreq
        s = list(shape)
        s[-1] = (s[-1]-1)*2
        shape = tuple(s)
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

    psd, kn = powerspectrum(data)

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

    plt.plot(log_kn, log_psd,
             label=rf'PSD, $\beta = {fc.beta}$', color='g')
    plt.plot(log_kn[idxs], m*log_kn[idxs]+b,
             label=rf'Fit, $\beta = {m}$', color='k')
    plt.ylabel(r"$\log{P(k)}$")
    plt.xlabel(r"$\log{k}$")
    plt.legend(loc='upper right')

    plt.show()
