"""
FFT estimator for power spectrum multipoles with respect
to a local orientation vectors or polarity field.

.. moduleauthor:: Michael O'Brien <michaelobrien@g.harvard.edu>

"""

import numpy as np
import sympy as sp
from finufft import nufft3d1
import pyfftw


def nufftpower(positions, orientations, weights,
               boxsize, modes, poles=0, **kwargs):
    r"""
    .. _nufftpower:

    Compute power spectrum multipoles with respect to local
    orientation vectors using non-uniform FFTS in three-dimensions.

    The non-uniform FFT transforms a field defined by

    .. math::
        \mathbf{w}(\mathbf{x}) = \sum_i \mathbf{w}_i \delta(\mathbf{x} - \mathbf{x}_i),

    with a corresponding polarity field

    .. math::
        \mathbf{p}(\mathbf{x}) = \sum_i \mathbf{p}_i \delta(\mathbf{x} - \mathbf{x}_i).

    The estimator for multipoles with respect to :math:`\mathbf{p}_i\cdot\hat{\mathbf{k}}`
    then is

    .. math::
        \mathcal{P}_{\ell}(k) = \frac{1}{V_k} \sum_{|\mathbf{k}| \in [k, k+1)} \sum_m Y_{\ell m}(\hat{\mathbf{k}}) (\textrm{FFT}[\mathbf{w}(\mathbf{x})] \cdot \textrm{FFT}[\mathbf{w}(\mathbf{x}) Y_{\ell m}(\mathbf{p}(\mathbf{x}))]^*),

    where :math:`Y_{\ell m}` are the real spherical harmonics, :math:`V_k` is a bin volume,
    :math:`\mathbf{w}_i` are weights, and :math:`\mathbf{p}_i` are orientations at positions :math:`\mathbf{x}_i`.

    The above is defined for a vector field :math:`\mathbf{w}(\mathbf{x})`,
    but the implementation is general for scalar and tensor fields.

    Parameters
    ----------
    positions : `np.ndarray`, shape `(N, 3)`
        Position vectors :math:`\mathbf{x}_i`.
    orientations : `np.ndarray`, shape `(N, 3)`
        Orientation vectors :math:`\mathbf{p}_i`.
    weights : `np.ndarray`, shape `(N, 3, ..., 3)`
        Scalar, vector, or tensor weights :math:`\mathbf{w}_i`
        or field :math:`\mathbf{w}(\mathbf{x})`.
        For example, taking :math:`w_i^{mn} = p_i^m p_i^n` computes
        the nematic order power spectrum.
        If ``weights = 1.``, computes
        the power spectrum of the distribution function
        :math:`g(\mathbf{x}) = \sum_i \delta(\mathbf{x} - \mathbf{x}_i)`.
    boxsize : `int` or `list`
        Box size of particle data.
    modes : `int` or `list`
        Number of modes in each dimension in Fourier space.
    poles : `int` or `list`, optional
        Multipole indices :math:`\ell`.
    kwargs
        Additional keyword arguments passed to ``finufft.nufft3d1``

    Returns
    -------
    spectra : list of `np.ndarray`, shape `(nk,)`
        Multipoles :math:`\mathcal{P}_{\ell}(k)`.
    k : `np.ndarray`, shape `(nk,)`
        Wavenumbers :math:`k`, where ``kmax = int(max(modes)/2)``
        and ``nk = kmax+1``. The zero mode is returned, so
        ``k = np.linspace(0, kmax, nk, endpoint=True)``.
    Nk : `np.ndarray`, shape `(nk,)`
        Number of points in a wavenumber bin :math:`N_k` with shell thickness
        :math:`[k, k+1)`.
    Vk : `np.ndarray`, shape `(nk,)`
        Volume of a wavenumber bin :math:`V_k`.
    """
    # Attributes
    N = positions.shape[0]
    ndim = 3
    scalar = True if weights.ndim == 1 else False

    # Modify args
    boxsize = ndim*[boxsize] if type(boxsize) is int else boxsize
    modes = tuple(ndim*[modes]) if type(modes) is int else tuple(modes)
    poles = [poles] if type(poles) is int else poles

    # Allow user to pass no weights
    if float(weights) == 1. or weights is None:
        weights = np.ones(N)

    if orientations.shape[0] != N or weights.shape[0] != N:
        raise ValueError("Number of particles in xi, pi, and wi not consistent.")

    # Flatten tensor input
    if weights.ndim > 2:
        tensor_shape = weights.shape[1:]
        weights = weights.reshape((N, np.prod(tensor_shape)))

    # Add modes to kwarg dict
    kwargs["n_modes"] = modes

    # First set of FFTs
    xi = 2*np.pi*positions/boxsize
    pi = [orientations[..., i] for i in range(ndim)]
    wi = weights.astype(np.complex128)
    ffts1 = nufft3d1(*xi.T, wi.T, modes, **kwargs)

    # Spherical harmonics and k grid
    Ylms = _compute_spherical_harmonics(poles)
    kmag, khat = _k_grid(modes)

    # Package args
    args = (nufft3d1, wi.T, pi, kmag, khat, Ylms, poles,
            ffts1, scalar, *xi.T)

    # Compute multipoles
    spectra, k, Nk, Vk = _compute_multipoles(*args, **kwargs)

    return spectra, k, Nk, Vk


def fftpower(field, polarity, poles=0, **kwargs):
    r"""
    See the documentation for
    :ref:`spatialstats.polyspectra.nufftpower <nufftpower>`.

    This computes power spectrum multipoles for a scalar, vector,
    or tensor field :math:`\mathbf{w}(\mathbf{x})` with respect to a polarity field
    :math:`\mathbf{p}(\mathbf{x})`.

    Parameters
    ----------
    field : `np.ndarray`, shape `(Nx, Ny, Nz, 3, ..., 3)`
        Scalar, vector, or tensor field :math:`\mathbf{w}(\mathbf{x})`.
    polarity : `np.ndarray`, shape `(Nx, Ny, Nz, 3)` or `(3,)`
        Polarity field :math:`\mathbf{p}(\mathbf{x})`. Can also pass
        one direction :math:`\mathbf{p}`.
    poles : `int` or `list`, optional
        Multipole indices :math:`\ell`.
    kwargs
        Additional keyword arguments passed to ``pyfftw.FFTW``.

    Returns
    -------
    spectra : list of `np.ndarray`, shape `(nk,)`
        Multipoles :math:`\mathcal{P}_{\ell}(k)`.
    k : `np.ndarray`, shape `(nk,)`
        Wavenumbers :math:`k`, where ``kmax = int(max(modes)/2)``
        and ``nk = kmax+1``. The zero mode is returned, so
        ``k = np.linspace(0, kmax, nk, endpoint=True)``.
    Nk : `np.ndarray`, shape `(nk,)`
        Number of points in a wavenumber bin :math:`N_k` with shell thickness
        :math:`[k, k+1)`.
    Vk : `np.ndarray`, shape `(nk,)`
        Volume of a wavenumber bin :math:`V_k`.
    """
    # Attributes
    ndim = 3
    shape = field.shape[0:ndim]
    scalar = True if field.ndim == ndim else False

    # Modify args
    poles = [poles] if type(poles) is int else poles

    if polarity.shape[0:ndim] != modes:
        raise ValueError("Shape of p(x) and w(x) are not consistent.")

    # Flatten tensor input and reshape
    if field.ndim > ndim:
        tensor_shape = field.shape[ndim:]
        field = np.transpose(field.reshape((*shape, np.prod(tensor_shape))),
                             axes=(3, 0, 1, 2))

    # First set of FFTs
    w = field
    p = tuple([polarity[..., i] for i in range(ndim)])
    ffts1 = _fft3d(w, **kwargs)

    # Spherical harmonics and k grid
    Ylms = _compute_spherical_harmonics(poles)
    kmag, khat = _k_grid(modes)

    # Package args
    args = (_fft3d, w, p, kmag, khat, Ylms, poles,
            ffts1, scalar)

    # Compute multipoles
    spectra, k, Nk, Vk = _compute_multipoles(*args, **kwargs)

    return spectra, k, Nk, Vk


def _compute_multipoles(FFT, w, p, kmag, khat, Ylms, poles,
                        ffts1, shape, scalar, *args, **kwargs):
    """Main loop for mulipole computation."""
    npoles = len(poles)
    Pl = np.zeros(shape, dtype=np.complex128)
    spectra = []
    for idx in range(npoles):
        l = poles[idx]
        for jdx in range(0, 2*l+1):
            Ylm = Ylms[idx][jdx]
            ffts2 = np.conj(FFT(*args, w*Ylm(*p), **kwargs))
            if scalar:
                Pl[...] += Ylm(*khat) * ffts1 * ffts2
            else:
                Pl[...] += Ylm(*khat) * np.einsum('ijkl,ijkl->jkl', ffts1, ffts2)
        spectrum, k, Nk, vk = _spherical_binning(Pl, kmag, modes)
        spectra.append(spectrum)
        Pl[...] = 0

    return spectra, k, Nk, vk


def _fft3d(field, threads=-1, **kwargs):
    """
    Compute FFT with pyfftw.

    Modified from
    https://github.com/oliverphilcox/Spectra-Without-Windows/blob/main/src/opt_utilities.py.
    """
    # Align
    shape = field.shape
    a_in = pyfftw.empty_aligned(shape, dtype='complex128')
    a_out = pyfftw.empty_aligned(shape, dtype='complex128')

    # Plan FFTW
    axes = (0, 1, 2) if field.ndim == 3 else (1, 2, 3)
    fftw_plan = pyfftw.FFTW(a_in, a_out, axes=axes,
                            flags=('FFTW_ESTIMATE',),
                            direction='FFTW_FORWARD',
                            threads=threads, **kwargs)

    # Perform FFTW
    a_in[:] = field
    fftw_plan(a_in, a_out)

    return a_out


def _k_grid(shape):
    """Create a k coordinate system."""
    ndim = len(shape)
    kcoords1D = []
    for i in range(ndim):
        ni = shape[i]
        sample = np.fft.fftfreq(ni)
        kcoords1D.append(sample * ni)
    kr = np.meshgrid(*kcoords1D, indexing='ij')
    kmag = np.zeros_like(kr[0])
    for i in range(ndim):
        kmag[...] += kr[i]**2
    kmag[...] = np.sqrt(kmag)
    khat = tuple([np.divide(kr[i], kmag, where=kmag!=0) for i in range(ndim)])
    return kmag, khat


def _spherical_binning(density, kmag, modes):
    """Radially average power spectral density."""
    kmin, kmax = 0, int(max(modes) / 2)
    kn = np.linspace(kmin, kmax, kmax-kmin+1, endpoint=True)  # Left edges of bins
    dk = kn[1] - kn[0]

    spectrum = np.zeros(kn.size, dtype=np.complex128)
    vol = np.zeros_like(kn)
    counts = np.zeros(kn.shape, dtype=np.int64)

    for i, ki in enumerate(kn):
        ii = np.where(np.logical_and(kmag >= ki, kmag < ki+dk))
        samples = density[ii]
        vk = 4./3.*np.pi*((ki+dk)**3-(ki)**3)
        Nk = samples.size
        spectrum[i] = np.sum(samples) / Nk
        Nk = samples.size
        vol[i] = vk
        counts[i] = Nk

    return spectrum, kn, counts, vol


def _compute_spherical_harmonics(l):
    """Compute array of valid spherical harmonic functions."""
    ls = [l] if l is int else l

    Y_lm_out = []
    for l in ls:
        Y_m_out = []
        for m in np.arange(-l, l+1, 1):
            Y_m_out.append(get_real_Ylm(l, m))
        Y_lm_out.append(np.asarray(Y_m_out))
    return Y_lm_out


def _get_real_Ylm(l, m):
    """
    Return a function that computes the real spherical harmonic of order (l,m).

    Modified from
    https://github.com/oliverphilcox/Spectra-Without-Windows/blob/main/src/opt_utilities.py.
    """
    # make sure l, m are integers
    l = int(l)
    m = int(m)

    # the relevant cartesian and spherical symbols
    x, y, z, r = sp.symbols('x y z r', real=True, positive=True)
    xhat, yhat, zhat = sp.symbols('xhat yhat zhat', real=True, positive=True)
    phi, theta = sp.symbols('phi theta')
    defs = [(sp.sin(phi), y / sp.sqrt(x**2 + y**2)),
            (sp.cos(phi), x / sp.sqrt(x**2 + y**2)),
            (sp.cos(theta), z / sp.sqrt(x**2 + y**2 + z**2))]

    # the normalization factors
    if m == 0:
        amp = sp.sqrt((2 * l + 1) / (4 * np.pi))
    else:
        amp = sp.sqrt(2 * (2 * l + 1) / (4 * np.pi) *
                      sp.factorial(l - abs(m)) / sp.factorial(l + abs(m)))

    # the cos(theta) dependence encoded by the associated Legendre poly
    expr = (-1)**m * sp.assoc_legendre(l, abs(m), sp.cos(theta))

    # the phi dependence
    if m < 0:
        expr *= sp.expand_trig(sp.sin(abs(m) * phi))
    elif m > 0:
        expr *= sp.expand_trig(sp.cos(m * phi))

    # simplify
    expr = sp.together(expr.subs(defs)).subs(x**2 + y**2 + z**2, r**2)
    expr = amp * expr.expand().subs([(x / r, xhat), (y / r, yhat),
                                     (z / r, zhat)])
    Ylm = sp.lambdify((xhat, yhat, zhat), expr, modules='numexpr')

    # attach some meta-data
    Ylm.expr = expr
    Ylm.l = l
    Ylm.m = m

    return Ylm


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    N = 1000000
    L = 50
    positions = L * np.random.rand(N, 3)
    orientations = np.random.rand(N, 3)
    orientations /= np.linalg.norm(orientations, axis=1)[:, np.newaxis]
    nematic = np.einsum('ki,kj->kij', orientations, orientations)
    ls = [0, 2, 4]
    boxsize = 3*[L]
    modes = 3*[40]

    spectra, k, Nk, vk = powerspectrum(positions, orientations, nematic,
                                       boxsize, modes, poles=ls)

    for idx, l in enumerate(ls):
        spectrum = spectra[idx]
        print(spectrum.real.mean(), spectrum.imag.mean())
        plt.plot(2*np.pi*k/L, np.abs(spectrum))
        plt.title(rf"$\ell = {l}$")
        plt.show()
