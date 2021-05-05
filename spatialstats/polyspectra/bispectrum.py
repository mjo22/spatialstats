"""
Bispectrum CPU implementation using Numba parallelization.

.. moduleauthor:: Michael O'Brien <michaelobrien@g.harvard.edu>

"""

import numpy as np
import numba as nb
from time import time


def bispectrum(*U, kmin=None, kmax=None, theta=None,
               nsamples=None, sample_thresh=None,
               exclude_upper=False, mean_subtract=False,
               compute_fft=True, diagnostics=False,
               use_pyfftw=False,
               bench=False, progress=False, **kwargs):
    """
    Compute the bispectrum :math:`B(k_1, k_2, \\theta)` and
    bicoherence index :math:`b(k_1, k_2, \\theta)` of a 2D or 3D
    real or complex-valued scalar or vector field :math:`U` by
    directly sampling triangles formed by wavevectors with sides
    :math:`\mathbf{k_1}` and :math:`\mathbf{k_2}` and averaging
    :math:`\hat{U}(\mathbf{k_1})\hat{U}(\mathbf{k_2})\hat{U}(\mathbf{k_1+k_2})`,
    where :math:`\hat{U}` is the FFT of :math:`U`.

    The implementation bins together
    triangles formed by wavevectors with constant wavenumber side lengths
    :math:`k_1` and :math:`k_2`, and
    it can return bispectra either binned by or summed over triangle angle
    :math:`\\theta`.

    :math:`b(k_1, k_2, \\theta)` is computed as
    :math:`|B(k_1, k_2, \\theta)|` divided by the sum over
    :math:`|\hat{U}(\mathbf{k_1})\hat{U}(\mathbf{k_2})\hat{U}(\mathbf{k_1+k_2})|`.

    .. note::
        This implementation returns an average over triangles,
        rather than a sum over triangles. One can recover the
        sum over triangles by multiplying ``counts * B``
        when ``nsamples = None``. Or, if ``theta = None``,
        evaulate ``omega * B``.

    .. note::
        When considering the bispectrum as a function of triangle
        angle, mesh points may be set to ``np.nan`` depending on
        :math:`k_1, \ k_2`. For example, a triangle angle of zero
        would yield a bispectrum equal to ``np.nan`` for all
        :math:`k_1 + k_2 > k_{nyq}`, where :math:`k_{nyq}` is the
        Nyquist frequency.
        Computing a boolean mask with ``np.isnan`` locates nan values
        in the result, and functions like ``np.nansum`` can be useful
        for reductions.

    .. note::
        Summing ``np.nansum(B, axis=0)`` recovers the
        bispectrum summed over triangle angles. To recover the
        bicoherence summed over triangle angles, evaulate
        ``np.nansum(B, axis=0) / np.nansum(np.abs(B)/b, axis=0)``

    Parameters
    ----------
    U : `np.ndarray`
        Real or complex vector or scalar data.
        If vector data, pass arguments as ``U1, U2, ..., Un``,
        where ``Ui`` is the ith vector component.
        Each ``Ui`` can be 2D or 3D, and all must have the
        same ``Ui.shape`` and ``Ui.dtype``. The vector
        bispectrum will be computed as the sum over bispectra
        of each component.
    kmin : `int`, optional
        Minimum wavenumber in bispectrum calculation.
        If ``None``, ``kmin = 1``.
    kmax : `int`, optional
        Maximum wavenumber in bispectrum calculation.
        If ``None``, ``kmax = max(U.shape)//2``.
    theta : `np.ndarray`, shape `(m,)`, optional
        Left edges of angular bins :math:`\\theta` between triangles
        formed by wavevectors :math:`\mathbf{k_1}, \ \mathbf{k_2}`.
        Values range between :math:`0` and :math:`\\pi`. If ``None``,
        sum over all triangle angles. Otherwise, return a bispectrum
        for each angular bin.
    nsamples : `int`, `float` or `np.ndarray`, shape `(kmax-kmin+1, kmax-kmin+1)`, optional
        Number of sample triangles or fraction of total
        possible triangles. This may be an array that
        specifies for a given :math:`k_1, \ k_2`.
        If ``None``, calculate the bispectrum exactly.
    sample_thresh : `int`, optional
        When the size of the sample space is greater than
        this number, start to use sampling instead of exact
        calculation. If ``None``, switch to exact calculation
        when ``nsamples`` is less than the size of the sample space.
    exclude_upper : `bool`, optional
        If ``True``, exclude the upper triangular part of the
        bispectrum. More specifically, points where
        :math:`k_1 + k_2` is greater than the Nyquist frequency.
        Excluded points will be set to ``np.nan``. This keyword
        has no effect when ``theta is not None``.
    mean_subtract : `bool`, optional
        Subtract mean from input data to highlight
        off-axis components in bicoherence.
    compute_fft : `bool`, optional
        If ``False``, do not take the FFT of the input data.
        FFTs should not be passed with the zero-frequency
        component in the center.
    diagnostics : `bool`, optional
        Return the optional sampling diagnostics,
        documented below.
    use_pyfftw : `bool`, optional
        If True, use ``pyfftw`` to compute the FFTs.
    bench : `bool`, optional
        If True, print calculation time.
    progress : `bool`, optional
        Print progress bar of calculation.
    kwargs
        Additional keyword arguments passed to
        ``np.fft.fftn`` or ``pyfftw.builders.fftn``.

    Returns
    -------
    B : `np.ndarray`, shape `(m, kmax-kmin+1, kmax-kmin+1)`
        Real or complex-valued bispectrum :math:`B(k_1, k_2, \\theta)`.
        Will be real-valued if the input data is real.
    b : `np.ndarray`, shape `(m, kmax-kmin+1, kmax-kmin+1)`
        Real-valued bicoherence index :math:`b(k_1, k_2, \\theta)`.
    kn : `np.ndarray`, shape `(kmax-kmin+1,)`
        Wavenumbers :math:`k_1` or :math:`k_2` along axis of bispectrum.
    theta : `np.ndarray`, shape `(m,)`, optional
        Left edges of angular bins :math:`\\theta`, ranging from
        :math:`0` to :math:`\\pi`. This is the same as the input
        ``theta`` and is returned for serialization convenience.
    omega : `np.ndarray`, shape `(kmax-kmin+1, kmax-kmin+1)`, optional
        Number of possible triangles in the sample space
        for a particular :math:`k_1, \ k_2`.
    counts : `np.ndarray`, shape `(m, kmax-kmin+1, kmax-kmin+1)`, optional
        Number of evaluations in the bispectrum sum.
    """
    shape, ndim = nb.typed.List(U[0].shape), U[0].ndim
    ncomp = len(U)

    if ndim not in [2, 3]:
        raise ValueError("Data must be 2D or 3D.")

    # Geometry of output image
    kmax = int(max(shape)/2) if kmax is None else int(kmax)
    kmin = 1 if kmin is None else int(kmin)
    kn = np.arange(kmin, kmax+1, 1, dtype=int)
    dim = kn.size
    # ...make costheta monotonically increase
    costheta = np.array([0.]) if theta is None else np.flip(np.cos(theta))

    # theta = 0 should be included
    if theta is not None:
        costheta[-1] += 1e-4

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
    ffts = []
    for i in range(ncomp):
        if compute_fft:
            temp = U[i] - U[i].mean() if mean_subtract else U[i]
            if use_pyfftw:
                fft = _fftn(temp, **kwargs)
            else:
                fft = np.fft.fftn(temp, **kwargs)
            del temp
        else:
            fft = U[i]
        ffts.append(fft)

    # Sampling settings
    if sample_thresh is None:
        sample_thresh = np.iinfo(np.int64).max
    if nsamples is None:
        nsamples = np.iinfo(np.int64).max
        sample_thresh = np.iinfo(np.int64).max

    # Sampling mask
    if np.issubdtype(type(nsamples), np.integer):
        nsamples = np.full((dim, dim), nsamples, dtype=np.int_)
    elif np.issubdtype(type(nsamples), np.floating):
        nsamples = np.full((dim, dim), nsamples)
    elif type(nsamples) is np.ndarray:
        if np.issubdtype(nsamples.dtype, np.integer):
            nsamples = nsamples.astype(np.int_)

    # Run main loop
    compute_point = eval(f"_compute_point{ndim}D")
    args = (kind, kn, costheta, kcoords, nsamples, sample_thresh, ndim,
            dim, shape, progress, exclude_upper, compute_point, *ffts)
    B, norm, omega, counts = _compute_bispectrum(*args)

    # If input data is real, so is the bispectrum.
    if np.issubdtype(U[0].dtype, np.floating):
        B = B.real

    # Set zero values to nan values for division
    mask = counts == 0.
    norm[mask] = np.nan
    counts[mask] = np.nan

    # Get bicoherence and average bispectrum
    b = np.abs(B) / norm
    B /= counts

    # Convert counts back to integer type
    if diagnostics:
        counts = counts.astype(np.int64)
        counts[mask] = 0

    # Switch back to theta monotonically increasing
    if costheta.size > 1:
        B[...] = np.flip(B, axis=0)
        b[...] = np.flip(b, axis=0)
        counts[...] = np.flip(counts, axis=0)
    else:
        B, b, counts = B[0], b[0], counts[0]

    if bench:
        print(f"Time: {time() - t0:.04f} s")

    result = [B, b, kn]
    if theta is not None:
        result.append(theta)
    if diagnostics:
        result.extend([omega, counts])

    return tuple(result)


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
def _compute_bispectrum(kind, kn, costheta, kcoords, nsamples, sample_thresh, ndim,
                        dim, shape, progress, exclude, compute_point, *ffts):
    knyq = max(shape) // 2
    ntheta = costheta.size
    bispec = np.full((ntheta, dim, dim), np.nan, dtype=np.complex128)
    binorm = np.full((ntheta, dim, dim), np.nan, dtype=np.float64)
    counts = np.full((ntheta, dim, dim), np.nan, dtype=np.float64)
    omega = np.zeros((dim, dim), dtype=np.int64)
    for i in range(dim):
        k1 = kn[i]
        k1ind = kind[i]
        nk1 = k1ind.size
        for j in range(i+1):
            k2 = kn[j]
            if ntheta == 1 and (exclude and k1 + k2 > knyq):
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
            cthetabuf = np.zeros(count, dtype=np.float64) if ntheta > 1 \
                else np.array([0.], dtype=np.float64)
            countbuf = np.zeros(count, dtype=np.float64)
            compute_point(k1ind, k2ind, kcoords, nk1, nk2,
                          shape, samp, count,
                          bispecbuf, binormbuf, cthetabuf, countbuf,
                          *ffts)
            if ntheta == 1:
                _fill_sum(i, j, bispec, binorm, counts, bispecbuf, binormbuf, countbuf)
            else:
                binned = np.searchsorted(costheta, cthetabuf)
                _fill_binned_sum(i, j, ntheta, binned, bispec, binorm, counts,
                                 bispecbuf, binormbuf, countbuf)
            omega[i, j], omega[j, i] = nk1*nk2, nk1*nk2
        if progress:
            with nb.objmode():
                _printProgressBar(i, dim-1)
    return bispec, binorm, omega, counts


@nb.njit(parallel=True, cache=True)
def _fill_sum(i, j, bispec, binorm, counts, bispecbuf, binormbuf, countbuf):
    N = countbuf.sum()
    norm = binormbuf.sum()
    value = bispecbuf.sum()
    bispec[0, i, j], bispec[0, j, i] = value, value
    binorm[0, i, j], binorm[0, j, i] = norm, norm
    counts[0, i, j], counts[0, j, i] = N, N


@nb.njit(parallel=True, cache=True)
def _fill_binned_sum(i, j, ntheta, binned, bispec, binorm, counts,
                     bispecbuf, binormbuf, countbuf):
    N = np.bincount(binned, weights=countbuf, minlength=ntheta)
    norm = np.bincount(binned, weights=binormbuf, minlength=ntheta)
    value = np.bincount(binned, weights=bispecbuf.real, minlength=ntheta) +\
        1.j*np.bincount(binned, weights=bispecbuf.imag, minlength=ntheta)
    bispec[:, i, j], bispec[:, j, i] = value, value
    binorm[:, i, j], binorm[:, j, i] = norm, norm
    counts[:, i, j], counts[:, j, i] = N, N


@nb.njit(parallel=True, cache=True)
def _compute_point3D(k1ind, k2ind, kcoords, nk1, nk2, shape,
                     samp, count, bispecbuf, binormbuf,
                     cthetabuf, countbuf, *ffts):
    kx, ky, kz = kcoords[0], kcoords[1], kcoords[2]
    Nx, Ny, Nz = shape[0], shape[1], shape[2]
    for idx in nb.prange(count):
        n, m = k1ind[samp[idx] % nk1], k2ind[samp[idx] // nk1]
        k1x, k1y, k1z = kx[n], ky[n], kz[n]
        k2x, k2y, k2z = kx[m], ky[m], kz[m]
        k3x, k3y, k3z = k1x+k2x, k1y+k2y, k1z+k2z
        if np.abs(k3x) > Nx//2 or np.abs(k3y) > Ny//2 or np.abs(k3z) > Nz//2:
            continue
        sample, norm = 0, 0
        for fft in ffts:
            temp = fft[k1x, k1y, k1z]*fft[k2x, k2y, k2z]*np.conj(fft[k3x, k3y, k3z])
            sample += temp
            norm += np.abs(temp)
        bispecbuf[idx] = sample
        binormbuf[idx] = norm
        countbuf[idx] = 1
        if cthetabuf.size > 1:
            k1dotk2 = (k1x*k2x+k1y*k2y+k1z*k2z)
            k1norm, k2norm = np.sqrt(k1x**2+k1y**2+k1z**2), np.sqrt(k2x**2+k2y**2+k2z**2)
            costheta = k1dotk2 / (k1norm*k2norm)
            cthetabuf[idx] = costheta


@nb.njit(parallel=True, cache=True)
def _compute_point2D(k1ind, k2ind, kcoords, nk1, nk2, shape,
                     samp, count, bispecbuf, binormbuf,
                     cthetabuf, countbuf, *ffts):
    kx, ky = kcoords[0], kcoords[1]
    Nx, Ny = shape[0], shape[1]
    for idx in nb.prange(count):
        n, m = k1ind[samp[idx] % nk1], k2ind[samp[idx] // nk1]
        k1x, k1y = kx[n], ky[n]
        k2x, k2y = kx[m], ky[m]
        k3x, k3y = k1x+k2x, k1y+k2y
        if np.abs(k3x) > Nx//2 or np.abs(k3y) > Ny//2:
            continue
        sample, norm = 0, 0
        for fft in ffts:
            temp = fft[k1x, k1y]*fft[k2x, k2y]*np.conj(fft[k3x, k3y])
            sample += temp
            norm += np.abs(temp)
        bispecbuf[idx] = sample
        binormbuf[idx] = norm
        countbuf[idx] = 1
        if cthetabuf.size > 1:
            k1dotk2 = (k1x*k2x+k1y*k2y)
            k1norm, k2norm = np.sqrt(k1x**2+k1y**2), np.sqrt(k2x**2+k2y**2)
            costheta = k1dotk2 / (k1norm*k2norm)
            cthetabuf[idx] = costheta


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

    N = 200
    np.random.seed(1234)
    data = np.random.normal(size=N**2).reshape((N, N))+1

    kmin, kmax = 1, 100
    theta = np.array([0, np.pi/4, np.pi/2, 3*np.pi/4])
    bispec, bicoh, kn, theta, omega, counts = bispectrum(data, nsamples=None,
                                                         kmin=kmin, kmax=kmax,
                                                         theta=theta, progress=True,
                                                         mean_subtract=True,
                                                         diagnostics=True, bench=True)
    print(np.nansum(bispec), np.nansum(bicoh))

    tidx = 0
    bispec, bicoh, counts = [x[tidx] for x in [bispec, bicoh, counts]]

    # Plot
    cmap = 'plasma'
    labels = [r"$B(k_1, k_2)$", "$b(k_1, k_2)$", "counts"]
    data = [np.log10(np.abs(bispec)), np.log10(bicoh), np.log10(counts)]
    fig, axes = plt.subplots(ncols=len(data))
    for i in range(len(data)):
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
