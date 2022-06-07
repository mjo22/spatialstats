"""
Implementation using CuPy acceleration.

.. moduleauthor:: Michael O'Brien <michaelobrien@g.harvard.edu>

"""

import numpy as np
import cupy as cp
from cupyx.scipy import fft as cufft
from time import time


def bispectrum(*u, ntheta=None, kmin=None, kmax=None,
               diagnostics=True, error=False,
               nsamples=None, sample_thresh=None,
               compute_fft=True, exclude_upper=False,
               double=True, blocksize=128,
               bench=False, progress=False, **kwargs):
    """
    See the documentation for the :ref:`CPU version<bispectrum>`.

    This implementation is written to minimize memory
    requirements on a single GPU. Set ``double = False``
    to save memory.

    Parameters
    ----------
    u : `np.ndarray` or `cp.ndarray`
        Scalar field or vector components to correlate.
        If vector data, pass arguments as ``u1, u2, u3``,
        where ``ui`` is a vector component.
        Each ``ui`` can be 2D or 3D, and all must have the
        same ``ui.shape`` and ``ui.dtype``.
    ntheta : `int`, optional
        Number of angular bins :math:`\\cos\\theta` between triangles
        formed by wavevectors :math:`\mathbf{k_1}, \ \mathbf{k_2}`.
        If ``None``, sum over all triangle angles. Otherwise,
        return a bispectrum for each angular bin.
    kmin : `int`, optional
        Minimum wavenumber in bispectrum calculation.
        If ``None``, ``kmin = 1``.
    kmax : `int`, optional
        Maximum wavenumber in bispectrum calculation.
        If ``None``, ``kmax = max(u.shape)//2``.
    diagnostics : `bool`, optional
        Return the optional sampling and normalization diagnostics,
        documented below. Set ``error = True`` to also return the
        standard error of the mean.
    error : `bool`, optional
        Return standard error of the mean of the Monte-Carlo integration.
        ``diagnostics = True`` must also be set. This will add a bit of
        time to the calculation.
    nsamples : `int`, `float` or `np.ndarray`, shape `(kmax-kmin+1, kmax-kmin+1)`, optional
        Number of sample triangles or fraction of total
        possible triangles. This may be an array that
        specifies for a given :math:`k_1, \ k_2`.
        If ``None``, calculate the exact sum.
    sample_thresh : `int`, optional
        When the size of the sample space is greater than
        this number, start to use sampling instead of exact
        calculation. If ``None``, switch to exact calculation
        when ``nsamples`` is less than the size of the sample space.
    compute_fft : `bool`, optional
        If ``False``, do not take the FFT of the input data.
        FFTs should not be passed with the zero-frequency
        component in the center.
    exclude_upper : `bool`, optional
        If ``True``, set points where :math:`k_1 + k_2 > k_{nyq}`
        to ``np.nan``. This keyword only applies when summing
        over angles, e.g. when ``ntheta is None``.
    double : `bool`, optional
        If ``False``, do calculation in single precision.
    blocksize : `int`, optional
        Number of threads per block for GPU kernels.
        The optimal value will vary depending on hardware.
    progress : `bool`, optional
        Print progress bar of calculation.
    bench : `bool`, optional
        If ``True``, print calculation time.
    kwargs
        Additional keyword arguments passed to
        ``cupyx.scipy.fft.fftn``.

    Returns
    -------
    B : `np.ndarray`, shape `(m, kmax-kmin+1, kmax-kmin+1)`
        Bispectrum :math:`B(k_1, k_2, \\cos\\theta)`.
    b : `np.ndarray`, shape `(m, kmax-kmin+1, kmax-kmin+1)`
        Bicoherence index :math:`b(k_1, k_2, \\cos\\theta)`.
    kn : `np.ndarray`, shape `(kmax-kmin+1,)`
        Wavenumbers :math:`k_1` or :math:`k_2` along axis of bispectrum.
    theta : `np.ndarray`, shape `(m,)`, optional
        Left edges of angular bins :math:`\\cos\\theta`, ranging from
        :math:`[-1, \ 1)`.
    counts : `np.ndarray`, shape `(m, kmax-kmin+1, kmax-kmin+1)`, optional
        Number of evaluations in the bispectrum sum, :math:`N`.
    omega : `np.ndarray`, shape `(kmax-kmin+1, kmax-kmin+1)`, optional
        Number of possible triangles in the sample space, :math:`|\\Omega|`.
        This is implemented for if sampling were *not* restricted by the Nyquist
        frequency. Note that this is only implemented for ``ntheta = None``.
    stderr : `np.ndarray`, shape `(m, kmax-kmin+1, kmax-kmin+1)`, optional
        Standard error of the mean for each bin. This can be an
        error estimate for the Monte Carlo integration. To convert
        to the standard deviation, evaluate ``stderr * np.sqrt(counts)``.
    """

    if double:
        float, complex = cp.float64, cp.complex128
    else:
        float, complex = cp.float32, cp.complex64

    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()

    shape, ndim = u[0].shape, u[0].ndim
    ncomp = len(u)

    if ndim not in [2, 3]:
        raise ValueError("Data must be 2D or 3D.")
    if ncomp not in [1, 3]:
        raise ValueError("Pass either 1 scalar field or 3 vector components.")

    # Geometry of output image
    kmax = int(max(shape)/2) if kmax is None else int(kmax)
    kmin = 1 if kmin is None else int(kmin)
    kn = np.arange(kmin, kmax+1, 1, dtype=int)
    dim = kn.size
    dtheta = 2/ntheta
    costheta = cp.arange(-1, 1, dtheta)+dtheta if ntheta is not None else cp.array([1.])

    # theta = 0 should be included
    if ntheta is not None:
        costheta[-1] += 1e-5

    if bench:
        t0 = time()

    # Get binned radial coordinates of FFT
    kv = cp.meshgrid(*([cp.fft.fftfreq(Ni).astype(cp.float32)*Ni
                        for Ni in shape]), indexing="ij")
    kr = cp.zeros_like(kv[0])
    tpb = blocksize
    bpg = (kr.size + (tpb - 1)) // tpb
    # Need to write custom kernels to save memory
    for i in range(ndim):
        _sqr_add((bpg,), (tpb,), (kr, kv[i], kr.size))
    _sqrt((bpg,), (tpb,), (kr, kr.size))

    # Convert coordinates to int16
    kcoords = []
    if ndim == 2:
        kx, ky = kv[0], kv[1]
        del kv
    else:
        kx, ky, kz = kv[0], kv[1], kv[2]
        del kv
        kcoords.append(kz.ravel().astype(np.int16))
        del kz
    kcoords.append(ky.ravel().astype(np.int16))
    del ky
    kcoords.append(kx.ravel().astype(np.int16))
    del kx
    kcoords.reverse()

    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()

    # Bin coordinates
    kbins = cp.arange(int(np.ceil(kr.max().get())))
    kbinned = cp.digitize(kr, kbins)
    kbinned[...] -= 1

    del kr
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()

    # Convert to int16
    kbinned = kbinned.astype(cp.int16)

    # Enumerate indices in each bin
    k1bins, k2bins = [], []
    for ki in kn:
        mask = kbinned == ki
        temp1 = cp.where(mask)
        temp2 = cp.where(mask[..., :shape[-1]//2+1])
        k1bins.append(cp.ravel_multi_index(temp1, shape))
        k2bins.append(cp.ravel_multi_index(temp2, shape))

    del kbinned
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()

    # FFT
    ffts = []
    for i in range(ncomp):
        if compute_fft:
            temp = cp.asarray(u[i], dtype=complex)
            fft = _cufftn(temp, **kwargs)
            del temp
        else:
            fft = cp.asarray(u[i], dtype=complex)
        ffts.append(fft[..., :shape[-1]//2])
        del fft
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

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
    f = "f" if not double else ""
    v = "Vec" if ncomp > 1 else ""
    compute_point = _module.get_function(f"computePoint{v}{ndim}D{f}")
    args = (k1bins, k2bins, kn, costheta, kcoords,
            nsamples, sample_thresh, ndim, dim, shape,
            double, progress, exclude_upper, error,
            blocksize, compute_point, *ffts)
    B, norm, omega, counts, stderr = _compute_bispectrum(*args)

    # Set zero values to nan values for division
    mask = counts == 0.
    norm[mask] = cp.nan
    counts[mask] = cp.nan

    # Throw away imaginary part
    B = B.real

    # Get bicoherence and average bispectrum
    b = np.abs(B) / norm
    B /= counts

    # Prepare diagnostics
    if error:
        stderr[counts <= 1.] = cp.nan

    # If no theta bins, get rid of m dimension
    if ntheta is None:
        B, b = B[0], b[0]
        if diagnostics:
            counts = counts[0]
            if error:
                stderr = stderr[0]

    if bench:
        print(f"Time: {time() - t0:.04f} s")

    result = [B.get(), b.get(), kn]
    if ntheta is not None:
        result.append(costheta.get()-dtheta)
    if diagnostics:
        result.extend([counts.get(), omega])
        if error:
            result.append(stderr.get())

    return tuple(result)


def _cufftn(data, overwrite_input=True, **kwargs):
    """
    Calculate the N-dimensional fft of an image
    with memory efficiency

    Parameters
    ----------
    data : cupy.ndarray
        Real or complex-valued 2D or 3D image.
    overwrite_input : bool, optional
        Specify whether input data can be destroyed.
        This is useful if low on memory.
        See cupyx.scipy.fft.fftn for more.

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
        fftn = cufft.rfftn  # if ndplan else cp.fft.rfftn
    elif data.dtype in [cp.complex64, cp.complex128]:
        value_type = 'C2C'
        fftn = cufft.fftn  # if ndplan else cp.fft.fftn
    else:
        raise ValueError(f"Unrecognized data type {data.dtype}.")

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


_sqr_add = cp.RawKernel(r'''
extern "C" __global__

void square_add(float* kr, float* ki, int size) {

    for (long idx = blockDim.x * blockIdx.x + threadIdx.x;
         idx < size;
         idx += blockDim.x * gridDim.x) {

        kr[idx] += ki[idx]*ki[idx];

    }

}

''', 'square_add')


_sqrt = cp.RawKernel(r'''
extern "C" __global__

void square_root(float* kr, int size) {

    for (long idx = blockDim.x * blockIdx.x + threadIdx.x;
         idx < size;
         idx += blockDim.x * gridDim.x) {

        kr[idx] = sqrt(kr[idx]);

    }

}

''', 'square_root')


def _compute_bispectrum(k1bins, k2bins, kn, costheta, kcoords,
                        nsamples, sample_thresh, ndim, dim, shape,
                        double, progress, exclude, error,
                        blocksize, compute_point, *ffts):
    knyq = max(shape) // 2
    shape = [cp.int16(Ni) for Ni in shape]
    ntheta = costheta.size
    nffts = len(ffts)
    if double:
        float, complex = cp.float64, cp.complex128
    else:
        float, complex = cp.float32, cp.complex64
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    bispec = cp.full((ntheta, dim, dim), cp.nan+1.j*cp.nan, dtype=complex)
    binorm = cp.full((ntheta, dim, dim), cp.nan, dtype=float)
    counts = cp.full((ntheta, dim, dim), cp.nan, dtype=float)
    omega = np.zeros((dim, dim), dtype=np.int64)
    if error:
        stderr = cp.full((ntheta, dim, dim), np.nan, dtype=float)
    else:
        stderr = cp.array([0.], dtype=float)
    for i in range(dim):
        k1 = kn[i]
        k1ind = k1bins[i]
        nk1 = k1ind.size
        dim2 = dim if nffts > 1 else i+1
        for j in range(dim2):
            k2 = kn[j]
            if ntheta == 1 and (exclude and k1 + k2 > knyq):
                continue
            k2ind = k2bins[j]
            nk2 = k2ind.size
            nsamp = nsamples[i, j]
            nsamp = int(nsamp) if type(nsamp) is np.int64 \
                else max(int(nsamp*nk1*nk2), 1)
            if nsamp < nk1*nk2 or nsamp > sample_thresh:
                samp = cp.random.randint(0, nk1*nk2,
                                         size=nsamp, dtype=cp.int64)
                count = nsamp
            else:
                samp = cp.arange(nk1*nk2, dtype=cp.int64)
                count = nk1*nk2
            tpb = blocksize
            bpg = (count + (tpb - 1)) // tpb
            bispecbuf = cp.zeros(count, dtype=complex)
            binormbuf = cp.zeros(count, dtype=float)
            cthetabuf = cp.zeros(count, dtype=float) if ntheta > 1 \
                else cp.array([0.], dtype=float)
            countbuf = cp.zeros(count, dtype=float)
            compute_point((bpg,), (tpb,), (k1ind, k2ind, *kcoords,
                                           cp.int32(ntheta),
                                           cp.int64(nk1), cp.int64(nk2),
                                           *shape, samp, cp.int64(count),
                                           bispecbuf, binormbuf,
                                           cthetabuf, countbuf,
                                           *ffts))
            if ntheta == 1:
                _fill_sum(i, j, bispec, binorm, counts, stderr,
                          bispecbuf, binormbuf, countbuf, nffts, error)
            else:
                binned = cp.searchsorted(costheta, cthetabuf)
                _fill_binned_sum(i, j, ntheta, binned,
                                 bispec, binorm, counts, stderr,
                                 bispecbuf, binormbuf, countbuf, nffts, error)
            omega[i, j] = nk1*nk2
            if nffts == 1:
                omega[j, i] = nk1*nk2
            del bispecbuf, binormbuf, countbuf, samp
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
        if progress:
            _printProgressBar(i, dim-1)

    return bispec, binorm, omega, counts, stderr


def _fill_sum(i, j, bispec, binorm, counts, stderr,
              bispecbuf, binormbuf, countbuf, nffts, error):
    N = countbuf.sum()
    norm = binormbuf.sum()
    value = bispecbuf.sum()
    bispec[0, i, j] = value
    binorm[0, i, j] = norm
    counts[0, i, j] = N
    if nffts == 1:
        bispec[0, j, i] = value
        binorm[0, j, i] = norm
        counts[0, j, i] = N
    if error and N > 1:
        variance = cp.abs(bispecbuf - (value / N))**2
        err = np.sqrt(variance.sum() / (N*(N - 1)))
        stderr[0, i, j] = err
        if nffts == 1:
            stderr[0, j, i] = err


def _fill_binned_sum(i, j, ntheta, binned, bispec, binorm, counts,
                     stderr, bispecbuf, binormbuf, countbuf, nffts, error):
    N = cp.bincount(binned, weights=countbuf, minlength=ntheta)
    norm = cp.bincount(binned, weights=binormbuf, minlength=ntheta)
    value = cp.bincount(binned, weights=bispecbuf.real, minlength=ntheta) +\
        1.j*cp.bincount(binned, weights=bispecbuf.imag, minlength=ntheta)
    bispec[:, i, j] = value
    binorm[:, i, j] = norm
    counts[:, i, j] = N
    if nffts == 1:
        bispec[:, j, i] = value
        binorm[:, j, i] = norm
        counts[:, j, i] = N
    if error:
        variance = cp.zeros_like(countbuf)
        for n in range(ntheta):
            if N[n] > 1:
                idxs = cp.where(binned == n)
                mean = value[n] / N[n]
                variance[idxs] = cp.abs(bispecbuf[idxs] - mean)**2 / (N[n]*(N[n]-1))
        err = cp.sqrt(cp.bincount(binned, weights=variance, minlength=ntheta))
        stderr[:, i, j] = err
        if nffts == 1:
            stderr[:, j, i] = err


_module = cp.RawModule(code=r'''
# include <cupy/complex.cuh>

extern "C" {
__global__ void computePoint3D(long* k1ind, long* k2ind,
                               short* kx, short* ky, short* kz,
                               int ntheta, long nk1, long nk2,
                               short Nx, short Ny, short Nz,
                               const long* samp, long count,
                               complex<double>* bispecbuf, double* binormbuf,
                               double* cthetabuf, double* countbuf,
                               const complex<double>* fft) {

    for (long idx = blockDim.x * blockIdx.x + threadIdx.x;
         idx < count;
         idx += blockDim.x * gridDim.x) {

        long n, m;

        n = k1ind[samp[idx] % nk1]; m = k2ind[samp[idx] / nk1];

        short k1x, k1y, k1z, k2x, k2y, k2z, k3x, k3y, k3z;

        k1x = kx[n]; k1y = ky[n]; k1z = kz[n];
        k2x = kx[m]; k2y = ky[m]; k2z = kz[m];
        k3x = k1x+k2x; k3y = k1y+k2y; k3z = k1z+k2z;

        if ((abs(k3x) > Nx/2) || (abs(k3y) > Ny/2) || (abs(k3z) > Nz/2)) { return; }

        __syncthreads();

        // Calculate angles
        if (ntheta > 1) {
            double k1dotk2, k1norm, k2norm, costheta;
            double x1, y1, z1, x2, y2, z2;
            x1 = double(k1x); y1 = double(k1y); z1 = double(k1z);
            x2 = double(k2x); y2 = double(k2y); z2 = double(k2z);
            k1dotk2 = x1*x2 + y1*y2 + z1*z2;
            k1norm = sqrt(x1*x1 + y1*y1 + z1*z1);
            k2norm = sqrt(x2*x2 + y2*y2 + z2*z2);
            costheta = k1dotk2 / (k1norm*k2norm);
            cthetabuf[idx] = costheta;
        }

        // Map frequency domain to index domain
        short q1x, q1y, q1z, q2x, q2y, q2z, q3x, q3y, q3z;
        short cj1, cj2, cj3;

        if (k1z < 0) { k1x *= -1; k1y *= -1; k1z *= -1; cj1 = 1; } else { cj1 = 0; }
        if (k2z < 0) { k2x *= -1; k2y *= -1; k2z *= -1; cj2 = 1; } else { cj2 = 0; }
        if (k3z < 0) { k3x *= -1; k3y *= -1; k3z *= -1; cj3 = 1; } else { cj3 = 0; }

        if (k1x < 0) { q1x = k1x + Nx; } else { q1x = k1x; }
        if (k1y < 0) { q1y = k1y + Ny; } else { q1y = k1y; }
        q1z = k1z;

        if (k2x < 0) { q2x = k2x + Nx; } else { q2x = k2x; }
        if (k2y < 0) { q2y = k2y + Ny; } else { q2y = k2y; }
        q2z = k2z;

        if (k3x < 0) { q3x = k3x + Nx; } else { q3x = k3x; }
        if (k3y < 0) { q3y = k3y + Ny; } else { q3y = k3y; }
        q3z = k3z;

        // Map multi-dimensional indices to 1-dimensional indices
        long idx1 = (q1x*Ny + q1y)*Nz + q1z;
        long idx2 = (q2x*Ny + q2y)*Nz + q2z;
        long idx3 = (q3x*Ny + q3y)*Nz + q3z;

        // Sample correlation function
        complex<double> sample, s1, s2, s3;
        double mod;

        if (cj1 == 1) { s1 = conj(fft[idx1]); } else { s1 = fft[idx1]; }
        if (cj2 == 1) { s2 = conj(fft[idx2]); } else { s2 = fft[idx2]; }
        if (cj3 == 1) { s3 = fft[idx3]; } else { s3 = conj(fft[idx3]); }
        sample = s1*s2*s3;
        mod = abs(sample);

        bispecbuf[idx] = sample;
        binormbuf[idx] = mod;
        countbuf[idx] = 1;

    }
}


__global__ void computePoint2D(const long* k1ind, const long* k2ind,
                               short* kx, short* ky,
                               int ntheta, long nk1, long nk2,
                               short Nx, short Ny, const long* samp, long count,
                               complex<double>* bispecbuf, double* binormbuf,
                               double* cthetabuf, double* countbuf,
                               const complex<double>* fft) {

    for (long idx = blockDim.x * blockIdx.x + threadIdx.x;
         idx < count;
         idx += blockDim.x * gridDim.x) {

        long n, m;

        n = k1ind[samp[idx] % nk1]; m = k2ind[samp[idx] / nk1];

        short k1x, k1y, k2x, k2y, k3x, k3y;

        k1x = kx[n]; k1y = ky[n];
        k2x = kx[m]; k2y = ky[m];
        k3x = k1x+k2x; k3y = k1y+k2y;

        if ((abs(k3x) > Nx/2) || (abs(k3y) > Ny/2)) { return; }

        __syncthreads();

        // Calculate angles
        if (ntheta > 1) {
            double k1dotk2, k1norm, k2norm, costheta;
            double x1, y1, x2, y2;
            x1 = double(k1x); y1 = double(k1y);
            x2 = double(k2x); y2 = double(k2y);
            k1dotk2 = x1*x2 + y1*y2;
            k1norm = sqrt(x1*x1 + y1*y1);
            k2norm = sqrt(x2*x2 + y2*y2);
            costheta = k1dotk2 / (k1norm*k2norm);
            cthetabuf[idx] = costheta;
        }

        // Map frequency domain to index domain
        short q1x, q1y, q2x, q2y, q3x, q3y;
        short cj1, cj2, cj3;

        if (k1y < 0) { k1x *= -1; k1y *= -1; cj1 = 1; } else { cj1 = 0; }
        if (k2y < 0) { k2x *= -1; k2y *= -1; cj2 = 1; } else { cj2 = 0; }
        if (k3y < 0) { k3x *= -1; k3y *= -1; cj3 = 1; } else { cj3 = 0; }

        if (k1x < 0) { q1x = k1x + Nx; } else { q1x = k1x; }
        q1y = k1y;

        if (k2x < 0) { q2x = k2x + Nx; } else { q2x = k2x; }
        q2y = k2y;

        if (k3x < 0) { q3x = k3x + Nx; } else { q3x = k3x; }
        q3y = k3y;

        // Map multi-dimensional indices to 1-dimensional indices
        long idx1 = (q1x*Ny + q1y);
        long idx2 = (q2x*Ny + q2y);
        long idx3 = (q3x*Ny + q3y);

        // Sample correlation function
        complex<double> sample, s1, s2, s3;
        double mod;

        if (cj1 == 1) { s1 = conj(fft[idx1]); } else { s1 = fft[idx1]; }
        if (cj2 == 1) { s2 = conj(fft[idx2]); } else { s2 = fft[idx2]; }
        if (cj3 == 1) { s3 = fft[idx3]; } else { s3 = conj(fft[idx3]); }
        sample = s1*s2*s3;
        mod = abs(sample);

        bispecbuf[idx] = sample;
        binormbuf[idx] = mod;
        countbuf[idx] = 1;

    }
}

__global__ void computePointVec3D(long* k1ind, long* k2ind,
                                  short* kx, short* ky, short* kz,
                                  int ntheta, long nk1, long nk2,
                                  short Nx, short Ny, short Nz,
                                  const long* samp, long count,
                                  complex<double>* bispecbuf, double* binormbuf,
                                  double* cthetabuf, double* countbuf,
                                  const complex<double>* fft1,
                                  const complex<double>* fft2,
                                  const complex<double>* fft3) {

    for (long idx = blockDim.x * blockIdx.x + threadIdx.x;
         idx < count;
         idx += blockDim.x * gridDim.x) {

        long n, m;

        n = k1ind[samp[idx] % nk1]; m = k2ind[samp[idx] / nk1];

        short k1x, k1y, k1z, k2x, k2y, k2z, k3x, k3y, k3z;

        k1x = kx[n]; k1y = ky[n]; k1z = kz[n];
        k2x = kx[m]; k2y = ky[m]; k2z = kz[m];
        k3x = k1x+k2x; k3y = k1y+k2y; k3z = k1z+k2z;

        if ((abs(k3x) > Nx/2) || (abs(k3y) > Ny/2) || (abs(k3z) > Nz/2)) { return; }

        __syncthreads();

        // Calculate angles
        if (ntheta > 1) {
            double k1dotk2, k1norm, k2norm, costheta;
            double x1, y1, z1, x2, y2, z2;
            x1 = double(k1x); y1 = double(k1y); z1 = double(k1z);
            x2 = double(k2x); y2 = double(k2y); z2 = double(k2z);
            k1dotk2 = x1*x2 + y1*y2 + z1*z2;
            k1norm = sqrt(x1*x1 + y1*y1 + z1*z1);
            k2norm = sqrt(x2*x2 + y2*y2 + z2*z2);
            costheta = k1dotk2 / (k1norm*k2norm);
            cthetabuf[idx] = costheta;
        }

        // Map frequency domain to index domain
        short q1x, q1y, q1z, q2x, q2y, q2z, q3x, q3y, q3z;
        short cj1, cj2, cj3;

        if (k1z < 0) { k1x *= -1; k1y *= -1; k1z *= -1; cj1 = 1; } else { cj1 = 0; }
        if (k2z < 0) { k2x *= -1; k2y *= -1; k2z *= -1; cj2 = 1; } else { cj2 = 0; }
        if (k3z < 0) { k3x *= -1; k3y *= -1; k3z *= -1; cj3 = 1; } else { cj3 = 0; }

        if (k1x < 0) { q1x = k1x + Nx; } else { q1x = k1x; }
        if (k1y < 0) { q1y = k1y + Ny; } else { q1y = k1y; }
        q1z = k1z;

        if (k2x < 0) { q2x = k2x + Nx; } else { q2x = k2x; }
        if (k2y < 0) { q2y = k2y + Ny; } else { q2y = k2y; }
        q2z = k2z;

        if (k3x < 0) { q3x = k3x + Nx; } else { q3x = k3x; }
        if (k3y < 0) { q3y = k3y + Ny; } else { q3y = k3y; }
        q3z = k3z;

        // Map multi-dimensional indices to 1-dimensional indices
        long idx1 = (q1x*Ny + q1y)*Nz + q1z;
        long idx2 = (q2x*Ny + q2y)*Nz + q2z;
        long idx3 = (q3x*Ny + q3y)*Nz + q3z;

        // Sample correlation function
        complex<double> s;
        complex<double> s1, s2, s3;

        if (cj1 == 1) {
            s1 = conj(fft1[idx1]);
        }
        else {
            s1 = fft1[idx1];
        }
        if (cj2 == 1) {
            s2 = conj(fft2[idx2]);
        }
        else {
            s2 = fft2[idx2];
        }
        if (cj3 == 1) {
            s3 = fft3[idx3];
        }
        else {
            s3 = conj(fft3[idx3]);
        }

        s = s1*s2*s3;

        bispecbuf[idx] = s;
        binormbuf[idx] = abs(s);
        countbuf[idx] = 1;

    }
}


__global__ void computePointVec2D(const long* k1ind, const long* k2ind,
                                  short* kx, short* ky,
                                  int ntheta, long nk1, long nk2,
                                  short Nx, short Ny, const long* samp, long count,
                                  complex<double>* bispecbuf, double* binormbuf,
                                  double* cthetabuf, double* countbuf,
                                  const complex<double>* fft1,
                                  const complex<double>* fft2,
                                  const complex<double>* fft3) {

    for (long idx = blockDim.x * blockIdx.x + threadIdx.x;
         idx < count;
         idx += blockDim.x * gridDim.x) {

        long n, m;

        n = k1ind[samp[idx] % nk1]; m = k2ind[samp[idx] / nk1];

        short k1x, k1y, k2x, k2y, k3x, k3y;

        k1x = kx[n]; k1y = ky[n];
        k2x = kx[m]; k2y = ky[m];
        k3x = k1x+k2x; k3y = k1y+k2y;

        if ((abs(k3x) > Nx/2) || (abs(k3y) > Ny/2)) { return; }

        __syncthreads();

        // Calculate angles
        if (ntheta > 1) {
            double k1dotk2, k1norm, k2norm, costheta;
            double x1, y1, x2, y2;
            x1 = double(k1x); y1 = double(k1y);
            x2 = double(k2x); y2 = double(k2y);
            k1dotk2 = x1*x2 + y1*y2;
            k1norm = sqrt(x1*x1 + y1*y1);
            k2norm = sqrt(x2*x2 + y2*y2);
            costheta = k1dotk2 / (k1norm*k2norm);
            cthetabuf[idx] = costheta;
        }

        // Map frequency domain to index domain
        short q1x, q1y, q2x, q2y, q3x, q3y;
        short cj1, cj2, cj3;

        if (k1y < 0) { k1x *= -1; k1y *= -1; cj1 = 1; } else { cj1 = 0; }
        if (k2y < 0) { k2x *= -1; k2y *= -1; cj2 = 1; } else { cj2 = 0; }
        if (k3y < 0) { k3x *= -1; k3y *= -1; cj3 = 1; } else { cj3 = 0; }

        if (k1x < 0) { q1x = k1x + Nx; } else { q1x = k1x; }
        q1y = k1y;

        if (k2x < 0) { q2x = k2x + Nx; } else { q2x = k2x; }
        q2y = k2y;

        if (k3x < 0) { q3x = k3x + Nx; } else { q3x = k3x; }
        q3y = k3y;

        // Map multi-dimensional indices to 1-dimensional indices
        long idx1 = (q1x*Ny + q1y);
        long idx2 = (q2x*Ny + q2y);
        long idx3 = (q3x*Ny + q3y);

        // Sample correlation function
        complex<double> s;
        complex<double> s1, s2, s3;

        if (cj1 == 1) {
            s1 = conj(fft1[idx1]);
        }
        else {
            s1 = fft1[idx1];
        }
        if (cj2 == 1) {
            s2 = conj(fft2[idx2]);
        }
        else {
            s2 = fft2[idx2];
        }
        if (cj3 == 1) {
            s3 = fft3[idx3];
        }
        else {
            s3 = conj(fft3[idx3]);
        }

        s = s1*s2*s3;

        bispecbuf[idx] = s;
        binormbuf[idx] = abs(s);
        countbuf[idx] = 1;

    }
}

__global__ void computePoint3Df(const long* k1ind, const long* k2ind,
                                const short* kx, const short* ky, const short* kz,
                                int ntheta, long nk1, long nk2,
                                short Nx, short Ny, short Nz,
                                const long* samp, long count,
                                complex<float>* bispecbuf, float* binormbuf,
                                float* cthetabuf, float* countbuf,
                                const complex<float>* fft) {

    for (long idx = blockDim.x * blockIdx.x + threadIdx.x;
         idx < count;
         idx += blockDim.x * gridDim.x) {

        long n, m;

        n = k1ind[samp[idx] % nk1]; m = k2ind[samp[idx] / nk1];

        short k1x, k1y, k1z, k2x, k2y, k2z, k3x, k3y, k3z;

        k1x = kx[n]; k1y = ky[n]; k1z = kz[n];
        k2x = kx[m]; k2y = ky[m]; k2z = kz[m];
        k3x = k1x+k2x; k3y = k1y+k2y; k3z = k1z+k2z;

        if ((abs(k3x) > Nx/2) || (abs(k3y) > Ny/2) || (abs(k3z) > Nz/2)) { return; }

        __syncthreads();

        // Calculate angles
        if (ntheta > 1) {
            float k1dotk2, k1norm, k2norm, costheta;
            float x1, y1, z1, x2, y2, z2;
            x1 = float(k1x); y1 = float(k1y); z1 = float(k1z);
            x2 = float(k2x); y2 = float(k2y); z2 = float(k2z);
            k1dotk2 = x1*x2 + y1*y2 + z1*z2;
            k1norm = sqrt(x1*x1 + y1*y1 + z1*z1);
            k2norm = sqrt(x2*x2 + y2*y2 + z2*z2);
            costheta = k1dotk2 / (k1norm*k2norm);
            cthetabuf[idx] = costheta;
        }

        // Map frequency domain to index domain
        short q1x, q1y, q1z, q2x, q2y, q2z, q3x, q3y, q3z;
        short cj1, cj2, cj3;

        if (k1z < 0) { k1x *= -1; k1y *= -1; k1z *= -1; cj1 = 1; } else { cj1 = 0; }
        if (k2z < 0) { k2x *= -1; k2y *= -1; k2z *= -1; cj2 = 1; } else { cj2 = 0; }
        if (k3z < 0) { k3x *= -1; k3y *= -1; k3z *= -1; cj3 = 1; } else { cj3 = 0; }

        if (k1x < 0) { q1x = k1x + Nx; } else { q1x = k1x; }
        if (k1y < 0) { q1y = k1y + Ny; } else { q1y = k1y; }
        q1z = k1z;

        if (k2x < 0) { q2x = k2x + Nx; } else { q2x = k2x; }
        if (k2y < 0) { q2y = k2y + Ny; } else { q2y = k2y; }
        q2z = k2z;

        if (k3x < 0) { q3x = k3x + Nx; } else { q3x = k3x; }
        if (k3y < 0) { q3y = k3y + Ny; } else { q3y = k3y; }
        q3z = k3z;

        // Map multi-dimensional indices to 1-dimensional indices
        long idx1 = (q1x*Ny + q1y)*Nz + q1z;
        long idx2 = (q2x*Ny + q2y)*Nz + q2z;
        long idx3 = (q3x*Ny + q3y)*Nz + q3z;

        // Sample correlation function
        complex<float> sample, s1, s2, s3;
        float mod;

        if (cj1 == 1) { s1 = conj(fft[idx1]); } else { s1 = fft[idx1]; }
        if (cj2 == 1) { s2 = conj(fft[idx2]); } else { s2 = fft[idx2]; }
        if (cj3 == 1) { s3 = fft[idx3]; } else { s3 = conj(fft[idx3]); }
        sample = s1*s2*s3;
        mod = abs(sample);

        bispecbuf[idx] = sample;
        binormbuf[idx] = mod;
        countbuf[idx] = 1;

    }
}

__global__ void computePoint2Df(const long* k1ind, const long* k2ind,
                                const short* kx, const short* ky,
                                int ntheta, long nk1, long nk2,
                                short Nx, short Ny, const long* samp, long count,
                                complex<float>* bispecbuf, float* binormbuf,
                                float* cthetabuf, float* countbuf,
                                const complex<float>* fft) {

    for (long idx = blockDim.x * blockIdx.x + threadIdx.x;
         idx < count;
         idx += blockDim.x * gridDim.x) {

        long n, m;

        n = k1ind[samp[idx] % nk1]; m = k2ind[samp[idx] / nk1];

        short k1x, k1y, k2x, k2y, k3x, k3y;

        k1x = kx[n]; k1y = ky[n];
        k2x = kx[m]; k2y = ky[m];
        k3x = k1x+k2x; k3y = k1y+k2y;

        if ((abs(k3x) > Nx/2) || (abs(k3y) > Ny/2)) { return; }

        __syncthreads();

        // Calculate angles
        if (ntheta > 1) {
            float k1dotk2, k1norm, k2norm, costheta;
            float x1, y1, x2, y2;
            x1 = float(k1x); y1 = float(k1y);
            x2 = float(k2x); y2 = float(k2y);
            k1dotk2 = x1*x2 + y1*y2;
            k1norm = sqrt(x1*x1 + y1*y1);
            k2norm = sqrt(x2*x2 + y2*y2);
            costheta = k1dotk2 / (k1norm*k2norm);
            cthetabuf[idx] = costheta;
        }

        // Map frequency domain to index domain
        short q1x, q1y, q2x, q2y, q3x, q3y;
        short cj1, cj2, cj3;

        if (k1y < 0) { k1x *= -1; k1y *= -1; cj1 = 1; } else { cj1 = 0; }
        if (k2y < 0) { k2x *= -1; k2y *= -1; cj2 = 1; } else { cj2 = 0; }
        if (k3y < 0) { k3x *= -1; k3y *= -1; cj3 = 1; } else { cj3 = 0; }

        if (k1x < 0) { q1x = k1x + Nx; } else { q1x = k1x; }
        q1y = k1y;

        if (k2x < 0) { q2x = k2x + Nx; } else { q2x = k2x; }
        q2y = k2y;

        if (k3x < 0) { q3x = k3x + Nx; } else { q3x = k3x; }
        q3y = k3y;

        // Map multi-dimensional indices to 1-dimensional indices
        long idx1 = (q1x*Ny + q1y);
        long idx2 = (q2x*Ny + q2y);
        long idx3 = (q3x*Ny + q3y);

        // Sample correlation function
        complex<double> sample, s1, s2, s3;
        double mod;

        if (cj1 == 1) { s1 = conj(fft[idx1]); } else { s1 = fft[idx1]; }
        if (cj2 == 1) { s2 = conj(fft[idx2]); } else { s2 = fft[idx2]; }
        if (cj3 == 1) { s3 = fft[idx3]; } else { s3 = conj(fft[idx3]); }
        sample = s1*s2*s3;
        mod = abs(sample);

        bispecbuf[idx] = sample;
        binormbuf[idx] = mod;
        countbuf[idx] = 1;

    }
}

__global__ void computePointVec3Df(const long* k1ind, const long* k2ind,
                                   const short* kx, const short* ky, const short* kz,
                                   int ntheta, long nk1, long nk2,
                                   short Nx, short Ny, short Nz,
                                   const long* samp, long count,
                                   complex<float>* bispecbuf, float* binormbuf,
                                   float* cthetabuf, float* countbuf,
                                   const complex<float>* fft1,
                                   const complex<float>* fft2,
                                   const complex<float>* fft3) {

    for (long idx = blockDim.x * blockIdx.x + threadIdx.x;
         idx < count;
         idx += blockDim.x * gridDim.x) {

        long n, m;

        n = k1ind[samp[idx] % nk1]; m = k2ind[samp[idx] / nk1];

        short k1x, k1y, k1z, k2x, k2y, k2z, k3x, k3y, k3z;

        k1x = kx[n]; k1y = ky[n]; k1z = kz[n];
        k2x = kx[m]; k2y = ky[m]; k2z = kz[m];
        k3x = k1x+k2x; k3y = k1y+k2y; k3z = k1z+k2z;

        if ((abs(k3x) > Nx/2) || (abs(k3y) > Ny/2) || (abs(k3z) > Nz/2)) { return; }

        __syncthreads();

        // Calculate angles
        if (ntheta > 1) {
            float k1dotk2, k1norm, k2norm, costheta;
            float x1, y1, z1, x2, y2, z2;
            x1 = float(k1x); y1 = float(k1y); z1 = float(k1z);
            x2 = float(k2x); y2 = float(k2y); z2 = float(k2z);
            k1dotk2 = x1*x2 + y1*y2 + z1*z2;
            k1norm = sqrt(x1*x1 + y1*y1 + z1*z1);
            k2norm = sqrt(x2*x2 + y2*y2 + z2*z2);
            costheta = k1dotk2 / (k1norm*k2norm);
            cthetabuf[idx] = costheta;
        }

        // Map frequency domain to index domain
        short q1x, q1y, q1z, q2x, q2y, q2z, q3x, q3y, q3z;
        short cj1, cj2, cj3;

        if (k1z < 0) { k1x *= -1; k1y *= -1; k1z *= -1; cj1 = 1; } else { cj1 = 0; }
        if (k2z < 0) { k2x *= -1; k2y *= -1; k2z *= -1; cj2 = 1; } else { cj2 = 0; }
        if (k3z < 0) { k3x *= -1; k3y *= -1; k3z *= -1; cj3 = 1; } else { cj3 = 0; }

        if (k1x < 0) { q1x = k1x + Nx; } else { q1x = k1x; }
        if (k1y < 0) { q1y = k1y + Ny; } else { q1y = k1y; }
        q1z = k1z;

        if (k2x < 0) { q2x = k2x + Nx; } else { q2x = k2x; }
        if (k2y < 0) { q2y = k2y + Ny; } else { q2y = k2y; }
        q2z = k2z;

        if (k3x < 0) { q3x = k3x + Nx; } else { q3x = k3x; }
        if (k3y < 0) { q3y = k3y + Ny; } else { q3y = k3y; }
        q3z = k3z;

        // Map multi-dimensional indices to 1-dimensional indices
        long idx1 = (q1x*Ny + q1y)*Nz + q1z;
        long idx2 = (q2x*Ny + q2y)*Nz + q2z;
        long idx3 = (q3x*Ny + q3y)*Nz + q3z;

        // Sample correlation function
        complex<float> s;
        complex<float> s1, s2, s3;

        if (cj1 == 1) {
            s1 = conj(fft1[idx1]);
        }
        else {
            s1 = fft1[idx1];
        }
        if (cj2 == 1) {
            s2 = conj(fft2[idx2]);
        }
        else {
            s2 = fft2[idx2];
        }
        if (cj3 == 1) {
            s3 = fft3[idx3];
        }
        else {
            s3 = conj(fft3[idx3]);
        }

        s = s1*s2*s3;

        bispecbuf[idx] = s;
        binormbuf[idx] = abs(s);
        countbuf[idx] = 1;

    }
}

__global__ void computePointVec2Df(const long* k1ind, const long* k2ind,
                                   const short* kx, const short* ky,
                                   int ntheta, long nk1, long nk2,
                                   short Nx, short Ny, const long* samp, long count,
                                   complex<float>* bispecbuf, float* binormbuf,
                                   float* cthetabuf, float* countbuf,
                                   const complex<float>* fft1,
                                   const complex<float>* fft2,
                                   const complex<float>* fft3) {

    for (long idx = blockDim.x * blockIdx.x + threadIdx.x;
         idx < count;
         idx += blockDim.x * gridDim.x) {

        long n, m;

        n = k1ind[samp[idx] % nk1]; m = k2ind[samp[idx] / nk1];

        short k1x, k1y, k2x, k2y, k3x, k3y;

        k1x = kx[n]; k1y = ky[n];
        k2x = kx[m]; k2y = ky[m];
        k3x = k1x+k2x; k3y = k1y+k2y;

        if ((abs(k3x) > Nx/2) || (abs(k3y) > Ny/2)) { return; }

        __syncthreads();

        // Calculate angles
        if (ntheta > 1) {
            float k1dotk2, k1norm, k2norm, costheta;
            float x1, y1, x2, y2;
            x1 = float(k1x); y1 = float(k1y);
            x2 = float(k2x); y2 = float(k2y);
            k1dotk2 = x1*x2 + y1*y2;
            k1norm = sqrt(x1*x1 + y1*y1);
            k2norm = sqrt(x2*x2 + y2*y2);
            costheta = k1dotk2 / (k1norm*k2norm);
            cthetabuf[idx] = costheta;
        }

        // Map frequency domain to index domain
        short q1x, q1y, q2x, q2y, q3x, q3y;
        short cj1, cj2, cj3;

        if (k1y < 0) { k1x *= -1; k1y *= -1; cj1 = 1; } else { cj1 = 0; }
        if (k2y < 0) { k2x *= -1; k2y *= -1; cj2 = 1; } else { cj2 = 0; }
        if (k3y < 0) { k3x *= -1; k3y *= -1; cj3 = 1; } else { cj3 = 0; }

        if (k1x < 0) { q1x = k1x + Nx; } else { q1x = k1x; }
        q1y = k1y;

        if (k2x < 0) { q2x = k2x + Nx; } else { q2x = k2x; }
        q2y = k2y;

        if (k3x < 0) { q3x = k3x + Nx; } else { q3x = k3x; }
        q3y = k3y;

        // Map multi-dimensional indices to 1-dimensional indices
        long idx1 = (q1x*Ny + q1y);
        long idx2 = (q2x*Ny + q2y);
        long idx3 = (q3x*Ny + q3y);

        // Sample correlation function
        complex<float> s;
        complex<float> s1, s2, s3;

        if (cj1 == 1) {
            s1 = conj(fft1[idx1]);
        }
        else {
            s1 = fft1[idx1];
        }
        if (cj2 == 1) {
            s2 = conj(fft2[idx2]);
        }
        else {
            s2 = fft2[idx2];
        }
        if (cj3 == 1) {
            s3 = fft3[idx3];
        }
        else {
            s3 = conj(fft3[idx3]);
        }

        s = s1*s2*s3;

        bispecbuf[idx] = s;
        binormbuf[idx] = abs(s);
        countbuf[idx] = 1;

    }
}
}''')


def _printProgressBar(iteration, total, prefix='', suffix='', decimals=1,
                      length=50, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar

    Adapted from
    https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    """
    prefix = '(%d/%d)' % (iteration, total) if prefix == '' else prefix
    percent = str("%."+str(decimals)+"f") % (100 * (iteration / float(total)))
    filledLength = int(length * iteration / total)
    bar = fill * filledLength + '-' * (length - filledLength)
    prog = '\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)
    print(prog, end=printEnd, flush=True)
    if iteration == total:
        print()


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    N = 30
    np.random.seed(1234)
    data = np.random.rand(N, N)+1

    kmin, kmax = 1, 15
    result = bispectrum(data, kmin=kmin, kmax=kmax,
                        ntheta=9, double=True, progress=True, bench=True)
    bispec, bicoh, kn, theta, counts, omega = result

    print(np.nansum(bispec), np.nansum(bicoh))

    tidx = 0
    bispec, bicoh, counts = [x[tidx] for x in [bispec, bicoh, counts]]

    # Plot
    cmap = 'plasma'
    labels = [r"$|B(k_1, k_2)|$", "$b(k_1, k_2)$",
              "$arg \ B(k_1, k_2)$", "counts"]#, "counts"]
    data = [np.log10(np.abs(bispec)), np.log10(bicoh),
            np.angle(bispec), np.log10(counts)]#, np.log10(counts)]
    fig, axes = plt.subplots(ncols=2, nrows=2)
    for i in range(len(data)):
        ax = axes.flat[i]
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
