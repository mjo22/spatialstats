"""
Bispectrum GPU implementation using CuPy parallelization.

.. moduleauthor:: Michael O'Brien <michaelobrien@g.harvard.edu>

"""

import numpy as np
import cupy as cp
from cupyx.scipy import fft as cufft
from time import time


def bispectrum(*U, kmin=None, kmax=None, ntheta=None,
               nsamples=None, sample_thresh=None,
               exclude_upper=False, mean_subtract=False,
               compute_fft=True, diagnostics=False,
               double=True, blocksize=128,
               bench=False, progress=False, **kwargs):
    """
    Compute the bispectrum :math:`B(k_1, k_2, \\theta)` and
    bicoherence index :math:`b(k_1, k_2, \\theta)` of a
    scalar or vector field :math:`U` by
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
    :math:`|B(k_1, k_2, \\theta)|` normalized by the sum of
    :math:`|\hat{U}(\mathbf{k_1})\hat{U}(\mathbf{k_2})\hat{U}(\mathbf{k_1+k_2})|`.

    .. note::
        This implementation returns an average over triangles,
        rather than a sum over triangles. One can recover the
        sum by multiplying ``counts * B`` when ``nsamples = None``.
        Or, if ``ntheta = None``, evaulate ``omega * B``.

    .. note::
        When considering the bispectrum as a function of triangle
        angle, mesh points may be set to ``np.nan`` depending on
        :math:`k_1, \ k_2`. For example, :math:`\\theta = 0`
        would yield ``np.nan`` for all
        :math:`k_1 + k_2 > \\sqrt{2} k_{nyq}`, where :math:`k_{nyq}`
        is the Nyquist frequency.
        Computing a boolean mask with ``np.isnan`` and reductions
        like ``np.nansum`` can be useful.

    .. note::
        Summing ``np.nansum(B, axis=0)`` recovers the
        bispectrum summed over triangle angles.
        To recover the corresponding bicoherence, evaulate
        ``np.abs(np.nansum(B, axis=0)) / np.nansum(np.abs(B)/b, axis=0)``

    Parameters
    ----------
    U : `np.ndarray` or `cp.ndarray`
        Scalar or vector field.
        If vector data, pass arguments as ``U1, U2`` or
        ``U1, U2, U3`` where ``Ui`` is the ith vector component.
        Each ``Ui`` should be 2D or 3D (respectively), and
        must have the same ``Ui.shape`` and ``Ui.dtype``.
        If ``Ui`` are type ``cp.ndarray`` and complex-valued, it will
        by default be overwritten when taking FFTs to save memory.
        The vector bispectrum will be computed as the sum over bispectra
        of each component.
    kmin : `int`, optional
        Minimum wavenumber in bispectrum calculation.
        If ``None``, ``kmin = 1``.
    kmax : `int`, optional
        Maximum wavenumber in bispectrum calculation.
        If ``None``, ``kmax = max(U.shape)//2``
    ntheta : `int`, optional
        Number of angular bins :math:`\\theta` between triangles
        formed by wavevectors :math:`\mathbf{k_1}, \ \mathbf{k_2}`.
        If ``None``, sum over all triangle angles. Otherwise,
        return a bispectrum for each angular bin.
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
    exclude_upper : `bool`, optional
        If ``True``, set points where :math:`k_1 + k_2 > k_{nyq}`
        to ``np.nan``. This keyword only applies when summing
        over angles, e.g. when ``ntheta is None``.
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
        Bispectrum :math:`B(k_1, k_2, \\theta)`.
        Will be real-valued if the input data is real.
    b : `np.ndarray`, shape `(m, kmax-kmin+1, kmax-kmin+1)`
        Bicoherence index :math:`b(k_1, k_2, \\theta)`.
    kn : `np.ndarray`, shape `(kmax-kmin+1,)`
        Wavenumbers :math:`k_1` or :math:`k_2` along axis of bispectrum.
    theta : `np.ndarray`, shape `(m,)`, optional
        Left edges of angular bins :math:`\\theta`, ranging from
        :math:`[0, \ \\pi)`.
    omega : `np.ndarray`, shape `(kmax-kmin+1, kmax-kmin+1)`, optional
        Number of possible triangles in the sample space
        for a particular :math:`k_1, \ k_2`, unrestricted by
        the Nyquist frequency.
    counts : `np.ndarray`, shape `(m, kmax-kmin+1, kmax-kmin+1)`, optional
        Number of evaluations in the bispectrum sum.
    """

    if double:
        float, complex = cp.float64, cp.complex128
    else:
        float, complex = cp.float32, cp.complex64

    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()

    shape, ndim = U[0].shape, U[0].ndim
    ncomp = len(U)

    if ndim not in [2, 3]:
        raise ValueError("Data must be 2D or 3D.")
    if (ndim == 2 and ncomp not in [1, 2]) \
       or (ndim == 3 and ncomp not in [1, 3]):
        raise ValueError(f"{ncomp} components not valid for {ndim}-D data.")

    # Geometry of output image
    kmax = int(max(shape)/2) if kmax is None else int(kmax)
    kmin = 1 if kmin is None else int(kmin)
    kn = np.arange(kmin, kmax+1, 1, dtype=int)
    dim = kn.size
    theta = cp.arange(0, np.pi, np.pi/ntheta) if ntheta is not None else None
    # ...make costheta monotonically increase
    costheta = cp.flip(np.cos(theta)) if theta is not None else cp.array([1.])

    # theta = 0 should be included
    if theta is not None:
        costheta[-1] += 1e-5

    if bench:
        t0 = time()

    # Get binned radial coordinates of FFT
    kv = cp.meshgrid(*([cp.fft.fftfreq(Ni).astype(cp.float32)*Ni
                        for Ni in shape]), indexing="ij")
    kr = cp.zeros_like(kv[0])
    tpb = blocksize
    bpg = (kr.size + (tpb - 1)) // tpb
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
    kbinned = cp.digitize(kr.ravel(), kbins)
    kbinned[...] -= 1

    del kr
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()

    # Convert to int16
    kbinned = kbinned.astype(cp.int16)

    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()

    # FFT
    ffts = []
    for i in range(ncomp):
        if compute_fft:
            temp = cp.asarray(U[i], dtype=complex)
            if mean_subtract:
                temp[...] -= temp.mean()
            fft = _cufftn(temp, **kwargs)
            del temp
        else:
            fft = U[i].astype(complex, copy=False)
        ffts.append(fft)

    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()

    # Enumerate indices in each bin
    kind = []
    for ki in kn:
        temp = cp.where(kbinned == ki)[0].astype(cp.int64)
        kind.append(temp)

    del kbinned
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
    args = (kind, kn, costheta, kcoords, nsamples, sample_thresh,
            ndim, dim, shape, double, progress, exclude_upper,
            blocksize, compute_point, *ffts)
    B, norm, omega, counts = _compute_bispectrum(*args)

    # If input data is real, so is bispectrum.
    if np.issubdtype(U[0].dtype, np.floating):
        B = B.real

    # Set zero values to nan values for division
    mask = counts == 0.
    norm[mask] = cp.nan
    counts[mask] = cp.nan

    # Get bicoherence and average bispectrum
    b = np.abs(B) / norm
    B /= counts

    # Convert counts to integer type
    if diagnostics:
        counts = counts.astype(np.int64)
        counts[mask] = 0

    # Switch back angular bins to monotonic increasing in theta
    if ntheta is not None:
        B[...] = cp.flip(B, axis=0)
        b[...] = cp.flip(b, axis=0)
        counts[...] = cp.flip(counts, axis=0)
    else:
        B, b, counts = B[0], b[0], counts[0]

    if bench:
        print(f"Time: {time() - t0:.04f} s")

    result = [B.get(), b.get(), kn]
    if theta is not None:
        result.append(theta.get())
    if diagnostics:
        result.extend([omega, counts.get()])

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


def _compute_bispectrum(kind, kn, costheta, kcoords, nsamples, sample_thresh,
                        ndim, dim, shape, double, progress,
                        exclude, blocksize, compute_point, *ffts):
    knyq = max(shape) // 2
    shape = [cp.int16(Ni) for Ni in shape]
    ntheta = costheta.size
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
            cthetabuf = cp.zeros(count, dtype=np.float64) if ntheta > 1 \
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
                _fill_sum(i, j, bispec, binorm, counts,
                          bispecbuf, binormbuf, countbuf)
            else:
                binned = cp.searchsorted(costheta, cthetabuf)
                _fill_binned_sum(i, j, ntheta, binned,
                                 bispec, binorm, counts,
                                 bispecbuf, binormbuf, countbuf)
            omega[i, j], omega[j, i] = nk1*nk2, nk1*nk2
            del bispecbuf, binormbuf, countbuf, samp
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
        if progress:
            _printProgressBar(i, dim-1)

    return bispec, binorm, omega, counts


def _fill_sum(i, j, bispec, binorm, counts, bispecbuf, binormbuf, countbuf):
    N = countbuf.sum()
    norm = binormbuf.sum()
    value = bispecbuf.sum()
    bispec[0, i, j], bispec[0, j, i] = value, value
    binorm[0, i, j], binorm[0, j, i] = norm, norm
    counts[0, i, j], counts[0, j, i] = N, N


def _fill_binned_sum(i, j, ntheta, binned, bispec, binorm, counts,
                     bispecbuf, binormbuf, countbuf):
    N = cp.bincount(binned, weights=countbuf, minlength=ntheta)
    norm = cp.bincount(binned, weights=binormbuf, minlength=ntheta)
    value = cp.bincount(binned, weights=bispecbuf.real, minlength=ntheta) +\
        1.j*cp.bincount(binned, weights=bispecbuf.imag, minlength=ntheta)
    bispec[:, i, j], bispec[:, j, i] = value, value
    binorm[:, i, j], binorm[:, j, i] = norm, norm
    counts[:, i, j], counts[:, j, i] = N, N


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

        // Map frequency domain to index domain
        short q1x, q1y, q1z, q2x, q2y, q2z, q3x, q3y, q3z;

        if (k1x < 0) { q1x = k1x + Nx; } else { q1x = k1x; }
        if (k1y < 0) { q1y = k1y + Ny; } else { q1y = k1y; }
        if (k1z < 0) { q1z = k1z + Nz; } else { q1z = k1z; }

        if (k2x < 0) { q2x = k2x + Nx; } else { q2x = k2x; }
        if (k2y < 0) { q2y = k2y + Ny; } else { q2y = k2y; }
        if (k2z < 0) { q2z = k2z + Nz; } else { q2z = k2z; }

        if (k3x < 0) { q3x = k3x + Nx; } else { q3x = k3x; }
        if (k3y < 0) { q3y = k3y + Ny; } else { q3y = k3y; }
        if (k3z < 0) { q3z = k3z + Nz; } else { q3z = k3z; }

        // Map multi-dimensional indices to 1-dimensional indices
        long idx1 = (q1x*Nz + q1y)*Ny + q1z;
        long idx2 = (q2x*Nz + q2y)*Ny + q2z;
        long idx3 = (q3x*Nz + q3y)*Ny + q3z;

        // Sample correlation function
        complex<double> sample;
        double mod;
        sample = fft[idx1] * fft[idx2] * conj(fft[idx3]);
        mod = abs(sample);

        bispecbuf[idx] = sample;
        binormbuf[idx] = mod;
        countbuf[idx] = 1;

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

        // Map frequency domain to index domain
        short q1x, q1y, q2x, q2y, q3x, q3y;

        if (k1x < 0) { q1x = k1x + Nx; } else { q1x = k1x; }
        if (k1y < 0) { q1y = k1y + Ny; } else { q1y = k1y; }

        if (k2x < 0) { q2x = k2x + Nx; } else { q2x = k2x; }
        if (k2y < 0) { q2y = k2y + Ny; } else { q2y = k2y; }

        if (k3x < 0) { q3x = k3x + Nx; } else { q3x = k3x; }
        if (k3y < 0) { q3y = k3y + Ny; } else { q3y = k3y; }

        // Map multi-dimensional indices to 1-dimensional indices
        long idx1 = (q1x*Ny + q1y);
        long idx2 = (q2x*Ny + q2y);
        long idx3 = (q3x*Ny + q3y);

        // Sample correlation function
        complex<double> sample;
        double mod;
        sample = fft[idx1] * fft[idx2] * conj(fft[idx3]);
        mod = abs(sample);

        bispecbuf[idx] = sample;
        binormbuf[idx] = mod;
        countbuf[idx] = 1;

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
    }
}

__global__ void computePointVec3D(long* k1ind, long* k2ind,
                                  short* kx, short* ky, short* kz,
                                  int ntheta, long nk1, long nk2,
                                  short Nx, short Ny, short Nz,
                                  const long* samp, long count,
                                  complex<double>* bispecbuf, double* binormbuf,
                                  double* cthetabuf, double* countbuf,
                                  const complex<double>* fftx,
                                  const complex<double>* ffty,
                                  const complex<double>* fftz) {

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

        // Map frequency domain to index domain
        short q1x, q1y, q1z, q2x, q2y, q2z, q3x, q3y, q3z;

        if (k1x < 0) { q1x = k1x + Nx; } else { q1x = k1x; }
        if (k1y < 0) { q1y = k1y + Ny; } else { q1y = k1y; }
        if (k1z < 0) { q1z = k1z + Nz; } else { q1z = k1z; }

        if (k2x < 0) { q2x = k2x + Nx; } else { q2x = k2x; }
        if (k2y < 0) { q2y = k2y + Ny; } else { q2y = k2y; }
        if (k2z < 0) { q2z = k2z + Nz; } else { q2z = k2z; }

        if (k3x < 0) { q3x = k3x + Nx; } else { q3x = k3x; }
        if (k3y < 0) { q3y = k3y + Ny; } else { q3y = k3y; }
        if (k3z < 0) { q3z = k3z + Nz; } else { q3z = k3z; }

        // Map multi-dimensional indices to 1-dimensional indices
        long idx1 = (q1x*Nz + q1y)*Ny + q1z;
        long idx2 = (q2x*Nz + q2y)*Ny + q2z;
        long idx3 = (q3x*Nz + q3y)*Ny + q3z;

        // Sample correlation function
        complex<double> sx, sy, sz;
        sx = fftx[idx1] * fftx[idx2] * conj(fftx[idx3]);
        sy = ffty[idx1] * ffty[idx2] * conj(ffty[idx3]);
        sz = fftz[idx1] * fftz[idx2] * conj(fftz[idx3]);

        bispecbuf[idx] = sx + sy + sz;
        binormbuf[idx] = abs(sx) + abs(sy) + abs(sz);
        countbuf[idx] = 1;

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
    }
}


__global__ void computePointVec2D(const long* k1ind, const long* k2ind,
                                  short* kx, short* ky,
                                  int ntheta, long nk1, long nk2,
                                  short Nx, short Ny, const long* samp, long count,
                                  complex<double>* bispecbuf, double* binormbuf,
                                  double* cthetabuf, double* countbuf,
                                  const complex<double>* fftx,
                                  const complex<double>* ffty) {

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

        // Map frequency domain to index domain
        short q1x, q1y, q2x, q2y, q3x, q3y;

        if (k1x < 0) { q1x = k1x + Nx; } else { q1x = k1x; }
        if (k1y < 0) { q1y = k1y + Ny; } else { q1y = k1y; }

        if (k2x < 0) { q2x = k2x + Nx; } else { q2x = k2x; }
        if (k2y < 0) { q2y = k2y + Ny; } else { q2y = k2y; }

        if (k3x < 0) { q3x = k3x + Nx; } else { q3x = k3x; }
        if (k3y < 0) { q3y = k3y + Ny; } else { q3y = k3y; }

        // Map multi-dimensional indices to 1-dimensional indices
        long idx1 = (q1x*Ny + q1y);
        long idx2 = (q2x*Ny + q2y);
        long idx3 = (q3x*Ny + q3y);

        // Sample correlation function
        complex<double> sx, sy;
        sx = fftx[idx1] * fftx[idx2] * conj(fftx[idx3]);
        sy = ffty[idx1] * ffty[idx2] * conj(ffty[idx3]);

        bispecbuf[idx] = sx + sy;
        binormbuf[idx] = abs(sx) + abs(sy);
        countbuf[idx] = 1;

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

        // Map frequency domain to index domain
        short q1x, q1y, q1z, q2x, q2y, q2z, q3x, q3y, q3z;

        if (k1x < 0) { q1x = k1x + Nx; } else { q1x = k1x; }
        if (k1y < 0) { q1y = k1y + Ny; } else { q1y = k1y; }
        if (k1z < 0) { q1z = k1z + Nz; } else { q1z = k1z; }

        if (k2x < 0) { q2x = k2x + Nx; } else { q2x = k2x; }
        if (k2y < 0) { q2y = k2y + Ny; } else { q2y = k2y; }
        if (k2z < 0) { q2z = k2z + Nz; } else { q2z = k2z; }

        if (k3x < 0) { q3x = k3x + Nx; } else { q3x = k3x; }
        if (k3y < 0) { q3y = k3y + Ny; } else { q3y = k3y; }
        if (k3z < 0) { q3z = k3z + Nz; } else { q3z = k3z; }

        // Map multi-dimensional indices to 1-dimensional indices
        long idx1 = (q1x*Nz + q1y)*Ny + q1z;
        long idx2 = (q2x*Nz + q2y)*Ny + q2z;
        long idx3 = (q3x*Nz + q3y)*Ny + q3z;

        // Sample correlation function
        complex<float> sample;
        float mod;
        sample = fft[idx1] * fft[idx2] * conj(fft[idx3]);
        mod = abs(sample);

        bispecbuf[idx] = sample;
        binormbuf[idx] = mod;
        countbuf[idx] = 1;

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

        // Map frequency domain to index domain
        short q1x, q1y, q2x, q2y, q3x, q3y;

        if (k1x < 0) { q1x = k1x + Nx; } else { q1x = k1x; }
        if (k1y < 0) { q1y = k1y + Ny; } else { q1y = k1y; }

        if (k2x < 0) { q2x = k2x + Nx; } else { q2x = k2x; }
        if (k2y < 0) { q2y = k2y + Ny; } else { q2y = k2y; }

        if (k3x < 0) { q3x = k3x + Nx; } else { q3x = k3x; }
        if (k3y < 0) { q3y = k3y + Ny; } else { q3y = k3y; }

        // Map multi-dimensional indices to 1-dimensional indices
        long idx1 = (q1x*Ny + q1y);
        long idx2 = (q2x*Ny + q2y);
        long idx3 = (q3x*Ny + q3y);

        // Sample correlation function
        complex<float> sample;
        float mod;
        sample = fft[idx1] * fft[idx2] * conj(fft[idx3]);
        mod = abs(sample);

        bispecbuf[idx] = sample;
        binormbuf[idx] = mod;
        countbuf[idx] = 1;

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
    }
}

__global__ void computePointVec3Df(const long* k1ind, const long* k2ind,
                                   const short* kx, const short* ky, const short* kz,
                                   int ntheta, long nk1, long nk2,
                                   short Nx, short Ny, short Nz,
                                   const long* samp, long count,
                                   complex<float>* bispecbuf, float* binormbuf,
                                   float* cthetabuf, float* countbuf,
                                   const complex<float>* fftx,
                                   const complex<float>* ffty,
                                   const complex<float>* fftz) {

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

        // Map frequency domain to index domain
        short q1x, q1y, q1z, q2x, q2y, q2z, q3x, q3y, q3z;

        if (k1x < 0) { q1x = k1x + Nx; } else { q1x = k1x; }
        if (k1y < 0) { q1y = k1y + Ny; } else { q1y = k1y; }
        if (k1z < 0) { q1z = k1z + Nz; } else { q1z = k1z; }

        if (k2x < 0) { q2x = k2x + Nx; } else { q2x = k2x; }
        if (k2y < 0) { q2y = k2y + Ny; } else { q2y = k2y; }
        if (k2z < 0) { q2z = k2z + Nz; } else { q2z = k2z; }

        if (k3x < 0) { q3x = k3x + Nx; } else { q3x = k3x; }
        if (k3y < 0) { q3y = k3y + Ny; } else { q3y = k3y; }
        if (k3z < 0) { q3z = k3z + Nz; } else { q3z = k3z; }

        // Map multi-dimensional indices to 1-dimensional indices
        long idx1 = (q1x*Nz + q1y)*Ny + q1z;
        long idx2 = (q2x*Nz + q2y)*Ny + q2z;
        long idx3 = (q3x*Nz + q3y)*Ny + q3z;

        // Sample correlation function
        complex<float> sx, sy, sz;
        sx = fftx[idx1] * fftx[idx2] * conj(fftx[idx3]);
        sy = ffty[idx1] * ffty[idx2] * conj(ffty[idx3]);
        sz = fftz[idx1] * fftz[idx2] * conj(fftz[idx3]);

        bispecbuf[idx] = sx + sy + sz;
        binormbuf[idx] = abs(sx) + abs(sy) + abs(sz);
        countbuf[idx] = 1;

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
    }
}

__global__ void computePointVec2Df(const long* k1ind, const long* k2ind,
                                   const short* kx, const short* ky,
                                   int ntheta, long nk1, long nk2,
                                   short Nx, short Ny, const long* samp, long count,
                                   complex<float>* bispecbuf, float* binormbuf,
                                   float* cthetabuf, float* countbuf,
                                   const complex<float>* fftx,
                                   const complex<float>* ffty) {

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

        // Map frequency domain to index domain
        short q1x, q1y, q2x, q2y, q3x, q3y;

        if (k1x < 0) { q1x = k1x + Nx; } else { q1x = k1x; }
        if (k1y < 0) { q1y = k1y + Ny; } else { q1y = k1y; }

        if (k2x < 0) { q2x = k2x + Nx; } else { q2x = k2x; }
        if (k2y < 0) { q2y = k2y + Ny; } else { q2y = k2y; }

        if (k3x < 0) { q3x = k3x + Nx; } else { q3x = k3x; }
        if (k3y < 0) { q3y = k3y + Ny; } else { q3y = k3y; }

        // Map multi-dimensional indices to 1-dimensional indices
        long idx1 = (q1x*Ny + q1y);
        long idx2 = (q2x*Ny + q2y);
        long idx3 = (q3x*Ny + q3y);

        // Sample correlation function
        complex<float> sx, sy;
        sx = fftx[idx1] * fftx[idx2] * conj(fftx[idx3]);
        sy = ffty[idx1] * ffty[idx2] * conj(ffty[idx3]);

        bispecbuf[idx] = sx + sy;
        binormbuf[idx] = abs(sx) + abs(sy);
        countbuf[idx] = 1;

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

    N = 200
    np.random.seed(1234)
    data = np.random.normal(size=N**2).reshape((N, N))+1

    kmin, kmax = 1, 100
    bispec, bicoh, kn, theta, omega, counts = bispectrum(data, nsamples=None,
                                                         kmin=kmin, kmax=kmax,
                                                         ntheta=2, progress=True,
                                                         mean_subtract=True,
                                                         diagnostics=True, bench=True)
    print(np.nansum(bispec), np.nansum(bicoh))

    tidx = 1
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
