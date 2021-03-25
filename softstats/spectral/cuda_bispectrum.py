"""
Bispectrum calculation using CUDA acceleration.

This implementation works on 2D and 3D rectangular domains for real
or complex valued data. It uses uniform sampling over all
triangles in fourier space for a given two wavenumbers.

See https://turbustat.readthedocs.io/en/latest/tutorials/statistics/bispectrum_example.html for details about the bispectrum.

Requires cupy>=8.0

Author:
    Michael J. O'Brien (2020)
    Biophysical Modeling Group
    Center for Computational Biology
    Flatiron Institute
"""

import numpy as np
import cupy as cp
from time import time
from astropy.utils.console import ProgressBar
from cupyx.scipy import fft as cufft
from cupy.cuda.memory import OutOfMemoryError


def bispectrum(data, nsamples=100000, vector=False, double=True,
               mean_subtract=False, seed=None, chunks=None,
               npts=None, kmin=None, kmax=None, compute_fft=True,
               bench=False, progress=False, **kwargs):
    """
    Compute the bispectrum of 2D or 3D data with
    CUDA acceleration in single or double-precision.

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
    double : bool
        Specify whether to do calculation in single or double precision
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
        large chunks, the user may receive an OutOfMemoryError.
    npts : int
        Number of wavenumbers in bispectrum calculation
    kmin : int
        Minimum wavenumber in bispectrum calculation
    kmax : int
        Maximum wavenumber in bispectrum calculation.
        This should not exceed the dimension of data
        divided by 4.
    bench : bool
        Return compute times of calculation
    progress : bool
        Display progress bar of calculation

    **kwargs passed to cufftn (defined below)

    Returns
    -------
    bispectrum : np.ndarray
        Complex-valued 2D image
    bicoherence : np.ndarray
        Real-valued normalized bispectrum
    kn : np.ndarray
        Wavenumbers along axis of bispectrum
    """
    if double:
        float, complex = cp.float64, cp.complex128
    else:
        float, complex = cp.float32, cp.complex64
    if vector:
        temp = data[0]
        N, ndim = max(temp.shape), temp.ndim
        ncomp = data.shape[0]
        shape = temp.shape
        norm = float(temp.size)**3
    else:
        N, ndim = max(data.shape), data.ndim
        ncomp = 1
        shape = data.shape
        norm = float(data.size)**3

    temp = f"bispectrum{ndim}D" if not vector else f"bispectrumVec{ndim}D"
    func = f"{temp}f" if not double else temp
    kernel = module.get_function(func)

    if ndim not in [2, 3]:
        raise ValueError("Data must be 2D or 3D")

    if bench:
        t0 = time()

    # Set geometry of output image
    kmax = cp.int32(N/2) if kmax is None else cp.int32(kmax)
    kmin = cp.int32(1.) if kmin is None else cp.int32(kmin)
    dim = kmax-kmin+1 if npts is None else cp.int32(npts)
    chunks = int(dim) if chunks is None else chunks
    kn = cp.linspace(kmin, kmax, dim, dtype=float, endpoint=True)

    if kmax > N//2:
        raise ValueError(f"kmax should not exceed {N//2}")

    # Get memory pools
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()

    fftflat = []
    for i in range(ncomp):
        temp = data[i] if vector else data
        comp = cp.asarray(temp, dtype=complex)
        if compute_fft:
            # Subtract mean of data to highlight non-linearities
            if mean_subtract:
                comp[...] -= comp.mean()
            fft = cufftn(comp, **kwargs)
        else:
            fft = comp
        fftflat.append(fft.ravel(order='C'))
        del fft, comp
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()

    # Array of pixel indices
    npix = dim**2
    bind = cp.indices((npix,), dtype=cp.int32)[0]

    # Random samples. Uses same samples for all spheres in k-space
    if seed is not None:
        cp.random.seed(seed=seed)
    fac = 2 if ndim == 2 else 4
    samples = cp.random.uniform(size=fac*nsamples, dtype=float)

    # Package args for dispatch
    args = (samples, kn, dim, cp.int64(nsamples), *fftflat, *shape)

    def dispatch(bind, buf, out, cbuf, count):

        buf[...] = 0.
        cbuf[...] = 0.
        out[...] = 0.

        npix = cp.int64(bind.size)

        threadsperblock = 32
        blockspergrid = (nsamples*npix +
                         (threadsperblock - 1)) // threadsperblock

        birebuf, biimbuf, binormbuf = buf[0], buf[1], buf[2]
        bire, biim, binorm = out[0], out[1], out[2]

        kernel((blockspergrid,), (threadsperblock,),
               (birebuf, biimbuf, binormbuf, cbuf, bind, npix, *args))

        bire[:] = cp.sum(birebuf, axis=1)
        biim[:] = cp.sum(biimbuf, axis=1)
        binorm[:] = cp.sum(binormbuf, axis=1)
        count[:] = cp.sum(cbuf, dtype=cp.int32, axis=1)

    # Chunk size
    nchunks = int(dim**2//chunks)
    if dim**2 % chunks != 0:
        msg = f"chunks {chunks} must divide squared image dimension {dim}"
        raise ValueError(msg)

    # Catch memory errors. Larger datasets require less parallelization.
    try:
        buf = cp.zeros((3, chunks, nsamples), dtype=float)
        cbuf = cp.zeros((chunks, nsamples), dtype=cp.int32)
    except OutOfMemoryError as err:
        msg = f"Out of memory allocating buffers of shape {(chunks, nsamples)}."
        msg += " Try decreasing chunks."
        raise ValueError(msg) from err

    if progress:
        bar = ProgressBar(nchunks)

    # Calculate chunks pixels at a time
    result = cp.zeros((3, npix), dtype=float)
    out = cp.zeros((3, chunks), dtype=float)
    count = cp.zeros(chunks, dtype=cp.int32)
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
    bicoh = cp.asnumpy(cp.abs(bispectrum) / biconorm).reshape((dim, dim))
    bispec = cp.asnumpy(bispectrum).reshape((dim, dim)) / norm
    k = cp.asnumpy(kn)

    # Release memory
    del fftflat, bispectrum, samples, kn
    del buf, cbuf, result, count, out, bire, biim, bind, biconorm
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()

    if bench:
        if progress:
            print()
        print(f"Time: {time() - t0:.04f} s")

    return bispec, bicoh, k


def cufftn(data, overwrite_input=True, **kwargs):
    """
    Calculate the N-dimensional fft of an image
    with memory efficiency

    Parameters
    ----------
    data : cupy.ndarray
        Real or complex valued 2D or 3D image

    Keywords
    --------
    overwrite_input : bool
        Specify whether input data can be destroyed.
        This is useful if low on memory.
        See cupyx.scipy.fft.fftn for more.

    **kwargs passes to cupyx.scipy.fft.fftn or cupyx.scipy.fft.rfftn

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


module = cp.RawModule(code=r'''
# include <cupy/complex.cuh>

const double PI = 3.14159265358979323846;
const float PIf = 3.14159265358979323846;

extern "C" {
__global__ void bispectrum3D(double* bire, double* biim, double* biconorm,
                             int* count, int* bind, long npix,
                             const double* samples, double* kn, int dim,
                             long nsamples, const complex<double>* fft,
                             int N1, int N2, int N3) {

    long idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx > nsamples*npix - 1) { return; }

    int l = idx / nsamples;
    int k = idx % nsamples;

    int bidx = bind[l];
    int i = bidx / dim;
    int j = bidx % dim;

    double phi1, phi2, costheta1, costheta2;
    double sintheta1, sintheta2;
    double cosphi1, cosphi2;
    double sinphi1, sinphi2;

    // Read random samples
    phi1 = samples[k] * 2*PI;
    phi2 = samples[k+nsamples] * 2*PI;
    costheta1 = 2*samples[k+2*nsamples]-1;
    costheta2 = 2*samples[k+3*nsamples]-1;

    // Compute coordinates along spherical shells
    sincos(phi1, &sinphi1, &cosphi1);
    sincos(phi2, &sinphi2, &cosphi2);
    sintheta1 = sqrt(1 - costheta1*costheta1);
    sintheta2 = sqrt(1 - costheta2*costheta2);

    double k1mag, k2mag;
    double k1x, k1y, k1z, k2x, k2y, k2z, k3x, k3y, k3z;
    int k1i, k1j, k1k, k2i, k2j, k2k, k3i, k3j, k3k;

    k1mag = kn[i];
    k2mag = kn[j];
    k1x = k1mag * cosphi1 * sintheta1;
    k1y = k1mag * sinphi1 * sintheta1;
    k1z = k1mag * costheta1;
    k2x = k2mag * cosphi2 * sintheta2;
    k2y = k2mag * sinphi2 * sintheta2;
    k2z = k2mag * costheta2;
    k3x = k1x + k2x;
    k3y = k1y + k2y;
    k3z = k1z + k2z;

    // Convert frequency domain coordinates to integer
    k1i = rint(k1x); k1j = rint(k1y); k1k = rint(k1z);
    k2i = rint(k2x); k2j = rint(k2y); k2k = rint(k2z);
    k3i = rint(k3x); k3j = rint(k3y); k3k = rint(k3z);

    // Ensure we have sampled an ok region
    if ((abs(k3i) > N1/2) || (abs(k3j) > N2/2) || (abs(k3k) > N3/2)) { return; }

    __syncthreads();

    // Map frequency domain to index domain
    long q1i, q1j, q1k, q2i, q2j, q2k, q3i, q3j, q3k;

    if (k1i < 0) { q1i = k1i + N1; } else { q1i = k1i; }
    if (k1j < 0) { q1j = k1j + N2; } else { q1j = k1j; }
    if (k1k < 0) { q1k = k1k + N3; } else { q1k = k1k; }

    if (k2i < 0) { q2i = k2i + N1; } else { q2i = k2i; }
    if (k2j < 0) { q2j = k2j + N2; } else { q2j = k2j; }
    if (k2k < 0) { q2k = k2k + N3; } else { q2k = k2k; }

    if (k3i < 0) { q3i = k3i + N1; } else { q3i = k3i; }
    if (k3j < 0) { q3j = k3j + N2; } else { q3j = k3j; }
    if (k3k < 0) { q3k = k3k + N3; } else { q3k = k3k; }

    // Map multi-dimensional indices to 1-dimensional indices
    long idx1 = (q1i*N3 + q1j)*N2 + q1k;
    long idx2 = (q2i*N3 + q2j)*N2 + q2k;
    long idx3 = (q3i*N3 + q3j)*N2 + q3k;

    // Sample correlation function
    complex<double> sample;
    double mod, re, im;
    sample = fft[idx1] * fft[idx2] * conj(fft[idx3]);
    mod = abs(sample);

    re = real(sample);
    im = imag(sample);

    bire[idx] = re;
    biim[idx] = im;
    biconorm[idx] = mod;
    count[idx] = 1;

}


__global__ void bispectrum2D(double* bire, double* biim, double* biconorm,
                             int* count, int* bind, long npix,
                             const double* samples, double* kn, int dim,
                             long nsamples, const complex<double>* fft,
                             int N1, int N2) {

    long idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx > nsamples*npix - 1) { return; }

    int l = idx / nsamples;
    int k = idx % nsamples;

    int bidx = bind[l];
    int i = bidx / dim;
    int j = bidx % dim;

    double phi1, phi2;
    double cosphi1, cosphi2;
    double sinphi1, sinphi2;

    // Read random samples
    phi1 = samples[k] * 2*PI;
    phi2 = samples[k+nsamples] * 2*PI;

    // Compute coordinates along spherical shells
    sincos(phi1, &sinphi1, &cosphi1);
    sincos(phi2, &sinphi2, &cosphi2);

    double k1x, k1y, k2x, k2y, k3x, k3y;
    int k1i, k1j, k2i, k2j, k3i, k3j;
    double k1mag, k2mag;

    k1mag = kn[i];
    k2mag = kn[j];
    k1x = k1mag * cosphi1;
    k1y = k1mag * sinphi1;
    k2x = k2mag * cosphi2;
    k2y = k2mag * sinphi2;
    k3x = k1x + k2x;
    k3y = k1y + k2y;

    // Convert frequency domain coordinates to integer
    k1i = rint(k1x); k1j = rint(k1y);
    k2i = rint(k2x); k2j = rint(k2y);
    k3i = rint(k3x); k3j = rint(k3y);

    // Ensure we have sampled an ok region
    if ((abs(k3i) > N1/2) || (abs(k3j) > N2/2)) { return; }

    __syncthreads();

    // Map frequency domain to index domain
    long q1i, q1j, q2i, q2j, q3i, q3j;

    if (k1i < 0) { q1i = k1i + N1; } else { q1i = k1i; }
    if (k1j < 0) { q1j = k1j + N2; } else { q1j = k1j; }

    if (k2i < 0) { q2i = k2i + N1; } else { q2i = k2i; }
    if (k2j < 0) { q2j = k2j + N2; } else { q2j = k2j; }

    if (k3i < 0) { q3i = k3i + N1; } else { q3i = k3i; }
    if (k3j < 0) { q3j = k3j + N2; } else { q3j = k3j; }

    // Map multi-dimensional indices to 1-dimensional indices
    long idx1 = (q1i*N2 + q1j);
    long idx2 = (q2i*N2 + q2j);
    long idx3 = (q3i*N2 + q3j);

    // Sample correlation function
    complex<double> sample;
    double mod, re, im;

    sample = fft[idx1] * fft[idx2] * conj(fft[idx3]);
    mod = abs(sample);

    re = real(sample);
    im = imag(sample);

    bire[idx] = re;
    biim[idx] = im;
    biconorm[idx] = mod;
    count[idx] = 1;

}

__global__ void bispectrumVec3D(double* bire, double* biim, double* biconorm,
                                int* count, int* bind, long npix,
                                const double* samples, double* kn,
                                int dim, long nsamples,
                                const complex<double>* fftx,
                                const complex<double>* ffty,
                                const complex<double>* fftz,
                                int N1, int N2, int N3) {

    long idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx > nsamples*npix - 1) { return; }

    int l = idx / nsamples;
    int k = idx % nsamples;

    int bidx = bind[l];
    int i = bidx / dim;
    int j = bidx % dim;

    double phi1, phi2, costheta1, costheta2;
    double sintheta1, sintheta2;
    double cosphi1, cosphi2;
    double sinphi1, sinphi2;

    // Read random samples
    phi1 = samples[k] * 2*PI;
    phi2 = samples[k+nsamples] * 2*PI;
    costheta1 = 2*samples[k+2*nsamples]-1;
    costheta2 = 2*samples[k+3*nsamples]-1;

    // Compute coordinates along spherical shells
    sincos(phi1, &sinphi1, &cosphi1);
    sincos(phi2, &sinphi2, &cosphi2);
    sintheta1 = sqrt(1 - costheta1*costheta1);
    sintheta2 = sqrt(1 - costheta2*costheta2);

    double k1mag, k2mag;
    double k1x, k1y, k1z, k2x, k2y, k2z, k3x, k3y, k3z;
    int k1i, k1j, k1k, k2i, k2j, k2k, k3i, k3j, k3k;

    k1mag = kn[i];
    k2mag = kn[j];
    k1x = k1mag * cosphi1 * sintheta1;
    k1y = k1mag * sinphi1 * sintheta1;
    k1z = k1mag * costheta1;
    k2x = k2mag * cosphi2 * sintheta2;
    k2y = k2mag * sinphi2 * sintheta2;
    k2z = k2mag * costheta2;
    k3x = k1x + k2x;
    k3y = k1y + k2y;
    k3z = k1z + k2z;

    // Convert frequency domain coordinates to integer
    k1i = rint(k1x); k1j = rint(k1y); k1k = rint(k1z);
    k2i = rint(k2x); k2j = rint(k2y); k2k = rint(k2z);
    k3i = rint(k3x); k3j = rint(k3y); k3k = rint(k3z);

    // Ensure we have sampled an ok region
    if ((abs(k3i) > N1/2) || (abs(k3j) > N2/2) || (abs(k3k) > N3/2)) { return; }

    __syncthreads();

    // Map frequency domain to index domain
    long q1i, q1j, q1k, q2i, q2j, q2k, q3i, q3j, q3k;

    if (k1i < 0) { q1i = k1i + N1; } else { q1i = k1i; }
    if (k1j < 0) { q1j = k1j + N2; } else { q1j = k1j; }
    if (k1k < 0) { q1k = k1k + N3; } else { q1k = k1k; }

    if (k2i < 0) { q2i = k2i + N1; } else { q2i = k2i; }
    if (k2j < 0) { q2j = k2j + N2; } else { q2j = k2j; }
    if (k2k < 0) { q2k = k2k + N3; } else { q2k = k2k; }

    if (k3i < 0) { q3i = k3i + N1; } else { q3i = k3i; }
    if (k3j < 0) { q3j = k3j + N2; } else { q3j = k3j; }
    if (k3k < 0) { q3k = k3k + N3; } else { q3k = k3k; }

    // Map multi-dimensional indices to 1-dimensional indices
    long idx1 = (q1i*N3 + q1j)*N2 + q1k;
    long idx2 = (q2i*N3 + q2j)*N2 + q2k;
    long idx3 = (q3i*N3 + q3j)*N2 + q3k;

    // Sample correlation function
    complex<double> sample, samplex, sampley, samplez;
    double mod, re, im;
    
    samplex = fftx[idx1] * fftx[idx2] * conj(fftx[idx3]);
    sampley = ffty[idx1] * ffty[idx2] * conj(ffty[idx3]);
    samplez = fftz[idx1] * fftz[idx2] * conj(fftz[idx3]);

    sample = samplex + sampley + samplez;
    mod = abs(sample);

    re = real(sample);
    im = imag(sample);

    bire[idx] = re;
    biim[idx] = im;
    biconorm[idx] = mod;
    count[idx] = 1;

}

__global__ void bispectrumVec2D(double* bire, double* biim, double* biconorm,
                                int* count, int* bind, long npix,
                                const double* samples, double* kn,
                                int dim, long nsamples,
                                const complex<double>* fftx,
                                const complex<double>* ffty,
                                int N1, int N2) {

    long idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx > nsamples*npix - 1) { return; }

    int l = idx / nsamples;
    int k = idx % nsamples;

    int bidx = bind[l];
    int i = bidx / dim;
    int j = bidx % dim;

    double phi1, phi2;
    double cosphi1, cosphi2;
    double sinphi1, sinphi2;

    // Read random samples
    phi1 = samples[k] * 2*PI;
    phi2 = samples[k+nsamples] * 2*PI;

    // Compute coordinates along spherical shells
    sincos(phi1, &sinphi1, &cosphi1);
    sincos(phi2, &sinphi2, &cosphi2);

    double k1x, k1y, k2x, k2y, k3x, k3y;
    int k1i, k1j, k2i, k2j, k3i, k3j;
    double k1mag, k2mag;

    k1mag = kn[i];
    k2mag = kn[j];
    k1x = k1mag * cosphi1;
    k1y = k1mag * sinphi1;
    k2x = k2mag * cosphi2;
    k2y = k2mag * sinphi2;
    k3x = k1x + k2x;
    k3y = k1y + k2y;

    // Convert frequency domain coordinates to integer
    k1i = rint(k1x); k1j = rint(k1y);
    k2i = rint(k2x); k2j = rint(k2y);
    k3i = rint(k3x); k3j = rint(k3y);

    // Ensure we have sampled an ok region
    if ((abs(k3i) > N1/2) || (abs(k3j) > N2/2)) { return; }

    __syncthreads();

    // Map frequency domain to index domain
    long q1i, q1j, q2i, q2j, q3i, q3j;

    if (k1i < 0) { q1i = k1i + N1; } else { q1i = k1i; }
    if (k1j < 0) { q1j = k1j + N2; } else { q1j = k1j; }

    if (k2i < 0) { q2i = k2i + N1; } else { q2i = k2i; }
    if (k2j < 0) { q2j = k2j + N2; } else { q2j = k2j; }

    if (k3i < 0) { q3i = k3i + N1; } else { q3i = k3i; }
    if (k3j < 0) { q3j = k3j + N2; } else { q3j = k3j; }

    // Map multi-dimensional indices to 1-dimensional indices
    long idx1 = (q1i*N2 + q1j);
    long idx2 = (q2i*N2 + q2j);
    long idx3 = (q3i*N2 + q3j);

    // Sample correlation function
    complex<double> samplex, sampley, sample; 
    double mod, re, im;
    samplex = fftx[idx1] * fftx[idx2] * conj(fftx[idx3]);
    sampley = ffty[idx1] * ffty[idx2] * conj(ffty[idx3]);

    sample = samplex + sampley;
    mod = abs(sample);

    re = real(sample);
    im = imag(sample);

    bire[idx] = re;
    biim[idx] = im;
    biconorm[idx] = mod;
    count[idx] = 1;

}


__global__ void bispectrum3Df(float* bire, float* biim, float* biconorm,
                              int* count, int* bind, long npix,
                              const float* samples, float* kn, int dim,
                              long nsamples, const complex<float>* fft,
                              int N1, int N2, int N3) {

    long idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx > nsamples*npix - 1) { return; }

    int l = idx / nsamples;
    int k = idx % nsamples;

    int bidx = bind[l];
    int i = bidx / dim;
    int j = bidx % dim;

    float phi1, phi2, costheta1, costheta2;
    float sintheta1, sintheta2;
    float cosphi1, cosphi2;
    float sinphi1, sinphi2;

    // Read random samples
    phi1 = samples[k] * 2*PIf;
    phi2 = samples[k+nsamples] * 2*PIf;
    costheta1 = 2*samples[k+2*nsamples]-1;
    costheta2 = 2*samples[k+3*nsamples]-1;

    // Compute coordinates along spherical shells
    sincos(phi1, &sinphi1, &cosphi1);
    sincos(phi2, &sinphi2, &cosphi2);
    sintheta1 = sqrt(1 - costheta1*costheta1);
    sintheta2 = sqrt(1 - costheta2*costheta2);

    float k1mag, k2mag;
    float k1x, k1y, k1z, k2x, k2y, k2z, k3x, k3y, k3z;
    int k1i, k1j, k1k, k2i, k2j, k2k, k3i, k3j, k3k;

    k1mag = kn[i];
    k2mag = kn[j];
    k1x = k1mag * cosphi1 * sintheta1;
    k1y = k1mag * sinphi1 * sintheta1;
    k1z = k1mag * costheta1;
    k2x = k2mag * cosphi2 * sintheta2;
    k2y = k2mag * sinphi2 * sintheta2;
    k2z = k2mag * costheta2;
    k3x = k1x + k2x;
    k3y = k1y + k2y;
    k3z = k1z + k2z;

    // Convert frequency domain coordinates to integer
    k1i = rint(k1x); k1j = rint(k1y); k1k = rint(k1z);
    k2i = rint(k2x); k2j = rint(k2y); k2k = rint(k2z);
    k3i = rint(k3x); k3j = rint(k3y); k3k = rint(k3z);

    // Ensure we have sampled an ok region
    if ((abs(k3i) > N1/2) || (abs(k3j) > N2/2) || (abs(k3k) > N3/2)) { return; }

    __syncthreads();

    // Map frequency domain to index domain
    long q1i, q1j, q1k, q2i, q2j, q2k, q3i, q3j, q3k;

    if (k1i < 0) { q1i = k1i + N1; } else { q1i = k1i; }
    if (k1j < 0) { q1j = k1j + N2; } else { q1j = k1j; }
    if (k1k < 0) { q1k = k1k + N3; } else { q1k = k1k; }

    if (k2i < 0) { q2i = k2i + N1; } else { q2i = k2i; }
    if (k2j < 0) { q2j = k2j + N2; } else { q2j = k2j; }
    if (k2k < 0) { q2k = k2k + N3; } else { q2k = k2k; }

    if (k3i < 0) { q3i = k3i + N1; } else { q3i = k3i; }
    if (k3j < 0) { q3j = k3j + N2; } else { q3j = k3j; }
    if (k3k < 0) { q3k = k3k + N3; } else { q3k = k3k; }

    // Map multi-dimensional indices to 1-dimensional indices
    long idx1 = (q1i*N3 + q1j)*N2 + q1k;
    long idx2 = (q2i*N3 + q2j)*N2 + q2k;
    long idx3 = (q3i*N3 + q3j)*N2 + q3k;

    // Sample correlation function
    complex<float> sample;
    float mod, re, im;
    sample = fft[idx1] * fft[idx2] * conj(fft[idx3]);
    mod = abs(sample);

    re = real(sample);
    im = imag(sample);

    bire[idx] = re;
    biim[idx] = im;
    biconorm[idx] = mod;
    count[idx] = 1;

}


__global__ void bispectrum2Df(float* bire, float* biim, float* biconorm,
                              int* count, int* bind, long npix,
                              const float* samples, float* kn, int dim,
                              long nsamples, const complex<float>* fft,
                              int N1, int N2) {

    long idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx > nsamples*npix - 1) { return; }

    int l = idx / nsamples;
    int k = idx % nsamples;

    int bidx = bind[l];
    int i = bidx / dim;
    int j = bidx % dim;

    float phi1, phi2;
    float cosphi1, cosphi2;
    float sinphi1, sinphi2;

    // Read random samples
    phi1 = samples[k] * 2*PIf;
    phi2 = samples[k+nsamples] * 2*PIf;

    // Compute coordinates along spherical shells
    sincos(phi1, &sinphi1, &cosphi1);
    sincos(phi2, &sinphi2, &cosphi2);

    float k1x, k1y, k2x, k2y, k3x, k3y;
    int k1i, k1j, k2i, k2j, k3i, k3j;
    float k1mag, k2mag;

    k1mag = kn[i];
    k2mag = kn[j];
    k1x = k1mag * cosphi1;
    k1y = k1mag * sinphi1;
    k2x = k2mag * cosphi2;
    k2y = k2mag * sinphi2;
    k3x = k1x + k2x;
    k3y = k1y + k2y;

    // Convert frequency domain coordinates to integer
    k1i = rint(k1x); k1j = rint(k1y);
    k2i = rint(k2x); k2j = rint(k2y);
    k3i = rint(k3x); k3j = rint(k3y);

    // Ensure we have sampled an ok region
    if ((abs(k3i) > N1/2) || (abs(k3j) > N2/2)) { return; }

    __syncthreads();

    // Map frequency domain to index domain
    long q1i, q1j, q2i, q2j, q3i, q3j;

    if (k1i < 0) { q1i = k1i + N1; } else { q1i = k1i; }
    if (k1j < 0) { q1j = k1j + N2; } else { q1j = k1j; }

    if (k2i < 0) { q2i = k2i + N1; } else { q2i = k2i; }
    if (k2j < 0) { q2j = k2j + N2; } else { q2j = k2j; }

    if (k3i < 0) { q3i = k3i + N1; } else { q3i = k3i; }
    if (k3j < 0) { q3j = k3j + N2; } else { q3j = k3j; }

    // Map multi-dimensional indices to 1-dimensional indices
    long idx1 = (q1i*N2 + q1j);
    long idx2 = (q2i*N2 + q2j);
    long idx3 = (q3i*N2 + q3j);

    // Sample correlation function
    complex<float> sample;
    float mod, re, im;
    sample = fft[idx1] * fft[idx2] * conj(fft[idx3]);
    mod = abs(sample);

    re = real(sample);
    im = imag(sample);

    bire[idx] = re;
    biim[idx] = im;
    biconorm[idx] = mod;
    count[idx] = 1;

}


__global__ void bispectrumVec3Df(float* bire, float* biim, float* biconorm,
                                 int* count, int* bind, long npix,
                                 const float* samples, float* kn,
                                 int dim, long nsamples,
                                 const complex<float>* fftx,
                                 const complex<float>* ffty,
                                 const complex<float>* fftz,
                                 int N1, int N2, int N3) {

    long idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx > nsamples*npix - 1) { return; }

    int l = idx / nsamples;
    int k = idx % nsamples;

    int bidx = bind[l];
    int i = bidx / dim;
    int j = bidx % dim;

    float phi1, phi2, costheta1, costheta2;
    float sintheta1, sintheta2;
    float cosphi1, cosphi2;
    float sinphi1, sinphi2;

    // Read random samples
    phi1 = samples[k] * 2*PIf;
    phi2 = samples[k+nsamples] * 2*PIf;
    costheta1 = 2*samples[k+2*nsamples]-1;
    costheta2 = 2*samples[k+3*nsamples]-1;

    // Compute coordinates along spherical shells
    sincos(phi1, &sinphi1, &cosphi1);
    sincos(phi2, &sinphi2, &cosphi2);
    sintheta1 = sqrt(1 - costheta1*costheta1);
    sintheta2 = sqrt(1 - costheta2*costheta2);

    float k1mag, k2mag;
    float k1x, k1y, k1z, k2x, k2y, k2z, k3x, k3y, k3z;
    int k1i, k1j, k1k, k2i, k2j, k2k, k3i, k3j, k3k;

    k1mag = kn[i];
    k2mag = kn[j];
    k1x = k1mag * cosphi1 * sintheta1;
    k1y = k1mag * sinphi1 * sintheta1;
    k1z = k1mag * costheta1;
    k2x = k2mag * cosphi2 * sintheta2;
    k2y = k2mag * sinphi2 * sintheta2;
    k2z = k2mag * costheta2;
    k3x = k1x + k2x;
    k3y = k1y + k2y;
    k3z = k1z + k2z;

    // Convert frequency domain coordinates to integer
    k1i = rint(k1x); k1j = rint(k1y); k1k = rint(k1z);
    k2i = rint(k2x); k2j = rint(k2y); k2k = rint(k2z);
    k3i = rint(k3x); k3j = rint(k3y); k3k = rint(k3z);

    // Ensure we have sampled an ok region
    if ((abs(k3i) > N1/2) || (abs(k3j) > N2/2) || (abs(k3k) > N3/2)) { return; }

    __syncthreads();

    // Map frequency domain to index domain
    long q1i, q1j, q1k, q2i, q2j, q2k, q3i, q3j, q3k;

    if (k1i < 0) { q1i = k1i + N1; } else { q1i = k1i; }
    if (k1j < 0) { q1j = k1j + N2; } else { q1j = k1j; }
    if (k1k < 0) { q1k = k1k + N3; } else { q1k = k1k; }

    if (k2i < 0) { q2i = k2i + N1; } else { q2i = k2i; }
    if (k2j < 0) { q2j = k2j + N2; } else { q2j = k2j; }
    if (k2k < 0) { q2k = k2k + N3; } else { q2k = k2k; }

    if (k3i < 0) { q3i = k3i + N1; } else { q3i = k3i; }
    if (k3j < 0) { q3j = k3j + N2; } else { q3j = k3j; }
    if (k3k < 0) { q3k = k3k + N3; } else { q3k = k3k; }

    // Map multi-dimensional indices to 1-dimensional indices
    long idx1 = (q1i*N3 + q1j)*N2 + q1k;
    long idx2 = (q2i*N3 + q2j)*N2 + q2k;
    long idx3 = (q3i*N3 + q3j)*N2 + q3k;

    // Sample correlation function
    complex<float> sample, samplex, sampley, samplez;
    float mod, re, im;

    samplex = fftx[idx1] * fftx[idx2] * conj(fftx[idx3]);
    sampley = ffty[idx1] * ffty[idx2] * conj(ffty[idx3]);
    samplez = fftz[idx1] * fftz[idx2] * conj(fftz[idx3]);

    sample = samplex + sampley + samplez;
    mod = abs(sample);

    re = real(sample);
    im = imag(sample);

    bire[idx] = re;
    biim[idx] = im;
    biconorm[idx] = mod;
    count[idx] = 1;

}


__global__ void bispectrumVec2Df(float* bire, float* biim, float* biconorm,
                                 int* count, int* bind,
                                 long npix, const float* samples,
                                 float* kn, int dim, long nsamples,
                                 const complex<float>* fftx,
                                 const complex<float>* ffty,
                                 int N1, int N2) {

    long idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx > nsamples*npix - 1) { return; }

    int l = idx / nsamples;
    int k = idx % nsamples;

    int bidx = bind[l];
    int i = bidx / dim;
    int j = bidx % dim;

    float phi1, phi2;
    float cosphi1, cosphi2;
    float sinphi1, sinphi2;

    // Read random samples
    phi1 = samples[k] * 2*PIf;
    phi2 = samples[k+nsamples] * 2*PIf;

    // Compute coordinates along spherical shells
    sincos(phi1, &sinphi1, &cosphi1);
    sincos(phi2, &sinphi2, &cosphi2);

    float k1x, k1y, k2x, k2y, k3x, k3y;
    int k1i, k1j, k2i, k2j, k3i, k3j;
    float k1mag, k2mag;

    k1mag = kn[i];
    k2mag = kn[j];
    k1x = k1mag * cosphi1;
    k1y = k1mag * sinphi1;
    k2x = k2mag * cosphi2;
    k2y = k2mag * sinphi2;
    k3x = k1x + k2x;
    k3y = k1y + k2y;

    // Convert frequency domain coordinates to integer
    k1i = rint(k1x); k1j = rint(k1y);
    k2i = rint(k2x); k2j = rint(k2y);
    k3i = rint(k3x); k3j = rint(k3y);

    // Ensure we have sampled an ok region
    if ((abs(k3i) > N1/2) || (abs(k3j) > N2/2)) { return; }

    __syncthreads();

    // Map frequency domain to index domain
    long q1i, q1j, q2i, q2j, q3i, q3j;

    if (k1i < 0) { q1i = k1i + N1; } else { q1i = k1i; }
    if (k1j < 0) { q1j = k1j + N2; } else { q1j = k1j; }

    if (k2i < 0) { q2i = k2i + N1; } else { q2i = k2i; }
    if (k2j < 0) { q2j = k2j + N2; } else { q2j = k2j; }

    if (k3i < 0) { q3i = k3i + N1; } else { q3i = k3i; }
    if (k3j < 0) { q3j = k3j + N2; } else { q3j = k3j; }

    // Map multi-dimensional indices to 1-dimensional indices
    long idx1 = (q1i*N2 + q1j);
    long idx2 = (q2i*N2 + q2j);
    long idx3 = (q3i*N2 + q3j);

    // Sample correlation function
    complex<float> samplex, sampley, sample; 
    float mod, re, im;
    samplex = fftx[idx1] * fftx[idx2] * conj(fftx[idx3]);
    sampley = ffty[idx1] * ffty[idx2] * conj(ffty[idx3]);

    sample = samplex + sampley;
    mod = abs(sample);

    re = real(sample);
    im = imag(sample);

    bire[idx] = re;
    biim[idx] = im;
    biconorm[idx] = mod;
    count[idx] = 1;

}

}''')


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Generate noise
    N = 128
    data = np.random.normal(size=N**2).reshape((N, N))

    # Calculate
    bispec, bicoh, kn = bispectrum(data,
                                   vector=False,
                                   double=True,
                                   chunks=64,
                                   nsamples=int(1e5),
                                   progress=True,
                                   mean_subtract=True)
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
