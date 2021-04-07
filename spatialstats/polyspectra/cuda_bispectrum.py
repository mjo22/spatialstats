"""
Bisprectrum calculation using CUDA acceleration

This implementation works on 2D and 3D rectangular domains for real
or complex valued data. It can compute the bispectrum exactly or
using uniform sampling.

Author:
    Michael J. O'Brien (2021)
    Biophysical Modeling Group
    Center for Computational Biology
    Flatiron Institute
"""

import numpy as np
import cupy as cp
from cupyx.scipy import fft as cufft
from time import time


def bispectrum(data, kmin=None, kmax=None, nsamples=None, sample_thresh=None,
               double=True, exclude=False, mean_subtract=False, blocksize=128,
               compute_fft=True, bench=False, progress=False, **kwargs):
    """
    Compute the bispectrum of 2D or 3D real or complex valued data.

    Arguments
    ---------
    data : np.ndarray
        Real or complex valued 2D or 3D scalar data.
        The shape should be (d1, d2) or (d1, d2, d3)
        where di is the ith dimension of the image.
        Can be CPU or GPU data. If it is GPU data and
        complex, it will be overwritten by default.

    Keywords
    --------
    kmin : int
        Minimum wavenumber in bispectrum calculation.
    kmax : int
        Maximum wavenumber in bispectrum calculation.
    nsamples : int, float, or np.ndarray
        Number of sample triangles to take. This may be
        an array of shape [kmax-kmin+1, kmax-kmin+1] to
        specify either 1) the number of samples to take
        for a given point or 2) the fraction of total
        possible triangles to sample. If None, calculate
        the bispectrum exactly.
    sample_thresh : int
        When the size of the sample space is greater than
        this number, start to use sampling instead of exact
        calculation. If None, switch to exact calculation
        when nsamples is less than the size of the sample space.
    double : bool
        If False, do calculation in single precision.
    exclude : bool
        If True, exclude k1, k2 such that k1 + k2 is greater
        than the Nyquist frequency. Excluded points will be
        set to nan.
    mean_subtract : bool
        Subtract mean off of image data to highlight
        non-linearities in bicoherence.
    compute_fft : bool
        If False, do not take the FFT of the input data.
    bench : bool
        If True, print calculation time.

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

    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()

    shape, ndim = data.shape, data.ndim
    norm = float(data.size)**3

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
    kv = cp.meshgrid(*([cp.fft.fftfreq(Ni).astype(cp.float32)*Ni
                        for Ni in shape]), indexing="ij")
    kr = cp.zeros_like(kv[0])
    tpb = blocksize
    bpg = (kr.size + (tpb - 1)) // tpb
    for i in range(ndim):
        sqr_add((bpg,), (tpb,), (kr, kv[i], kr.size))
    sqrt((bpg,), (tpb,), (kr, kr.size))

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
    if compute_fft:
        temp = cp.asarray(data, dtype=complex)
        if mean_subtract:
            temp[...] -= temp.mean()
        fft = cufftn(temp, **kwargs)
        del temp
    else:
        fft = data.astype(complex, copy=False)

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
    f = "" if double else "f"
    compute_point = module.get_function(f"compute_point{ndim}D{f}")
    bispec, binorm = compute_bispectrum(kind, kn, kcoords, fft,
                                        nsamples, sample_thresh,
                                        ndim, dim, shape, double, progress,
                                        exclude, blocksize, compute_point)

    bicoh = cp.abs(bispec) / binorm
    bispec /= norm

    if bench:
        print(f"Time: {time() - t0:.04f} s")

    return bispec.get(), bicoh.get(), kn


def cufftn(data, overwrite_input=True, **kwargs):
    """
    Calculate the N-dimensional fft of an image
    with memory efficiency

    Parameters
    ----------
    data : cupy.ndarray
        Real or complex valued 2D or 3D image.

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


sqr_add = cp.RawKernel(r'''
#include <cupy/carray.cuh>

extern "C" __global__

void square_add(float* kr, float* ki, int size) {

    for (long idx = blockDim.x * blockIdx.x + threadIdx.x;
         idx < size;
         idx += blockDim.x * gridDim.x) {

        kr[idx] += ki[idx]*ki[idx];

    }

}

''', 'square_add')


sqrt = cp.RawKernel(r'''
#include <cupy/carray.cuh>

extern "C" __global__

void square_root(float* kr, int size) {

    for (long idx = blockDim.x * blockIdx.x + threadIdx.x;
         idx < size;
         idx += blockDim.x * gridDim.x) {

        kr[idx] = sqrt(kr[idx]);

    }

}

''', 'square_root')


def compute_bispectrum(kind, kn, kcoords, fft, nsamples, sample_thresh,
                       ndim, dim, shape, double, progress,
                       exclude, blocksize, compute_point):
    knyq = max(shape) // 2
    shape = [cp.int16(Ni) for Ni in shape]
    if double:
        float, complex = cp.float64, cp.complex128
    else:
        float, complex = cp.float32, cp.complex64
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    bispec = cp.full((dim, dim), cp.nan+1.j*cp.nan, dtype=complex)
    binorm = cp.full((dim, dim), cp.nan, dtype=float)
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
            countbuf = cp.zeros(count, dtype=cp.int16)
            compute_point((bpg,), (tpb,), (k1ind, k2ind, *kcoords, fft,
                                           cp.int64(nk1), cp.int64(nk2),
                                           *shape, samp, cp.int64(count),
                                           bispecbuf, binormbuf, countbuf))
            N = countbuf.sum()
            value = nk1*nk2*(bispecbuf.sum() / N)
            norm = nk1*nk2*(binormbuf.sum() / N)
            bispec[i, j], bispec[j, i] = value, value
            binorm[i, j], binorm[j, i] = norm, norm
            del bispecbuf, binormbuf, countbuf, samp
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()
        if progress:
            printProgressBar(i, dim-1)

    return bispec, binorm


module = cp.RawModule(code=r'''
# include <cupy/complex.cuh>

extern "C" {
__global__ void compute_point3D(long* k1ind, long* k2ind,
                                short* kx, short* ky, short* kz,
                                const complex<double>* fft, long nk1, long nk2,
                                short Nx, short Ny, short Nz,
                                const long* samp, long count,
                                complex<double>* bispecbuf, double* binormbuf,
                                short* countbuf) {

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

    }
}


__global__ void compute_point2D(const long* k1ind, const long* k2ind,
                                short* kx, short* ky,
                                const complex<double>* fft, long nk1, long nk2,
                                short Nx, short Ny, const long* samp, long count,
                                complex<double>* bispecbuf, double* binormbuf,
                                short* countbuf) {

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

    }

}


__global__ void compute_point3Df(const long* k1ind, const long* k2ind,
                                 const short* kx, const short* ky, const short* kz,
                                 const complex<float>* fft, long nk1, long nk2,
                                short Nx, short Ny, short Nz,
                                const long* samp, long count,
                                complex<float>* bispecbuf, float* binormbuf,
                                short* countbuf) {

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

    }
}


__global__ void compute_point2Df(const long* k1ind, const long* k2ind,
                                 const short* kx, const short* ky,
                                 const complex<float>* fft, long nk1, long nk2,
                                 short Nx, short Ny, const long* samp, long count,
                                 complex<float>* bispecbuf, float* binormbuf,
                                 short* countbuf) {

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

    }
}
}''')


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1,
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

    N = 512
    data = np.random.normal(size=N**2).reshape((N, N))+1

    kmin, kmax = 1, 32
    bispec, bicoh, kn = bispectrum(data, nsamples=int(5e7),
                                   kmin=kmin, kmax=kmax, progress=True,
                                   mean_subtract=True, bench=True, exclude=True)
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
