"""
Bisprectrum calculation using numba acceleration

This implementation works on 2D and 3D rectangular domains for real
or complex valued data. It includes code for exact bispectrum
calculations and bispectrum calculations using uniform sampling.

See https://turbustat.readthedocs.io/en/latest/tutorials/statistics/bispectrum_example.html for more details.

Author:
    Michael J. O'Brien (2020)
    Biophysical Modeling Group
    Center for Computational Biology
    Flatiron Institute
"""

import numpy as np
import cupy as cp
from numba import cuda
from cupyx.scipy import fft as cufft
from time import time


def bispectrum(data, kmin=None, kmax=None, nsamples=None, double=True,
               mean_subtract=False, compute_fft=True, bench=False, **kwargs):
    """
    Compute bispectrum in 2D or 3D.

    Arguments
    ---------
    data : np.ndarray
        Real or complex valued 2D or 3D scalar data.
        The shape should be (d1, d2) or (d1, d2, d3)
        where di is the ith dimension of the image.

    Keywords
    --------
    kmin : int
        Minimum wavenumber in bispectrum calculation.
    kmax : int
        Maximum wavenumber in bispectrum calculation.
    mean_subtract : bool
        Subtract mean off of image data to highlight
        non-linearities in bicoherence.
    compute_fft : bool
        If False, do not take the FFT of the input data.

    **kwargs passed to fftn (defined below)

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

    shape, ndim = data.shape, data.ndim
    norm = float(data.size)**3

    if ndim not in [2, 3]:
        raise ValueError("Image must be a 2D or 3D")

    # Geometry of output image
    kmax = int(max(shape)/2) if kmax is None else int(kmax)
    kmin = 1 if kmin is None else int(kmin)
    kn = np.arange(kmin, kmax+1, 1, dtype=int)
    dim = kn.size

    if bench:
        t0 = time()

    # FFT
    if compute_fft:
        temp = cp.asarray(data, dtype=complex)
        if mean_subtract:
            temp[...] = temp - temp.mean()
        fft = cufftn(temp, **kwargs)
    else:
        fft = data

    del temp

    # Get binned radial coordinates of FFT
    kv = cp.meshgrid(*([cp.fft.fftfreq(Ni).astype(np.float32)*Ni
                        for Ni in shape]), indexing="ij")
    kr = cp.zeros_like(kv[0])
    for i in range(ndim):
        kr[...] += kv[i]**2
    kr[...] = cp.sqrt(kr)

    kcoords = []
    for i in range(ndim):
        temp = kv[i].ravel().astype(cp.int16)
        kcoords.append(temp)

    del kv, temp

    kbins = cp.arange(int(np.ceil(kr.max().get())))
    kbinned = (cp.digitize(kr.ravel(), kbins)-1).astype(cp.int16)

    del kr

    # Enumerate indices in each bin
    kind = []
    for ki in kn:
        temp = cp.where(kbinned == ki)[0].astype(cp.int64)
        kind.append(temp)

    del kbinned

    if nsamples is None:
        nsamples = np.iinfo(np.int64).max
    if np.issubdtype(type(nsamples), np.integer):
        nsamples = np.full((dim, dim), nsamples, dtype=int)

    # Run main loop
    f = "" if double else "f"
    compute_pixel = module.get_function(f"compute_pixel{ndim}D{f}")
    bispec, binorm = compute_bispectrum(kind, kcoords, fft, nsamples,
                                        ndim, dim, shape, double,
                                        compute_pixel)

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


def compute_bispectrum(kind, kcoords, fft, nsamples,
                       ndim, dim, shape, double, compute_pixel):
    shape = [cp.int16(Ni) for Ni in shape]
    if double:
        float, complex = cp.float64, cp.complex128
    else:
        float, complex = cp.float32, cp.complex64
    bispec = cp.zeros((dim, dim), dtype=cp.complex128)
    binorm = cp.zeros((dim, dim), dtype=cp.float64)
    for i in range(dim):
        k1ind = kind[i]
        nk1 = k1ind.size
        for j in range(i+1):
            k2ind = kind[j]
            nk2 = k2ind.size
            nsamp = int(nsamples[i, j])
            if nsamp < nk1*nk2:
                samp = cp.random.randint(0, nk1*nk2, size=nsamp, dtype=cp.int64)
                count = nsamp
            else:
                samp = cp.arange(nk1*nk2, dtype=cp.int64)
                count = nk1*nk2
            tpb = 32
            bpg = (count + (tpb - 1)) // tpb
            bispecbuf = cp.zeros(count, dtype=complex)
            binormbuf = cp.zeros(count, dtype=float)
            countbuf = cp.zeros(count, dtype=cp.int16)
            compute_pixel((bpg,), (tpb,), (k1ind, k2ind, *kcoords, fft,
                                           cp.int64(nk1), cp.int64(nk2),
                                           *shape, samp, cp.int64(count),
                                           bispecbuf, binormbuf, countbuf))
            value = bispecbuf.sum() / countbuf.sum()
            norm = binormbuf.sum() / countbuf.sum()
            bispec[i, j], bispec[j, i] = value, value
            binorm[i, j], binorm[j, i] = norm, norm
    return bispec, binorm


module = cp.RawModule(code=r'''
# include <cupy/complex.cuh>

extern "C" {
__global__ void compute_pixel3D(long* k1ind, long* k2ind,
                                short* kx, short* ky, short* kz,
                                const complex<double>* fft, long nk1, long nk2,
                                short Nx, short Ny, short Nz,
                                const long* samp, long count,
                                complex<double>* bispecbuf, double* binormbuf,
                                short* countbuf) {
    long idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx > count - 1) { return; }

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


__global__ void compute_pixel2D(const long* k1ind, const long* k2ind,
                                short* kx, short* ky,
                                const complex<double>* fft, long nk1, long nk2,
                                short Nx, short Ny, const long* samp, long count,
                                complex<double>* bispecbuf, double* binormbuf,
                                short* countbuf) {
    long idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx > count - 1) { return; }

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


__global__ void compute_pixel3Df(const long* k1ind, const long* k2ind,
                                 const short* kx, const short* ky, const short* kz,
                                 const complex<float>* fft, long nk1, long nk2,
                                short Nx, short Ny, short Nz,
                                const long* samp, long count,
                                complex<float>* bispecbuf, float* binormbuf,
                                short* countbuf) {
    long idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx > count - 1) { return; }

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


__global__ void compute_pixel2Df(const long* k1ind, const long* k2ind,
                                 const short* kx, const short* ky,
                                 const complex<float>* fft, long nk1, long nk2,
                                 short Nx, short Ny, const long* samp, long count,
                                 complex<float>* bispecbuf, float* binormbuf,
                                 short* countbuf) {
    long idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx > count - 1) { return; }

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

}''')


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from astropy.io import fits

    # Open file
    N = 128
    data = np.random.normal(size=N**2).reshape((N, N))+1

    # Calculate
    import os
    #fn = os.path.expanduser("~/dev/pybispec/dens.fits.gz")
    #fn = "/mnt/home/mobrien/ceph/driving/b1p.05_r512_f7/data/dn_b1p.05_512_f7_500.fits"
    data = fits.open(fn)[0].data.astype(np.float64)#.sum(axis=0)
    kmin, kmax = 0, 32
    bispec, bicoh, kn = bispectrum(data, nsamples=None, kmin=kmin, kmax=kmax,
                                   mean_subtract=False, bench=True, double=False)
    print(bispec.mean(), bicoh.mean())
    print(bicoh.max())

    # Plot
    cmap = 'plasma'
    labels = [r"$B(k_1, k_2)$", "$b(k_1, k_2)$"]
    data = [np.log10(np.abs(bispec)), bicoh]
    fig, axes = plt.subplots(ncols=2)
    for i in range(2):
        ax = axes[i]
        #ax.set_xscale("log")
        #ax.set_yscale("log")
        im = ax.imshow(data[i], origin="lower",
                       interpolation="nearest",
                       cmap=cmap,
                       extent=[kmin, kmax, kmin, kmax])
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label(labels[i])
        if i == 0:
            ax.contour(data[i], colors='k', extent=[kmin, kmax, kmin, kmax])
        ax.set_xlabel(r"$k_1$")
        ax.set_ylabel(r"$k_2$")

    plt.tight_layout()

    plt.show()
