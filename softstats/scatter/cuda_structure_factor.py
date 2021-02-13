"""
Calculating the structure factor for a collection
of identical particle positions with CUDA acceleration

Requires cupy>=8.0

Author:
    Michael O'Brien (2020)
    Biophysical Modeling Group
    Center for Computational Biology
    Flatiron Institute
"""

import cupy as cp
import numpy as np
import time
from astropy.utils.console import ProgressBar
from cupy.cuda.memory import OutOfMemoryError


def structure_factor(positions, a=1., qmax=5, average=True, corners=False,
                     nsamples=None, rchunks=None, qchunks=None,
                     bench=True, progress=False):
    """
    Compute the structure factor for a set
    of positions of identical particles.

    Arguments
    ---------
    positions : np.ndarray [ndim, N]
        The particle positions. This should be
        shaped as an [ndim, N] array, where
        ndim is the number of dimensions and
        N is the number of particles.

    Keywords
    --------
    qmax: float
        The maximum wavenumber to consider.
    a : float or list of floats
        The length scale of each dimension.
    average : bool
        If True, return radially averaged
        structure factor as a function of wavenumber.
        If False, return structure factor as a function
        of wavevector.
    qchunks : int
        Number of wavevectors to calculate at a time in loop.
        Defaults to the total number of nonzero wavevectors.
    rchunks : int
        Number of particles to calculate at a time in
        inner loop. Defaults to total number of particles.
    bench : bool
        Output time of calculation.
    progress : bool
        Display progress bar of calculation.

    Returns
    -------
    Sq : np.ndarray
        The structure factor as a function of wavenumber or wavevector
    q : np.ndarray
        Wavenumber or list of components for wavevector
    """
    int, float = cp.int32, cp.float32
    ndim = positions.shape[0]

    if ndim not in [2, 3]:
        msg = "Data must be 2D or 3D. "
        msg += "Positions array should be shaped (ndim, N)."
        raise ValueError(msg)

    kernel = kernel3D if ndim == 3 else kernel2D

    freqs = []
    for i in range(ndim):
        if i < ndim-1:
            freq = cp.fft.fftfreq
        else:
            freq = cp.fft.rfftfreq
        freqs.append(2*qmax*freq(2*qmax).astype(float))
    grid = cp.meshgrid(*freqs, indexing="ij")

    # Number of particles and number of wavevectors
    N = positions.shape[1]
    M = grid[0].size
    shape = grid[0].shape

    # Wavevectors
    a = ndim*[a] if type(a) is not list else a
    q = cp.zeros((ndim, *shape), dtype=float)
    for i in range(ndim):
        q[i] = (1. / a[i]) * grid[i]

    del grid

    # Sample subset of particles
    if nsamples is not None:
        idxs = np.indices((N,))[0]
        samples = np.random.choice(idxs, size=nsamples, replace=True)
        positions = positions[:, samples]
        N = positions.shape[1]

    # Position vectors
    R = cp.asarray(positions, dtype=float)

    # Params for splitting calculation into chunks
    qchunks = M if qchunks is None else qchunks
    rchunks = N if rchunks is None else rchunks

    if M % qchunks != 0:
        raise ValueError(f"qchunks must divide # nonzero wavevectors {M}")
    if N % rchunks != 0:
        raise ValueError(f"rchunks must divide # particles {N}")

    nr = N // rchunks
    mq = M // qchunks

    if bench:
        t0 = time.time()
    if progress:
        bar = ProgressBar(nr*mq)

    threadsperblock = 32
    blockspergrid = (qchunks*N*rchunks +
                     (threadsperblock - 1)) // threadsperblock

    # Allocate buffers
    try:
        jv, kv = (cp.empty((N, rchunks), dtype=int),
                  cp.empty((N, rchunks), dtype=int))
        buf = cp.zeros((qchunks, N*rchunks), dtype=float)
    except OutOfMemoryError as err:
        msg = "Out of memory allocating buffers. "
        msg += "Decrease rchunks and/or qchunks."
        raise ValueError(msg) from err
    j = cp.arange(N, dtype=int)
    k = cp.zeros(rchunks, dtype=int)
    qm = cp.zeros((ndim, qchunks), dtype=float)
    qflat = q.reshape((ndim, M))
    out = cp.zeros(M, dtype=float)
    # Set arguments for kernel
    if ndim == 3:
        rx, ry, rz = R[0], R[1], R[2]
        qx, qy, qz = qm[0], qm[1], qm[2]
        args = (jv, kv, rx, ry, rz, qx, qy, qz,
                cp.int64(qchunks), cp.int64(N*rchunks), buf)
    else:
        rx, ry = R[0], R[1]
        qx, qy = qm[0], qm[1]
        args = (jv, kv, rx, ry, qx, qy,
                cp.int64(qchunks), cp.int64(N*rchunks), buf)
    # Calculate
    for n in range(nr):
        rstart, rstop = n*rchunks, (n+1)*rchunks
        k[:] = cp.arange(rstart, rstop, dtype=int)
        jv[...], kv[...] = cp.meshgrid(j, k, indexing="ij")
        for m in range(mq):
            qstart, qstop = m*qchunks, (m+1)*qchunks
            qm[:] = qflat[:, qstart:qstop]
            kernel((blockspergrid,), (threadsperblock,), args)
            out[qstart:qstop] += buf.sum(axis=1)
            if progress:
                bar.update(n*mq+m+1)

    out = out/N
    Sq = out.reshape(shape)

    # Isotropically average
    if average:
        qr = cp.zeros_like(q[0])
        for i in range(ndim):
            qr += (q[i]**2)
        qr = cp.sqrt(qr)
        qrmax = qr.max().get() if corners else qmax
        qn = cp.unique(qr)
        qn = qn[cp.where(qn <= qrmax)]
        spectrum = cp.zeros(qn.size, dtype=float)
        for i, qi in enumerate(qn):
            ii = cp.where(qr == qi)
            spectrum[i] = cp.mean(Sq[ii])
        Sq = spectrum
        q = qn
    else:
        Sq[...] = cp.fft.fftshift(Sq, axes=np.arange(ndim-1))
        q[...] = cp.fft.fftshift(q, axes=1+np.arange(ndim-1))

    Sq = cp.asnumpy(Sq)
    q = cp.asnumpy(q)

    if bench:
        print(f"\nTime: {time.time() - t0:.04f}")

    return Sq, q


kernel3D = cp.RawKernel('''

const float PI = 3.14159265358979323846;

extern "C" __global__
void struct3D(const int* jv, const int* kv,
              const float* rx, const float* ry, const float* rz,
              const float* qx, const float* qy, const float* qz,
              long M, long N, float* out) {

    long idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx > N*M - 1) { return; }

    __syncthreads();

    int j = jv[idx % N];
    int k = kv[idx % N];
    long m = idx / N;

    float qr = qx[m]*(rx[j]-rx[k])+qy[m]*(ry[j]-ry[k])+qz[m]*(rz[j]-rz[k]);
    out[idx] = cosf(2*PI*qr);
}
''', 'struct3D')


kernel2D = cp.RawKernel('''

const float PI = 3.14159265358979323846;

extern "C" __global__
void struct2D(const int* jv, const int* kv,
              const float* rx, const float* ry,
              const float* qx, const float* qy,
              long M, long N, float* out) {

    long idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx > N*M - 1) { return; }

    __syncthreads();

    int j = jv[idx % N];
    int k = kv[idx % N];
    long m = idx / N;

    float qr = qx[m]*(rx[j]-rx[k])+qy[m]*(ry[j]-ry[k]);
    out[idx] = cosf(2*PI*qr);
}
''', 'struct2D')


if __name__ == "__main__":

    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import pyplot as plt

    # Sample problem of Face-centered cubic crystal
    sample = "FCC"
    if sample == "FCC":
        average = False
        ndim = 3
        a = 10
        N = 4
        r = np.zeros((ndim, N))
        r[:, 0] = 0
        r[:, 1] = np.array((a/2, a/2, 0))
        r[:, 2] = np.array((0, a/2, a/2))
        r[:, 3] = np.array((a/2, 0, a/2))
        qmax = 3
    elif sample == "BCC":
        average = False
        ndim = 3
        a = 10
        N = 2
        r = np.zeros((ndim, N))
        r[:, 0] = 0
        r[:, 1] = a/2
        qmax = 3
    elif sample == "uniform":
        average = True
        a = 1
        N = 100
        ndim = 2
        r = np.random.uniform(low=-a/2, high=a/2, size=ndim*N)
        r = r.reshape((ndim, N))
        qmax = 10
    else:
        exit()

    Sq, q = structure_factor(r, qmax=qmax, a=a,
                             average=average,  # rchunks=200,
                             progress=True, corners=False)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*[r[i] for i in range(ndim)])
    plt.show()

    if not average:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        q = (q*a).astype(int)
        qx, qy, qz = q
        scat = ax.scatter(qx, qy, qz, c=Sq)
        fig.colorbar(scat)
        plt.show()
        if sample == "BCC":
            F_hkl = 2 * (~np.sum(q, axis=0) % 2)
        elif sample == "FCC":
            F_hkl = 4*(np.logical_or(np.all(q % 2 == 0, axis=0),
                                     np.all(q % 2 == 1, axis=0)))
        else:
            exit()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        qx, qy, qz = q
        scat = ax.scatter(qx, qy, qz, c=Sq - F_hkl)
        print(f"Residuals: {(Sq - F_hkl).sum()}")
        fig.colorbar(scat)
        plt.show()
    else:
        plt.plot(q[1:], Sq[1:], color="r")
        #plt.ylim(0, 3)
        plt.show()
