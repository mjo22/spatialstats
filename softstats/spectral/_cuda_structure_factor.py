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

    # Generate grid of wavevectors
    npts = 2*qmax+1
    qn = cp.linspace(-qmax, qmax, npts, dtype=float, endpoint=True)
    grid = [r[..., npts//2:] for r in cp.meshgrid(*(ndim*[qn]), indexing="ij")]

    # Number of particles and number of wavevectors
    N = positions.shape[1]
    M = grid[0].size

    # Sample subset of particles
    if nsamples is not None:
        idxs = np.indices((N,))[0]
        samples = np.random.choice(idxs, size=nsamples, replace=True)
        positions = positions[:, samples]
        N = positions.shape[1]

    # Position vectors
    R = cp.asarray(positions, dtype=float)

    # Wavevectors
    a = ndim*[a] if type(a) is not list else a
    Q = cp.zeros((3, M), dtype=float)
    for i in range(ndim):
        Q[i] = (2*cp.pi / a[i]) * grid[i].ravel()

    # Params for splitting calculation into chunks
    qchunks = M-1 if qchunks is None else qchunks
    rchunks = N if rchunks is None else rchunks

    if (M-1) % qchunks != 0:
        raise ValueError(f"qchunks must divide # nonzero wavevectors {M-1}")
    if N % rchunks != 0:
        raise ValueError(f"rchunks must divide # particles {N}")

    nr = N // rchunks
    mq = (M-1) // qchunks

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
    k = cp.empty(rchunks, dtype=int)
    q = cp.zeros((ndim, qchunks), dtype=float)
    out = cp.zeros(M, dtype=float)
    # Set arguments for kernel
    if ndim == 3:
        rx, ry, rz = R[0], R[1], R[2]
        qx, qy, qz = q[0], q[1], q[2]
        args = (jv, kv, rx, ry, rz, qx, qy, qz,
                cp.int64(qchunks), cp.int64(N*rchunks), buf)
    else:
        rx, ry = R[0], R[1]
        qx, qy = q[0], q[1]
        args = (jv, kv, rx, ry, qx, qy,
                cp.int64(qchunks), cp.int64(N*rchunks), buf)
    # Calculate
    for n in range(nr):
        rstart, rstop = n*rchunks, (n+1)*rchunks
        k[:] = cp.arange(rstart, rstop, dtype=int)
        jv[...], kv[...] = cp.meshgrid(j, k, indexing="ij")
        for m in range(mq):
            qstart, qstop = m*qchunks, (m+1)*qchunks
            q[:] = Q[:, qstart:qstop]
            kernel((blockspergrid,), (threadsperblock,), args)
            out[qstart+1:qstop+1] += buf.sum(axis=1)
            if progress:
                bar.update(n*mq+m+1)

    out[0] = 2 * N**2
    Sq = out.reshape(grid[0].shape)/N

    # Isotropically average
    if average:
        qr = cp.zeros_like(grid[0])
        for i in range(ndim):
            qr += grid[i]**2
        qr = cp.sqrt(qr)
        qrmax = int(qr.max().get()) if corners else qmax+1
        qm = cp.arange(1, qrmax)
        spectrum = cp.zeros(qm.size, dtype=float)
        for i, qi in enumerate(qm):
            ii = cp.where(cp.logical_and(qr >= qi, qr < qi+1))
            spectrum[i] = cp.mean(Sq[ii])
        Sq = spectrum
        q = cp.asnumpy(qm)
    else:
        q = [cp.asnumpy(qi) for qi in grid]

    Sq = cp.asnumpy(Sq)

    if bench:
        print(f"\nTime: {time.time() - t0:.04f}")

    return Sq, q


kernel3D = cp.RawKernel('''
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
    out[idx] = 2*cosf(qr);
}
''', 'struct3D')


kernel2D = cp.RawKernel('''
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
    out[idx] = 2*cosf(qr);
}
''', 'struct2D')


if __name__ == "__main__":

    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import pyplot as plt

    # Sample problem of Face-centered cubic crystal
    average = False
    ndim = 3
    a = 1
    N = 4
    r = np.zeros((ndim, N))
    r[:, 0] = 0
    r[:, 1] = np.array((a/2, a/2, 0))
    r[:, 2] = np.array((0, a/2, a/2))
    r[:, 3] = np.array((a/2, 0, a/2))
    qmax = 8

    Sq, q = structfactor(r, qmax=qmax, a=a, average=average,
                         rchunks=4, progress=True)

    print(np.unique(Sq))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(r[0], r[1], r[2])
    plt.show()

    if not average:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        qx, qy, qz = q
        scat = ax.scatter(qx, qy, qz, c=Sq)
        ax.set_xlim((-qmax, qmax))
        ax.set_ylim((-qmax, qmax))
        ax.set_zlim((0, qmax))
        fig.colorbar(scat)
        plt.show()
    else:
        plt.plot(q, Sq, color="r")
        plt.show()
