
Installing spatialstats
=======================

You can install the latest version of spatialstats from pip::

      >>> pip install spatialstats

or you can clone the `github repository <https://github.com/mjo22/spatialstats>`_

spatialstats does not load any of its routines until the time of import (lazy loading), so the only installation requirement is `numpy <https://github.com/numpy/numpy>`_. This is to keep the flexibility and extensibility of spatialstats as a package of disconnected routines. Users may need to add additional dependencies after installation, such as

 * `scipy <https://github.com/scipy/scipy>`_
 * `numba <https://github.com/numba/numba>`_>=0.50
 * `pyfftw <https://github.com/pyFFTW/pyFFTW>`_
 * `cupy <https://github.com/cupy/cupy>`_)>=8.0
