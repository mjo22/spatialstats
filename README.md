# spatialstats #
<tt>spatialstats</tt> is a collection of statistical tools and utility routines used to analyze the multi-scale structure of 2D and 3D spatial fields and particle distributions.

Routines are designed to work with large datasets and some include optional CuPy acceleration. Each routine aims to be independent from the rest of the package, so feel free to just pull out the routine that you need!

## Submodules ##

### polyspectra ###
Calculate the bispectrum and power spectrum of 2D and 3D grids.

### points ###
Calculate statistics about the multi-scale structure of 2D and 3D point distributions, like the radial distribution function and structure factor.

## GPU usage ##

The following example demonstrates how to access the <tt>spatialstats</tt> configuration object to toggle gpu usage

```python
import numpy as np
import spatialstats as ss

ss.config.gpu = True

shape = (100, 100)
data = np.random.rand(*shape)
result = ss.polyspectra.bispectrum(data)
```

## Installation ##

### Option 1 ###

Clone from github and build by running

```shell
python setup.py install
```

### Option 2 ###

Install from PyPI

```shell
pip install spatialstats
```

#### Additional Dependencies ####

<tt>spatialstats</tt> does not load any of its routines until the time of import (lazy loading), so installation requirements are minimal. This is to keep the flexibility and extensibility of <tt>spatialstats</tt> as a package of disconnected routines. Users may need to add additional dependencies after installation, such as [scipy](https://github.com/scipy/scipy), [numba](https://github.com/numba/numba)>=0.50, [pyfftw](https://github.com/pyFFTW/pyFFTW), [cupy](https://github.com/cupy/cupy)>=8.0.
