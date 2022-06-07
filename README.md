# spatialstats #
<tt>spatialstats</tt> is a collection of statistical tools and utility routines used to analyze the multi-scale structure of 2D and 3D spatial fields and particle distributions.

Routines are designed to work with large datasets and some include optional CuPy acceleration. Each routine aims to be independent from the rest of the package, so feel free to just pull out the routine that you need!

You can read the docs at https://spatialstats.readthedocs.io/.

If you have a routine that you think would fit in this package, please do reach out! I currently have no plans to implement specific routines--only ones that come up in my research.

### polyspectra ###
Calculate the bispectrum and power spectrum of 2D and 3D grids.

### particles ###
Calculate statistics about the multi-scale structure of 2D and 3D particle distributions, like the spatial distribution function and structure factor.

## GPU usage ##

The following example demonstrates how to interact with the <tt>spatialstats</tt> configuration object to toggle gpu usage

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

This is the recommended method of installation.

### Option 2 ###

Install from PyPI

```shell
pip install spatialstats
```

#### Additional Dependencies ####

<tt>spatialstats</tt> does not load any of its routines until the time of import (lazy loading), so the only installation requirement is [numpy](https://github.com/numpy/numpy). This is to keep the flexibility of <tt>spatialstats</tt> as a package of disconnected routines. Users may need to add additional dependencies after installation, such as [scipy](https://github.com/scipy/scipy), [numba](https://github.com/numba/numba)>=0.50, [cupy](https://github.com/cupy/cupy)>=8.0, and [pyfftw](https://github.com/pyFFTW/pyFFTW).
