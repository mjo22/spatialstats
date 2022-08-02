# spatialstats #
<tt>spatialstats</tt> is a collection of correlation function routines used to analyze the multi-scale structure of 2D and 3D spatial fields and particle distributions. As of August 2022, this package will no longer be maintained.

Each routine aims to be independent from the rest of the package, so feel free to just pull out the routine that you need!

You can read the docs at https://spatialstats.readthedocs.io/.

If you have a routine that you think would fit in this package, please do reach out! I currently have no plans to implement specific routines--only ones that come up in my research.

### paircount ###
Calculate two-point correlation multipoles using pair counting.

### polyspectra ###
Compute power spectrum multipoles of scalar, vector, and tensor
fields and particle data, and compute bispectra on 2D and 3D grids.

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

Clone from github and build by running

```shell
python setup.py install
```

Installation from PyPI has been discontinued.

#### Additional Dependencies ####

<tt>spatialstats</tt> does not load any of its routines until the time of import (lazy loading), so the only installation requirements are [numpy](https://github.com/numpy/numpy) and [scipy](https://github.com/scipy/scipy).
This is to keep the flexibility of <tt>spatialstats</tt> as a package of disconnected routines. Users may need to add additional dependencies after installation, such as [numba](https://github.com/numba/numba)>=0.50, [cupy](https://github.com/cupy/cupy)>=8.0, and [pyfftw](https://github.com/pyFFTW/pyFFTW), [finufft](https://github.com/flatironinstitute/finufft), and
[sympy](https://github.com/sympy/sympy).
