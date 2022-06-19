"""
Calculating correlation functions with FFT estimators.

.. moduleauthor:: Michael O'Brien <michaelobrien@g.harvard.edu>
"""

import spatialstats

from .fftpower import fftpower, nufftpower

if spatialstats.config.gpu is False:
    from .powerspectrum import powerspectrum
    from .bispectrum import bispectrum
else:
    from .cuda_bispectrum import bispectrum
    from .cuda_powerspectrum import powerspectrum

del spatialstats
