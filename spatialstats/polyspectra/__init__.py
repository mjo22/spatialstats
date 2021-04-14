"""
Implementations for calculating spectral statistics of
vector and scalar fields in two and three dimensions.

.. moduleauthor:: Michael O'Brien <michaelobrien@g.harvard.edu>
"""

import spatialstats

if spatialstats.config.gpu is False:
    from .powerspectrum import powerspectrum
    from .bispectrum import bispectrum
else:
    from .cuda_powerspectrum import powerspectrum
    from .cuda_bispectrum import bispectrum

del spatialstats
