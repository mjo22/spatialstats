
import softstats

if softstats.config.gpu is False:
    from ._powerspectrum import powerspectrum
    from ._bispectrum import bispectrum
else:
    from ._cuda_powerspectrum import powerspectrum
    from ._cuda_bispectrum import bispectrum

del softstats
