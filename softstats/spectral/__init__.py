
import softstats

if softstats.config.gpu is False:
    from ._powerspectrum import powerspectrum
    from ._bispectrum import bispectrum
    from ._structure_factor import structure_factor
else:
    from ._cuda_powerspectrum import powerspectrum
    from ._cuda_bispectrum import bispectrum
    from ._cuda_structure_factor import structure_factor

del softstats
