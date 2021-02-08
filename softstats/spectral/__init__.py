
import softstats

if softstats.__config__.gpu is False:
    from softstats.spectral.powerspectrum import powerspectrum
    from softstats.spectral.bispectrum import bispectrum
    from softstats.spectral.structure_factor import structure_factor
else:
    from softstats.spectral.cuda_powerspectrtum import powerspectrum
    from softstats.spectral.cuda_bispectrum import bispectrum
    from softstats.spectral.cuda_structure_factor import structure_factor
