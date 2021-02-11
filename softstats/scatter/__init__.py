
import softstats

if softstats.config.gpu is False:
    from ._structure_factor import structure_factor
else:
    from ._cuda_structure_factor import structure_factor

del softstats
