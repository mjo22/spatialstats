

import softstats as ss


if ss.config.gpu is False:
    from .structure_factor import structure_factor
else:
    from .cuda_structure_factor import structure_factor
