

import lazy_import
import warnings
from .Configuration import Configuration

#
# Lazy load subpackages
#


spectral = lazy_import.lazy_module("softstats.spectral")
scatter = lazy_import.lazy_module("softstats.scatter")
util = lazy_import.lazy_module("softstats.util")

#
# Set Configuration object
#


def warn(action):
    warnings.simplefilter(action)
    return action


def gpu(id):
    """
    Configure CuPy GPU usage
    """
    if id is not False:
        try:
            import cupy
            cupy.zeros(0)
            if type(id) is int:
                cupy.cuda.Device(id).use()
        except (ImportError, ModuleNotFoundError) as err:
            warnings.warn(f"{str(err)}. Falling back to CPU usage.",
                          ImportWarning)
            id = False
        except Exception as err:
            warnings.warn(f"{str(err)}. Falling back to CPU usage.",
                          RuntimeWarning)
            id = False
    return id


config = Configuration({warn: "ignore", gpu: False})


#
# Clean namespace
#
del gpu, warn, Configuration, lazy_import
