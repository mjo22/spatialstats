

import warnings
from .Configuration import Configuration


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


config = Configuration({warn: "default", gpu: False})

del gpu, warn, Configuration
