
import importlib
import warnings
from .Configuration import Configuration

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
# Lazy load subpackages
#


def __getattr__(name):
    if name in ['spectral', 'scatter', 'utils']:
        return importlib.import_module("."+name, __name__)
    elif name == 'config':
        return config
    else:
        raise AttributeError(
            f"{__name__!r} has no attribute {name!r}")


#
# Clean namespace
#
del gpu, warn, Configuration
