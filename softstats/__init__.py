
from softstats.utilities.Configuration import Configuration, get_config


def _gpu(id):
    """
    Configure CuPy GPU usage
    """
    import warnings
    if id is not False:
        try:
            import cupy
            cupy.zeros(0)
            if type(id) is int:
                cupy.cuda.Device(id).use()
        except (ImportError, ModuleNotFoundError) as err:
            warnings.warn(str(err), category=ImportWarning)
            id = False
        except Exception as err:
            warnings.warn(str(err), category=RuntimeWarning)
            id = False
    return id


CONFIG_FILE = get_config(__file__)
__config__ = Configuration(CONFIG_FILE, setters=[_gpu])
