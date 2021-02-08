
from .Configuration import Configuration, get_config
import logging


def _logging(level):
    """
    Set level of logger
    """
    level = level.upper()
    logger = logging.getLogger("softstats")
    logging.basicConfig()
    logger.setLevel(getattr(logging, level))
    return level


def _gpu(id):
    """
    Configure CuPy GPU usage
    """
    if id is not False:
        try:
            import cupy
            cupy.zeros(0)
            if type(id) is int:
                cupy.cuda.Device(id).use()
        except Exception as err:
            logging.warning(str(err))
            id = False
    return id


CONFIG_FILE = get_config(__file__)
__config__ = Configuration(CONFIG_FILE, setters=[_logging, _gpu])
