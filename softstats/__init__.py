

from .Configuration import Configuration


def logging(level):
    """
    Set level of logger
    """
    import logging
    level = level.upper()
    logger = logging.getLogger("softstats")
    logging.basicConfig()
    logger.setLevel(getattr(logging, level))
    return level


def gpu(id):
    """
    Configure CuPy GPU usage
    """
    import logging
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


config = Configuration({logging: "WARNING", gpu: True})

del logging, gpu
