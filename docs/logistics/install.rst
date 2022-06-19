
Installing spatialstats
=======================

You can clone the latest version of spatialstats from
`github <https://github.com/mjo22/spatialstats>`_ and build using::

      >>> python setup.py install

Installation from PyPI has been discontinued.

spatialstats does not load any of its routines until the time of import (lazy loading), so the only installation requirement is `numpy <https://github.com/numpy/numpy>`_  and `scipy <https://github.com/scipy/scipy>`_. This is to keep the flexibility and extensibility of spatialstats as a package of disconnected routines. Users may need to add additional dependencies after installation.
