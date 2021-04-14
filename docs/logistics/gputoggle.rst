
Enabling GPU acceleration
=========================

The spatialstats configuration object allows one to set package configuration after the time of import. Here is an example of toggling GPU usage:

  >>> import numpy as np
  >>> import spatialstats as ss
  >>> shape = (100, 100)
  >>> data = np.random.rand(*shape)
  >>> ss.config.gpu = True
  >>> result = ss.polyspectra.bispectrum(data)

One cannot subsequently set:

  >>> ss.config.gpu = False

and call the bispectrum again as this will not trigger another import.
