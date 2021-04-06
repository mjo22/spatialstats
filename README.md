# spatialstats #
<tt>spatialstats</tt> is a collection of my statistical tools and utility routines used to analyze the multi-scale structure of 2D and 3D spatial fields and particle distributions.

Routines are designed to work with large datasets and some include optional CuPy acceleration. Each routine aims to be independent from the rest of the package, so feel free to use anything you like!

Current submodules are

### polyspectra ###
Calculate the bispectrum and power spectra of 2D and 3D grids.

### points ###
Calculate statistics about the multi-scale structure of 2D and 3D particle distributions, like the radial distribution function and structure factor.
