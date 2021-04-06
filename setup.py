
import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='spatialstats',
    version="1.0.0",
    license="MIT",
    author="Michael O'Brien",
    author_email="mobrien-temp@flatironinstitute.org",
    keywords="soft-condensed biophysics bispectrum cupy",
    url="https://github.com/mjo22/spatialstats",
    description="spatialstats is collection of statistical tools used to analyze the multi-scale structure of spatial fields and particle distributions.",
    long_description=read("README.md"),
    packages=["spatialstats"],
    install_requires=["numpy", "scipy", "numba"],
    classifiers=["License :: OSI Approved :: MIT License",
                 "Programming Language :: Python :: 3",
                 "Programming Language :: Python :: 3.7",
                 "Programming Language :: Python :: 3.8"
                 ],
    zip_safe=False
)
