
import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='softstats',
    version="0.0.0",
    author="Michael O'Brien",
    author_email="mobrien-temp@flatironinstitute.org",
    keywords="soft-condensed biophysics bispectrum cupy",
    description=(
        "softstats is collection of statistical tools used to analyze multi-scale soft and biological matter systems."),
    long_description=read("README.md"),
    packages=find_packages(),
    zip_safe=False
)
