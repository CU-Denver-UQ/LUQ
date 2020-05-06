# LUQ: Learning Uncertain Quantities
LUQ is a Python package that provides simple implementations of the algorithms for learning uncertain quantities.
LUQ utilizes several publicly available Python packages that are commonly used for scientific computing ([NumPy](https://numpy.org/) and [SciPy](https://www.scipy.org/)) and machine learning ([scikit-learn](https://scikit-learn.org/).
The package provides a simple end-to-end workflow going from raw time series data to QoI which can be used for data-consistent inversion.

## Installation
The current development branch of LUQ can be installed from GitHub,  using ``pip``:

    pip install git+https://github.com/CU-Denver-UQ/LUQ
    
Another option is to clone the repository and install LUQ using
``python setup.py install``

## Dependencies
LUQ is tested on Python 3.6 and depends on scikit-learn, NumPy, SciPy, and matplotlib (see ``requirements.txt`` for version information).
