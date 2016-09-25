"""GBDT package: https://github.com/yarny/gbdt
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path
import os
import sys

here = path.abspath(path.dirname(__file__))
sys.path.append(os.path.join(here, "python/gbdt"))
import _version

setup(
    name='gbdt',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=_version.__version__,

    description='High performance implementation of GBDT family of algorithm',
    long_description="""GBDT is a high performance and full featured C++ implementation of [Jerome H. Friedman's Gradient Boosting Decision Trees Algorithm](http://statweb.stanford.edu/~jhf/ftp/stobst.pdf) and its modern offsprings,. It features high efficiency, low memory footprint, collections of loss functions and built-in mechanisms to handle categorical features and missing values.

When is GBDT good for you?
-----------
* **You are looking beyond linear models.**
  * Gradient Boosting Decision Trees Algorithms is one of the best offshelf ML algorithms with built-in capabilities of non-linear transformation and feature crossing.
* **Your data is too big to load into memory with existing ML packages.**
  * GBDT reduces memory footprint dramatically with feature bucketization. For some tested datasets, it used 1/7 of the memory of its counterpart and took only 1/2 time to train. See [docs/PERFORMANCE_BENCHMARK.md](https://github.com/yarny/gbdt/blob/master/docs/PERFORMANCE_BENCHMARK.md) for more details.
* **You want better handling of categorical features and missing values.**
  * GBDT has built-in mechanisms to figure out how to split categorical features and place missing values in the trees.
* **You want to try different loss functions.**
  * GBDT implements various pointwise, pairwise, listingwis loss functions including mse, logloss, huberized hinge loss, pairwise logloss,
[GBRank](http://www.cc.gatech.edu/~zha/papers/fp086-zheng.pdf) and [LambdaMart](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf). It supports easily addition of your own custom loss functions.
    """,

    # The project's main homepage.
    url='https://github.com/yarny/gbdt',

    # Author details.
    author='Jiang Chen',
    author_email='criver@gmail.com',

    # Choose your license
    license='Apache 2.0',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: Apache Software License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: C++',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7'
    ],

    # What does your project relate to?
    keywords='gbdt,machine learning,decision trees,forest',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    package_dir = {'': 'python'},
    packages=['gbdt'],
    package_data = {'gbdt': ['lib/darwin_x86_64/libgbdt.so', 'lib/linux_x86_64/libgbdt.so'] }
)
