"""GBDT package: https://github.com/yarny/gbdt
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path
import sys

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='gbdt',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version="0.21-beta",

    description='GBDT',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/yarny/gbdt',

    # Author details
    author='Jiang Chen',
    author_email='criver@gmail.com',

    # Choose your license
    license='Apache 2.0',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: Machine Learning Practitioners',
        'Topic :: Machine Learning',

        # Pick your license as you wish (should match "license" above)
        'License :: Apache 2.0',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7'
    ],

    # What does your project relate to?
    keywords='gbdt,machine learning,decision tree',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    package_dir = {'': 'python'},
    packages=['gbdt'],
    package_data = {'gbdt': ['lib/darwin_x86_64/libgbdt.so', 'lib/linux_x86_64/libgbdt.so'] }
)
