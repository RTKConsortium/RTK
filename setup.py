# -*- coding: utf-8 -*-
from __future__ import print_function
from os import sys

try:
    from skbuild import setup
except ImportError:
    print('scikit-build is required to build from source.', file=sys.stderr)
    print('Please run:', file=sys.stderr)
    print('', file=sys.stderr)
    print('  python -m pip install scikit-build')
    sys.exit(1)

setup(
    name='itk-rtk',
    version='0.0.1',
    author='RTK Consortium',
    author_email='rtk-users@openrtk.org',
    packages=['itk'],
    package_dir={'itk': 'itk'},
    download_url=r'https://github.com/SimonRit/RTK.git',
    description=r'The Reconstruction Toolkit (RTK) is an open-source and cross-platform software for fast circular cone-beam CT reconstruction.',
    long_description='Based on the Insight Toolkit ITK, RTK provides: Basic operators for reconstruction (e.g. filtering, forward, projection and backprojection), Multithreaded CPU and GPU versions, Tools for respiratory motion correction, I/O for several scanners, Preprocessing of raw data for scatter correction.',
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: C++",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries",
        "Operating System :: Android",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS"
        ],
    license='Apache',
    keywords='RTK Reconstruction Toolkit',
    url=r'https://www.openrtk.org/',
    install_requires=[
        r'itk'
    ]
    )
