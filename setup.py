# -*- coding: utf-8 -*-
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
    version='2.3.0',
    author='RTK Consortium',
    author_email='rtk-users@openrtk.org',
    packages=['itk'],
    package_dir={'itk': 'itk'},
    download_url=r'https://github.com/SimonRit/RTK',
    description=r'The Reconstruction Toolkit (RTK) for fast circular cone-beam CT reconstruction.',
    long_description='Based on the Insight Toolkit ITK, RTK provides: basic operators for reconstruction (e.g. filtering, forward, projection and backprojection), multithreaded CPU and GPU versions, tools for respiratory motion correction, I/O for several scanners, preprocessing of raw data for scatter correction.',
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
    scripts=[
        "lib/rtkorageometry.py",
        "lib/rtksimulatedgeometry.py",
        "lib/rtkvarianobigeometry.py"
        ],
    license='Apache',
    keywords='RTK Reconstruction Toolkit',
    url=r'https://www.openrtk.org/',
    install_requires=[
        r'itk>=5.3rc02.post1'
    ]
    )
