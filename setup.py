
# Generated using: python setup_py_configure.py 'itk'

from __future__ import print_function
from os import sys, path

try:
    from skbuild import setup
except ImportError:
    print('scikit-build is required to build from source.', file=sys.stderr)
    print('Please run:', file=sys.stderr)
    print('', file=sys.stderr)
    print('  python -m pip install scikit-build')
    sys.exit(1)

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from itkVersion import get_versions

setup(
    name='rtk',
    version=get_versions()['package-version'],
    author='RTK consortium',
    author_email='simon.rit@creatis.insa-lyon.fr',
    packages=['rtk'],
    package_dir={'itk': 'itk'},
    cmake_args=[],
    py_modules=[
        'itkRTK'
    ],
    download_url=r'https://itk.org/ITK/resources/software.html',
    description=r'ITK is an open-source toolkit for multidimensional image analysis',
    long_description='ITK is an open-source, cross-platform library that '
                     'provides developers with an extensive suite of software '
                     'tools for image analysis. Developed through extreme '
                     'programming methodologies, ITK employs leading-edge '
                     'algorithms for registering and segmenting '
                     'multidimensional scientific images.',
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
    keywords='ITK InsightToolkit segmentation registration image imaging',
    url=r'http://openrtk.org/',
    install_requires=[
    ]
    )
