# RTK installation

## Configuration, compilation and installation with ITK
RTK is a module of [ITK](https://www.itk.org), the Insight Toolkit. Follow the instructions of the [ITK software guide](https://itk.org/ITKSoftwareGuide/html) ([chapter 2](https://itk.org/ITKSoftwareGuide/html/Book1/ITKSoftwareGuide-Book1ch2.html) mainly) for configuring and compiling ITK. The following CMake options are RTK specific:

* `Module_RTK`: Activates RTK download and compilation. Default is `OFF`. Turn it `ON` to activate RTK or compile RTK independently (see below).
* `Module_RTK_GIT_TAG`: Git tag for the RTK download. By default, the RTK version which is downloaded and compiled is the one given in the [RTK.remote.cmake](https://github.com/InsightSoftwareConsortium/ITK/blob/main/Modules/Remote/RTK.remote.cmake). Change this option to build another version. For example, you can change it to `main` to build the latest RTK version. RTK is only maintained to be backward compatible with the latest ITK release and ITK main branch.
* `RTK_BUILD_APPLICATIONS`: Activates the compilation of RTK's command line tools. Although RTK is mainly a toolkit, we also provide several command line tools for doing most of the available processing. These command line tools use [gengetopt](https://www.gnu.org/software/gengetopt/gengetopt.html). Several examples are available in the [documentation](http://docs.openrtk.org).
* `RTK_USE_CUDA`: Activates CUDA computation. Default is `ON` if CMake has automatically found the CUDA package and a CUDA-compatible GPU, and `OFF` otherwise.
* `RTK_CUDA_VERSION`: Specifies an exact version of the CUDA toolkit which must be used. If unspecified, RTK only checks if the found version is recent enough.
* `RTK_CUDA_PROJECTIONS_SLAB_SIZE`: Set the number of projections processed at once in CUDA processing. Default is 16.
* `RTK_PROBE_EACH_FILTER`: Activates the timing, CPU and CUDA memory consumption of each filter. Defaults is `OFF`. When activated, each filter processing is probed and a summary can be displayed. All command line applications display the result with `--verbose`.

RTK will automatically be installed when installing ITK.

## Independent configuration and compilation
For RTK developpers, it may be useful to compile RTK independently from ITK. This is possible, simply:
* Compile ITK with `Module_RTK=OFF`.
* If you want to use CUDA, also activate `Module_CudaCommon` or compile it separately as RTK in the following two bullet points (cloning its [GitHub repository](https://github.com/RTKConsortium/ITKCudaCommon) or downloading it as a [zip package](https://codeload.github.com/RTKConsortium/ITKCudaCommon/zip/main)).
* Manually download RTK's source repository from [GitHub](https://github.com/RTKConsortium/RTK) with `git` (recommended) or as a [zip package](https://codeload.github.com/RTKConsortium/RTK/zip/main).
* Configure the project with CMake pointing to RTK's source directory and setting the CMake option `ITK_DIR` to ITK's compilation directory. All CMake options above can be set except `Module_RTK`.

Installation is currently not supported for independent RTK compilations.

## Python pre-compiled binaries
We only provide pre-compiled binaries for the Python package which depends on ITK. Use the following commands to install the RTK module with `pip`.
```
python -m pip install --upgrade pip
python -m pip install itk-rtk
```
The same operating systems and Python versions are supported as ITK's packages, see the list on [Pypi](https://pypi.org/project/itk-rtk).

We also provide pre-compiled [CUDA](https://developer.nvidia.com/cuda-toolkit) packages for Windows and Linux. They require an installed version of CUDA compatible with the package. Currently, RTK is available for CUDA 12.4 on Windows and Linux via:
```
python -m pip install itk-rtk-cuda124
```
**Note**: the official PyPI packages are built from the [latest RTK release](https://github.com/RTKConsortium/RTK/releases/latest) and may not include features described in the latest documentation. If you install from PyPI, consult the corresponding [stable documentation](https://docs.openrtk.org/en/stable/index.html).

To try the newest features before they are published on PyPI, download the most recent wheels produced by our CI workflows:

- [Non-CUDA packages](https://github.com/RTKConsortium/RTK/actions/workflows/build-test-package.yml?query=branch%3Amain+is%3Asuccess)
- [CUDA packages](https://github.com/RTKConsortium/RTK/actions/workflows/build-test-package-python-cuda.yml?query=branch%3Amain+is%3Asuccess)

Each workflow shows a list of *Artifacts* at the bottom of its front page if you are logged in on GitHub. Just select the version corresponding to your operating system and Python version, unzip the wheel and `pip install wheel_filename`.

### GUI tools
Some RTK command-line tools, such as `rtkshowgeometry`, require graphical visualization capabilities. To use these tools, install the GUI optional dependencies:
```
python -m pip install itk-rtk[gui]
```

This installs:
* `pyvista` - for 3D visualization
* `matplotlib` - for additional graphics support

### Testing
To run RTK Python tests with pytest, install the test dependencies from the directory containing `pyproject.toml`:

```
pip install --group test
```

This installs:
* `pytest>=7.0` - for running Python unit tests
* `matplotlib` - for visualization in tests

After installing, run the tests with:
```
pytest test/
```

### Documentation
To build the RTK documentation, install the documentation dependencies from the directory containing `pyproject.toml`:

```
pip install --group doc
```

This installs Sphinx and related tools for documentation generation.

## Getting started
See [GettingStarted.md](GettingStarted.md). Your `CMakeLists.txt` can now use RTK when importing ITK as shown in the [FirstReconstruction's CMakeLists.txt](https://github.com/RTKConsortium/RTK/blob/main/examples/FirstReconstruction/CMakeLists.txt#L7).
