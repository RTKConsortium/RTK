RTK installation
================

Configuration, compilation and installation with ITK
----------------------------------------------------
RTK is a module of [ITK](http://www.itk.org), the Insight Toolkit. Follow the instructions of the [ITK software guide](https://itk.org/ITKSoftwareGuide/html) ([chapter 2](https://itk.org/ITKSoftwareGuide/html/Book1/ITKSoftwareGuide-Book1ch2.html) mainly) for configuring and compiling ITK. The following CMake options are RTK specific:

* `Module_RTK`: Activates RTK download and compilation. Default is `OFF`. Turn it `ON` to activate RTK or compile RTK independently (see below).
* `REMOTE_GIT_TAG_RTK`: Git tag for the RTK download. By default, the RTK version which is downloaded and compiled is the one given in the [RTK.remote.cmake](https://github.com/InsightSoftwareConsortium/ITK/blob/master/Modules/Remote/RTK.remote.cmake). Change this option to build another version. For example, you can change it to `master` to build the latest RTK version.
* `RTK_BUILD_APPLICATIONS`: Activates the compilation of RTK's command line tools. Although RTK is mainly a toolkit, we also provide several command line tools for doing most of the available processing. These command line tools use [gengetopt](https://www.gnu.org/software/gengetopt/gengetopt.html). Several examples are available on the [Applications](http://wiki.openrtk.org/index.php/RTK_wiki_help#Applications) section of the [wiki](http://wikiopenrtk.org).
* `RTK_USE_CUDA`: Activates CUDA computation. Default is `ON` if CMake has automatically found the CUDA package and a CUDA-compatible GPU, and `OFF` otherwise.
* `RTK_CUDA_PROJECTIONS_SLAB_SIZE`: Set the number of projections processed at once in CUDA processing. Default is 16.
* `RTK_PROBE_EACH_FILTER`: Activates the timing, CPU and CUDA memory consumption of each filter. Defaults is `OFF`. When activated, each filter processing is probed and a summary can be displayed. All command line applications display the result with `--verbose`.
* `REMOTE_GIT_TAG_RTK`: Selects RTK version. RTK is only maintained to be backward compatible with the latest ITK release and ITK master branch.

RTK will automatically be installed when installing ITK.

Independent configuration and compilation
-----------------------------------------
For RTK developpers, it may be useful to compile RTK independently from ITK. This is possible, simply:
* Compile ITK with `Module_RTK=OFF`.
* Manually download RTK's source repository from [GitHub](https://github.com/SimonRit/RTK) with `git` (recommended) or as a [zip package](https://codeload.github.com/SimonRit/RTK/zip/master).
* Configure the project with CMake pointing to RTK's source directory and setting the CMake option `ITK_DIR` to ITK's compilation directory. All CMake options above can be set except `Module_RTK`.

Installation is currently not supported for independent RTK compilations.

Pre-compiled binaries
---------------------
We only provide pre-compiled binaries for the Python package which depends on ITK. Use the following commands to install RTK module with `pip`.
```
python -m pip install --upgrade pip
python -m pip install itk-rtk
```

The same operating systems and Python versions are supported as ITK's packages, see the list on [Pypi](https://pypi.org/project/itk-rtk).

Getting started
---------------
See [GettingStarted.md](GettingStarted.md). Your `CMakeLists.txt` can now use RTK when importing ITK as shown in the [FirstReconstruction's CMakeLists.txt](https://github.com/SimonRit/RTK/blob/master/examples/FirstReconstruction/CMakeLists.txt#L7).
