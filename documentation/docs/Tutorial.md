# Tutorials

## Building a HelloWorld application

RTK is a library, therefore it's meant to be integrated into application. This tutorial shows how to create a simple FirstReconstruction project that links with RTK. The source code for this tutorial is located in [RTK/examples/FirstReconstruction](https://github.com/RTKConsortium/RTK/blob/master/examples/FirstReconstruction).

*   First you need to create a [CMakeLists.txt](https://github.com/RTKConsortium/RTK/blob/master/examples/FirstReconstruction/CMakeLists.txt)

```{literalinclude} ../../examples/FirstReconstruction/CMakeLists.txt
:language: cmake
```
*   Create a [FirstReconstruction.cxx](https://github.com/RTKConsortium/RTK/blob/master/examples/FirstReconstruction/FirstReconstruction.cxx) file
```{literalinclude} ../../examples/FirstReconstruction/FirstReconstruction.cxx
```
*   Run CMake on the FirstReconstruction directory and create a HelloWorld-bin,
*   Configure and build the project using your favorite compiler,
*   Run `FirstReconstruction image.mha geometry.xml`. If everything runs correctly, you should see a few messages ending with `Done!` and two new files in the current directory, image.mha and geometry.xml. image.mha is the reconstruction of a sphere from 360 projections and geometry.xml is the geometry file in the [RTK format](./Geometry.md).

## Modifying a basic RTK application

In [applications/rtktutorialapplication/](https://github.com/RTKConsortium/RTK/blob/master/applications/rtktutorialapplication), you will find a very basic RTK application that can be used as a starting point for building more complex applications.