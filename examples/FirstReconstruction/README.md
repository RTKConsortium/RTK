# First reconstruction

RTK is a library, therefore it's meant to be integrated into application. This tutorial shows how to create a simple FirstReconstruction project that links with RTK. The source code for this tutorial is located in [RTK/examples/FirstReconstruction](https://github.com/RTKConsortium/RTK/blob/master/examples/FirstReconstruction).


`````{tab-set}

````{tab-item} C++

## CMakeLists

*   First you need to create a [CMakeLists.txt](https://github.com/RTKConsortium/RTK/blob/master/examples/FirstReconstruction/CMakeLists.txt)

```{literalinclude} ./CMakeLists.txt
:language: cmake
```

*   Run CMake on the FirstReconstruction directory and create a FirstReconstruction-bin,
*   Configure and build the project using your favorite compiler,
*   Run `FirstReconstruction image.mha geometry.xml`. If everything runs correctly, you should see a few messages ending with `Done!` and two new files in the current directory, image.mha and geometry.xml. image.mha is the reconstruction of a sphere from 360 projections and geometry.xml is the geometry file in [RTK's geometry format](../../documentation/docs/Geometry.md).

## Code

```{literalinclude} ./FirstReconstruction.cxx
:language: c++
```
````

````{tab-item} C++ Cuda

## CMakeLists

*   First you need to create a [CMakeLists.txt](https://github.com/RTKConsortium/RTK/blob/master/examples/FirstReconstruction/CMakeLists.txt)

```{literalinclude} ./CMakeLists.txt
:language: cmake
```

*   Run CMake on the FirstReconstruction directory and create a FirstReconstruction-bin,
*   Configure and build the project using your favorite compiler,
*   Run `FirstReconstruction image.mha geometry.xml`. If everything runs correctly, you should see a few messages ending with `Done!` and two new files in the current directory, image.mha and geometry.xml. image.mha is the reconstruction of a sphere from 360 projections and geometry.xml is the geometry file in [RTK's geometry format](../../documentation/docs/Geometry.md).

## Code

```{literalinclude} ./FirstCudaReconstruction.cxx
:language: c++
```

````

````{tab-item} Python

## Code

```{literalinclude} ./FirstReconstruction.py
:language: python
```

````

````{tab-item} Python Cuda

## Code

```{literalinclude} ./FirstCudaReconstruction.py
:language: python
```
````
`````