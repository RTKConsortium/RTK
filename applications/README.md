Command-line examples
========

RTK provides command line applications that can be built from the C++ code by turning `ON` the `RTK_BUILD_APPLICATIONS` CMake option. A few of these applications have also been translated to Python and integrated in the [Pypi package](https://pypi.org/project/itk-rtk/). The options of each command line application can be listed with the `--help option`.

```{toctree}
:maxdepth: 1

./rtkfdk/README.md
./rtkconjugategradient/README.md
./rtkforwardprojections/README.md
./rtkdrawgeometricphantom/README.md
./rtkamsterdamshroud/README.md
./rtkelektasynergygeometry/README.md
./rtkvarianobigeometry/README.md
./rtkregularizedconjugategradient/README.md
./rtkadmmwavelets/README.md
./rtkfourdrooster/README.md
```

In [applications/rtktutorialapplication/](https://github.com/RTKConsortium/RTK/blob/master/applications/rtktutorialapplication), you will find a very basic RTK application that can be used as a starting point for building your own new application.
