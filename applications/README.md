# Command-line applications

RTK provides command line applications which can be built from the C++ code by turning `ON` the `RTK_BUILD_APPLICATIONS` CMake option. A few of these applications have also been translated to Python and integrated in the [Pypi package](https://pypi.org/project/itk-rtk/). The options of each command line application can be listed with the `--help option`.

The following are examples using RTK applications:

```{toctree}
:maxdepth: 1

./rtkfdk/README.md
./rtkconjugategradient/README.md
./rtkforwardprojections/README.md
./rtkdrawgeometricphantom/README.md
./rtkamsterdamshroud/README.md
./rtkelektasynergygeometry/README.md
./rtkvarianobigeometry/README.md
./rtkadmmtotalvariation/README.md
./rtkadmmwavelets/README.md
./rtkfourdrooster/README.md
./rtkshowgeometry/README.md
```

In [applications/rtktutorialapplication/](https://github.com/RTKConsortium/RTK/blob/main/applications/rtktutorialapplication), you will find a very basic RTK application that can be used as a starting point for building your own new application.

## Using RTK applications in Python

RTK applications which have been translated to Python can also be called from Python in addition to the command line, thus avoiding reloading the RTK package (which is slow) and enabling the flexibility of Python scripting. The Python interface dynamically maps the RTK applications to Python functions. You can use `help(rtk.<application_name>)` to see all required and optional arguments. Note that only file paths are supported as input, ITK images or RTK geometry objects cannot be passed directly.

Each Python application accepts either:

1. **A single positional string argument**: This mimics the command-line usage.
2. **A list of keyword arguments**: This provides a more Pythonic interface.

```python
rtk.<application_name>(<arguments>)
```

For example,

```bash
rtkfdk -p . -r projections.mha -o fdk.mha -g geometry.xml --spacing 2 --dimension 256
```

is equivalent to execute from Python

```python
from itk import RTK as rtk

rtk.rtkfdk(
    "-p . -r projections.mha -o fdk.mha -g geometry.xml --spacing 2 --dimension 256"
)
```

or

```python
from itk import RTK as rtk

rtk.rtkfdk(
    path=".",
    regexp="projections.mha",
    output="fdk.mha",
    geometry="geometry.xml",
    spacing="2,2,2",
    dimension=[256,256,256] # You can use a string or a list
)
```
