# Command-line applications

RTK provides command line applications that can be built from the C++ code by turning `ON` the `RTK_BUILD_APPLICATIONS` CMake option. A few of these applications have also been translated to Python and integrated in the [Pypi package](https://pypi.org/project/itk-rtk/). The options of each command line application can be listed with the `--help option`.

The following are examples of RTK applications:

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

In [applications/rtktutorialapplication/](https://github.com/RTKConsortium/RTK/blob/master/applications/rtktutorialapplication), you will find a very basic RTK application that can be used as a starting point for building your own new application.

## Using RTK Applications in Python

RTK applications can be accessed directly in Python, allowing you to use the same functionality as the command-line applications but with the flexibility of Python scripting. The Python interface dynamically maps the RTK applications to Python functions. You can use `help(rtk.<application_name>)` to see all required and optional arguments. Note that only file paths are supported as input, ITK objects cannot be passed directly, and positional arguments are not handled.

### General Usage

Each application accepts either:

1. **A single string argument**: This mimics the command-line usage.
2. **Keyword arguments**: This provides a more Pythonic interface.

```python
rtk.<application_name>(<arguments>)
```

### Example: Running `rtkfdk`

#### Command-Line:
```bash
rtkfdk -p . -r projections.mha -o fdk.mha -g geometry.xml --spacing 2 --dimension 256,256,256
```

#### Python Equivalent:

Single string argument:
```python
rtk.rtkfdk(
    "-p . -r projections.mha -o fdk.mha -g geometry.xml --spacing 2 --dimension 256"
)
```

Keyword arguments:
```python
rtk.rtkfdk(
    path=".",
    regexp="projections.mha",
    output="fdk.mha",
    geometry="geometry.xml",
    spacing="2,2,2",
    dimension=[256,256,256] # You can use a string or a list
)
```