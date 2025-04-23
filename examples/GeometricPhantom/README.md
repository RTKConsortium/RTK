# Project, draw and reconstruct geometric phantom

GeometricPhantom demonstrates how to create 2D projections of a Forbild phantom, draw a reference 3D image and reconstruct from the projections with [FDK](http://www.openrtk.org/Doxygen/classrtk_1_1FDKConeBeamReconstructionFilter.html). You can find how to create your own phantom file in the [phantom definition page](../../documentation/docs/Phantom.md).

This example uses the [Thorax](https://raw.githubusercontent.com/RTKConsortium/Forbild/refs/heads/main/Thorax) Forbild phantom, available in the [dedicated GitHub repository](https://github.com/RTKConsortium/Forbild) for Forbild phantom files compatible with RTK.

`````{tab-set}

````{tab-item} C++

```{literalinclude} ./GeometricPhantom.cxx
:language: c++
```
````

````{tab-item} Python

```{literalinclude} ./GeometricPhantom.py
:language: python
```

````
`````

The results displayed with [rtkshowgeometry](../../applications/rtkshowgeometry/README.md) are:

![visu](../../documentation/docs/ExternalData/Thorax-visualisation.png){alt="Visualisation"}
