#   Generate 4D data

This example shows how to generate a full set of input data to run 4D examples. It contains:
- a moving phantom, made of two ellipsoids, one of which moves along the first dimension
- a geometry
- a phase signal
- the set of projections of the moving phantom
- a deformation vector field describing the motion of the moving ellipsoid
- the corresponding inverse deformation vector field

You can also skip this part and [download the data](https://data.kitware.com/#collection/5a7706878d777f0649e04776/folder/69611c373b9bcc32c3188531).

`````{tab-set}

````{tab-item} C++

```{literalinclude} ./GenerateFourDData.cxx
:language: c++
```
````

````{tab-item} Python

```{literalinclude} ./GenerateFourDData.py
:language: python
```
````
`````
