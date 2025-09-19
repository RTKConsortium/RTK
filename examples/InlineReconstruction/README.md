# Inline Reconstruction

InlineReconstruction demonstrates how to perform cone-beam CT reconstruction using
RTK's implementation of [FDK](https://doi.org/10.1364/JOSAA.1.000612) while
projections are being acquired.

The program simulates the acquisition of X-ray projections in a separate thread,
gradually reconstructing the 3D volume as new projections become available.


`````{tab-set}

````{tab-item} C++

```{literalinclude} ./InlineReconstruction.cxx
:language: c++
```
````

````{tab-item} Python

```{literalinclude} ./InlineReconstruction.py
:language: python
```

````
`````
