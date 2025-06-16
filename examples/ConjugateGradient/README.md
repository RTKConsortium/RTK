#   Conjugate Gradient

ConjugateGradient shows how to perform iterative cone-beam CT reconstruction using either CPU and GPU resources.
You can refere to the [projectors documentation](https://docs.openrtk.org/en/latest/documentation/docs/Projectors.html) to see all options available for the back and forwardprojections.

This example uses the [Thorax](https://raw.githubusercontent.com/RTKConsortium/Forbild/refs/heads/main/Thorax) Forbild phantom, available in the [dedicated GitHub repository](https://github.com/RTKConsortium/Forbild) for Forbild phantom files compatible with RTK.

`````{tab-set}

````{tab-item} C++

```{literalinclude} ./ConjugateGradient.cxx
:language: c++
```
````

````{tab-item} Python

```{literalinclude} ./ConjugateGradient.py
:language: python
```

````
`````

The results displayed with [rtkshowgeometry](../../applications/rtkshowgeometry/README.md) are:

![visu](../../documentation/docs/ExternalData/Thorax-visualisation.png){alt="Visualisation"}
