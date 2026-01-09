#   4D Conjugate Gradient

FourDConjugateGradient shows how to perform iterative cone-beam CT reconstruction using either CPU or GPU resources.
You can refer to the [projectors documentation](../../documentation/docs/Projectors.md) to see all options available for the back and forwardprojections.

This example generates its own input data: a phantom made of two ellipsoids, one of which is moving. Projections are computed through this moving phantom.

`````{tab-set}

````{tab-item} C++

```{literalinclude} ./FourDConjugateGradient.cxx
:language: c++
```
````

`````
