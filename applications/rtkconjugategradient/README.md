# Conjugate gradient

This example uses the Shepp–Logan phantom.

`````{tab-set}

````{tab-item} 3D

## 3D

![sin](../../documentation/docs/ExternalData/ConjugateGradient-Sinogram.png){w=200px alt="Conjugate sinogram"}
![img](../../documentation/docs/ExternalData/ConjugateGradient.png){w=200px alt="ConjugateGradient image"}

```{literalinclude} ConjugateGradient3D.sh
```

````
````{tab-item} 2D

## 2D

The same reconstruction can be performed using the original 2D Shepp-Logan phantom.
RTK can perform 2D reconstructions through images wide of 1 pixel in the y direction.
The following script performs the same reconstruction as above in a 2D environment and use the Shepp–Logan phantom.

```{literalinclude} ConjugateGradient2D.sh
```
````
````{tab-item} Noisy Reconstruction

## Noisy Reconstruction

In the presence of noise, all projection data may not be equally reliable. The conjugate gradient algorithm can be modified to take this into account, and each pixel of the projections can be associated with a weight. The higher the weight, the more reliable the pixel data. Download [noisy projections](https://data.kitware.com/api/v1/item/5be99cdf8d777f2179a2e63d/download) and [the associated weights](https://data.kitware.com/api/v1/item/5be99d268d777f2179a2e6f8/download), as well as [the geometry](https://data.kitware.com/api/v1/item/5be99d268d777f2179a2e700/download), and run the following to compare the regular least squares reconstruction (without weights) and the weighted least squares reconstruction.

Taking the weights into account slows down convergence. This can be corrected by using a preconditioner in the conjugate gradient algorithm. The preconditioner is computed automatically.

```{literalinclude} NoisyConjugateGradient.sh
```
````
`````