# Conjugate gradient

`````{tab-set}

````{tab-item} 3D

## 3D

![sin](../../documentation/docs/ExternalData/SheppLogan-Sinogram-3D.png){w=200px alt="SheppLogan sinogram 3D"}
![img](../../documentation/docs/ExternalData/ConjugateGradient-3D.png){w=200px alt="ConjugateGradient reconstruction"}

This script uses the SheppLogan phantom.

```{literalinclude} ConjugateGradient3D.sh
```

````
````{tab-item} 2D

## 2D

![sin](../../documentation/docs/ExternalData/ConjugateGradient-Sinogram-2D.png){w=200px alt="SHeppLogan sinogram 2D"}
![img](../../documentation/docs/ExternalData/ConjugateGradient-2D.png){w=200px alt="ConjugateGradient reconstruction"}

The same reconstruction can be performed using the original 2D Shepp-Logan phantom.
RTK can perform 2D reconstructions through images wide of 1 pixel in the y direction.
The following script performs the same reconstruction as above in a 2D environment and use the Shepp–Logan phantom (https://data.kitware.com/api/v1/file/67d1ff45c6dec2fc9c534d0d/download).

```{literalinclude} ConjugateGradient2D.sh
```
````
````{tab-item} Noisy Reconstruction

## Noisy Reconstruction

In the presence of noise, all projection data may not be equally reliable. The conjugate gradient algorithm can be modified to take this into account, and each pixel of the projections can be associated with a weight. The higher the weight, the more reliable the pixel data. Download [noisy projections](https://data.kitware.com/api/v1/item/5be99cdf8d777f2179a2e63d/download) and [the associated weights](https://data.kitware.com/api/v1/item/5be99d268d777f2179a2e6f8/download), as well as [the geometry](https://data.kitware.com/api/v1/item/5be99d268d777f2179a2e700/download), and run the following to compare the regular least squares reconstruction (without weights) and the weighted least squares reconstruction.

```{literalinclude} NoisyConjugateGradient.sh
```
````
`````
