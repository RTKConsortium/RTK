# Conjugate gradient

This example uses the Shepp–Logan phantom.

## 3D

![sin](ConjugateGradient-Sinogram.png){w=200px alt="Conjugate sinogram"}
![img](ConjugateGradient.png){w=200px alt="ConjugateGradient image"}

```shell
 # Create a simulated geometry
 rtksimulatedgeometry -n 180 -o geometry.xml
 # You may add "--arc 200" to make the scan short or "--proj_iso_x 200" to offset the detector

 # Create projections of the phantom file
 rtkprojectshepploganphantom -g geometry.xml -o projections.mha --spacing 2 --dimension 256

 # Reconstruct
 rtkconjugategradient -p . -r projections.mha -o 3dcg.mha -g geometry.xml --spacing 2 --dimension 256 -n 20

 # Create a reference volume for comparison
 rtkdrawshepploganphantom --spacing 2 --dimension 256  -o ref.mha
```

In the presence of noise, all projection data may not be equally reliable. The conjugate gradient algorithm can be modified to take this into account, and each pixel of the projections can be associated with a weight. The higher the weight, the more reliable the pixel data. Download [noisy projections](https://data.kitware.com/api/v1/item/5be99cdf8d777f2179a2e63d/download) and [the associated weights](https://data.kitware.com/api/v1/item/5be99d268d777f2179a2e6f8/download), as well as [the geometry](https://data.kitware.com/api/v1/item/5be99d268d777f2179a2e700/download), and run the following to compare the regular least squares reconstruction (without weights) and the weighted least squares reconstruction.

```shell
 # Perform least squares reconstruction
 rtkconjugategradient -p . -r noisyLineIntegrals.mha -o LeastSquares.mha -g geom.xml -n 20

 # Perform weighted least squares reconstruction
 rtkconjugategradient -p . -r noisyLineIntegrals.mha -o WeightedLeastSquares.mha -g geom.xml -w weightsmap.mha -n 20
```


Taking the weights into account slows down convergence. This can be corrected by using a preconditioner in the conjugate gradient algorithm. The preconditioner is computed automatically from the weights map, you just need to activate the flag :
```shell
 # Perform preconditioned conjugate gradient reconstruction with weighted least squares cost function
 rtkconjugategradient -p . -r noisyLineIntegrals.mha -o WeightedLeastSquares.mha -g geom.xml -w weightsmap.mha -n 20 --preconditioned
```

## 2D

The same reconstruction can be performed using the original 2D Shepp-Logan phantom.
RTK can perform 2D reconstructions through images wide of 1 pixel in the y direction.
The following script performs the same reconstruction as above in a 2D environment and use the Shepp–Logan phantom.

```{literalinclude} ConjugateGradient2D.sh
```
