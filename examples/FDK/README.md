# FDK

Reconstruction of the Shepp–Logan phantom using Feldkamp, David and Kress cone-beam reconstruction.

## 3D

![sin_3D](SheppLogan-3D-Sinogram.png){w=200px alt="Shepp-Logan 3D sinogram"}
![img_3D](SheppLogan-3D.png){w=200px alt="Shepp-Logan 3D image"}

This script uses the file [SheppLogan.txt](https://data.kitware.com/api/v1/item/5b179c938d777f15ebe2020b/download) as input.

```{literalinclude} FDK3D.sh
```

## 2D

![sin_2D](SheppLogan-2D-Sinogram.png){w=200px alt="Shepp-Logan 2D sinogram"}
![img_2D](SheppLogan-2D.png){w=200px alt="Shepp-Logan 2D image"}

The same reconstruction can be performed using the original 2D Shepp-Logan phantom.
RTK can perform 2D reconstructions through images wide of 1 pixel in the y direction.
The following script performs the same reconstruction as above in a 2D environment and uses the [2D Shepp-Logan](http://wiki.openrtk.org/images/7/73/SheppLogan-2d.txt) phantom as input.

```{literalinclude} FDK2D.sh
```
