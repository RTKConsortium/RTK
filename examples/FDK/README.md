# FDK

Reconstruction of a phantom using Feldkamp, David and Kress cone-beam reconstruction.

---

This example uses the Shepp–Logan phantom.

## 3D

![sin_3D](SheppLogan-3D-Sinogram.png){w=200px alt="Shepp-Logan 3D sinogram"}
![img_3D](SheppLogan-3D.png){w=200px alt="Shepp-Logan 3D image"}

```{literalinclude} Code3D.sh
```

## 2D

![sin_2D](SheppLogan-2D-Sinogram.png){w=200px alt="Shepp-Logan 2D sinogram"}
![img_2D](SheppLogan-2D.png){w=200px alt="Shepp-Logan 2D image"}

The same reconstruction can be performed using the original 2D Shepp-Logan phantom.
RTK can perform 2D reconstructions through images wide of 1 pixel in the y direction.

```{literalinclude} Code2D.sh
```

---

If you want to create your own phantom, you can follow the documentation [here](../../documentation/docs/Phantom.md).
You can find the SheppLogan example [here](https://data.kitware.com/api/v1/file/674da1cc2da68b3050ac6a02/download).

```{literalinclude} CustomPhantom.sh
```
