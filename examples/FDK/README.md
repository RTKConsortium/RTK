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

```shell
# Create a simulated geometry
rtksimulatedgeometry -n 180 -o geometry.xml

# Create projections of the phantom file
rtkprojectgeometricphantom -g geometry.xml -o projections.mha --spacing 2 --dimension=512,3,512 --phantomfile YourPhantom.txt --phantomscale=256,1,256

# Reconstruct
rtkfdk -p . -r projections.mha -o fdk.mha -g geometry.xml --spacing 2 --dimension 256,1,256

# Create a reference volume for comparison
rtkdrawshepploganphantom --spacing 2 --dimension=256,1,256 --phantomfile YourPhantom.txt -o ref.mha --phantomscale=256,1,256
```
