# Forward Projection

![sin](../../documentation/docs/ExternalData/POPI-Sinogram.png){w=400px alt="POPI sinogram"}
![img](../../documentation/docs/ExternalData/POPI-Reconstruction.png){w=400px alt="POPI reconstruction"}

This script uses the files [00.mhd](http://www.creatis.insa-lyon.fr/~srit/POPI/MedPhys11/bl/mhd/00.mhd) and [00.raw](http://www.creatis.insa-lyon.fr/~srit/POPI/MedPhys11/bl/mhd/00.raw) of the [POPI](https://github.com/open-vv/popi-model/blob/master/popi-model.md) as input.

```{literalinclude} ForwardProjection.sh
```

Note that the original file is in Hounsfield units which explains the negative values in the projection images since, e.g., the attenuation of air is -1000 HU.

It is also worth of note that the file is oriented in the DICOM coordinate system although RTK uses the IEC 61217 which results in a rotation around the antero-posterior axis of the patient. This can be easily changed by modifying the TransformMatrix in the 00.mhd file.
