# 3D, 2D and motion-compensated FDK

The following exampels illustrates the command line application `rtkfdk` by reconstructing a Shepp Logan phantom with Feldkamp, David and Kress algorithm in 3D (cone-beam) and 2D (fan-beam).

`````{tab-set}

````{tab-item} 3D

## 3D

![sin_3D](../../documentation/docs/ExternalData/SheppLogan-Sinogram-3D.png){w=200px alt="SheppLogan sinogram 3D "}
![img_3D](../../documentation/docs/ExternalData/Fdk-3D.png){w=200px alt="Fdk reconstruction 3D"}

This script uses the SheppLogan phantom.

```{literalinclude} FDK3D.sh
```

````
````{tab-item} 2D

## 2D

![sin_2D](../../documentation/docs/ExternalData/SheppLogan-Sinogram-2D.png){w=200px alt="SheppLogan sinogram 2D"}
![img_2D](../../documentation/docs/ExternalData/Fdk-2D.png){w=200px alt="Fdk reconstruction 2D"}

The same reconstruction can be performed using the original 2D Shepp-Logan phantom.
RTK can perform 2D reconstructions through images wide of 1 pixel in the y direction.
The following script performs the same reconstruction as above in a 2D environment and uses the [2D Shepp-Logan](https://data.kitware.com/api/v1/file/67d1ff45c6dec2fc9c534d0d/download) phantom as input.

```{literalinclude} FDK2D.sh
```
````

````{tab-item} Motion compensated

## Motion-compensated reconstruction

RTK provides the necessary tools to reconstruct an image with motion compensation. The implementation is based on two articles that we have published ([article 1](https://hal.archives-ouvertes.fr/hal-00443440) and [article 2](https://hal.archives-ouvertes.fr/hal-01967313)) but only the FDK-based motion-compensated CBCT reconstruction (analytic algorithm in article 1) and without optimization (very slow reconstruction compared to article 2). You should read the articles to understand the basics of the algorithm before trying to use the software.

The algorithm requires a set of projection images with the associated RTK geometry, the respiratory phase of each projection image and the 4D motion vector field over a respiratory cycle in the cone-beam coordinate system. Each piece of data is described in more details below and can be downloaded using [Girder](https://data.kitware.com/#collection/5a7706878d777f0649e04776). It is assumed that we have a breathing motion that is cyclic and similar to that described by the vector field. Note that you could modify the code and create your own motion model if you want to, in which case you should probably [contact us](http://www.openrtk.org/RTK/project/contactus.html).

### Projection images

This example is illustrated with a set of projection images of the [POPI patient](https://github.com/open-vv/popi-model/blob/master/popi-model.md). This dataset has been used in the first previously-mentioned article. You can [download the projections](https://data.kitware.com/api/v1/item/5be99af88d777f2179a2e144/download) and the required tables of the Elekta database, [FRAME.DBF](https://data.kitware.com/api/v1/item/5be99a068d777f2179a2cf4f/download) and [IMAGE.DBF](https://data.kitware.com/api/v1/item/5be99a078d777f2179a2cf65/download). The dataset is first used to reconstruct a blurry image:

```bash
# Convert Elekta database to RTK geometry
rtkelektasynergygeometry \
  -o geometry.rtk \
  -f FRAME.DBF \
  -i IMAGE.DBF \
  -u 1.3.46.423632.141000.1169042526.68

# Reconstruct from all projection images without any motion compensation
rtkfdk \
  -p . \
  -r .*.his \
  -o fdk.mha \
  -g geometry.rtk \
  --hann 0.5 \
  --pad 1.0

# Keep only the field-of-view of the image
rtkfieldofview \
  --reconstruction fdk.mha \
  --output fdk.mha \
  --geometry geometry.rtk \
  --path . \
  --regexp '.*.his'
```

You should obtain something like that with [VV](http://vv.creatis.insa-lyon.fr/):

![Blurred](../../documentation/docs/ExternalData/Blurred.jpg){w=600px alt="Blurred image"}

### Deformation vector field

The next piece of data is a 4D deformation vector field that describes a respiratory cycle. Typically, it can be obtained from the 4D planning CT with deformable image registration. Here, I have used [Elastix](http://elastix.lumc.nl/) with the [sliding module](http://elastix.lumc.nl/modelzoo/par0016) developed by Vivien Delmon. The registration uses a [patient mask](https://data.kitware.com/api/v1/item/5be99a408d777f2179a2dde8/download) (red+green) and a [motion mask](https://data.kitware.com/api/v1/item/5be99a088d777f2179a2cf6f/download) (red) as described in [Jef's publication](http://www.creatis.insa-lyon.fr/site/fr/publications/VAND-12):

![Mm](../../documentation/docs/ExternalData/MotionMask.jpg){w=400px alt="Motion mask"}

The registration can easily be scripted, here with bash, where each phase image of the POPI 4D CT has been stored in files 00.mhd to 50.mhd:

```bash
for i in $(seq -w 0 10 90)
do
  mkdir $i
  elastix -f 50.mhd \
          -m $i.mhd \
          -out $i \
          -labels mm_50.mha \
          -fMask patient_50.mha \
          -p Par0016.multibsplines.lung.sliding.txt
done
```

Deformable Image Registration is a complex and long process so you will have to be patient here. Note that the reference frame is phase 50% and it is registered to each phase from 0% to 90%. One subtle step is that the vector field is a displacement vector field, i.e., each vector is the local displacement of the point at its location. Since I ran the registration on the 4D planning CT, the coordinate system is not that of the cone-beam CT. In order to produce the vector field in the cone-beam coordinate system, I have used the following bash script that combines transformix and several "clitk" tools that are provided along with VV:

```bash
# Create 4x4 matrix that describes the CT to CBCT change of coordinate system.
# This matrix is a combination of the knowledge of the isocenter position / axes orientation
# and a rigid alignment that has been performed with Elastix
echo "-0.0220916855767852  0.9996655273534405 -0.0134458487848415 -83.6625731437426197"  >CT_CBCT.mat
echo " 0.0150924269790251 -0.0131141301144939 -0.9998000991394341  -4.0763571826687057" >>CT_CBCT.mat
echo " 0.9996420239647088  0.0222901999207823  0.0147976657359281  77.8903364738220034" >>CT_CBCT.mat
echo " 0.0000000000000000  0.0000000000000000  0.0000000000000000   1.0000000000000000" >>CT_CBCT.mat

# Transform 4x4 matrix that describes the transformation
# from planning CT to CBCT to a vector field
clitkMatrixTransformToVF --like 50.mhd \
                         --matrix CT_CBCT.mat \
                         --output CT_CBCT.mha

# Inverse transformation. Also remove upper slices that are outside the
# planning CT CBCT_CT.mat is the inverse of CT_CBCT.mha
clitkMatrixInverse -i CT_CBCT.mat \
                   -o CBCT_CT.mat
clitkMatrixTransformToVF --origin -127.5,-107.5,-127.5 \
                         --spacing 1,1,1 \
                         --size 256,236,256 \
                         --matrix CBCT_CT.mat \
                         --output CBCT_CT.mha

# Go over each elastix output file, generate the vector field with
# transformix and compose with the two rigid vector fields
for i in $(seq -w 0 10 90)
do
  transformix -in 50.mhd \
              -out $i \
              -tp $i/TransformParameters.0.txt \
              -def all -threads 16
  clitkComposeVF --input1 CBCT_CT.mha \
                 --input2 $i/deformationField.mhd \
                 --output $i/deformationField.mhd
  clitkComposeVF --input1 $i/deformationField.mhd \
                 --input2 CT_CBCT.mha \
                 --output $i/deformationField.mhd
done
```

This is a bit complicated and there are probably other ways of doing this. For example, Vivien has resampled the planning CT frames on the CBCT coordinate system before doing the registrations, in which case you do not need to do all this. Just pick one of your choice but motion-compensated CBCT reconstruction requires a 4D vector field that is nicely displayed on top of a CBCT image, for example the fdk.mha that has been produced in the first step (the vector field is downsampled and displayed with VV):

![Vf](../../documentation/docs/ExternalData/VectorField.gif){w=400px alt="Vector field"}

The elastix output files and the transformed 4D DVF are available [here](https://data.kitware.com/api/v1/item/5be99a058d777f2179a2cf42/download).

### Respiratory signal

The motion model requires that we associate each projection image with one frame of the 4D vector field. We used the Amsterdam shroud solution of Lambert Zijp (described [here](http://www.creatis.insa-lyon.fr/site/fr/publications/RIT-12a)) which is implemented in RTK

```bash
rtkamsterdamshroud --path . \
                   --regexp '.*.his' \
                   --output shroud.mha \
                   --unsharp 650
rtkextractshroudsignal --input shroud.mha \
                       --output signal.txt
```

Post-process with Matlab to obtain the phase signal, ensuring the phase ranges from 0 to 1 (e.g., 0.3 corresponds to 30% of the respiratory cycle). The resulting phase is visualized [here](https://data.kitware.com/api/v1/item/5be99af98d777f2179a2e160/download):

![Signal](../../documentation/docs/ExternalData/Signal.jpg){w=800px alt="Phase signal"}

---

### Motion-compensated cone-beam CT reconstruction

Gather all the pieces to perform motion-compensated reconstruction. Use the following commands:

Reconstruct with Motion Compensation :
```bash
rtkfdk \
  -p . \
  -r .*.his \
  -o fdk.mha \
  -g geometry.rtk \
  --hann 0.5 \
  --pad 1.0 \
  --signal sphase.txt \
  --dvf deformationField_4D.mhd
```

Apply the Field-of-View Filter :
```bash
rtkfieldofview \
  --reconstruction fdk.mha \
  --output fdk.mha \
  --geometry geometry.rtk \
  --path . \
  --regexp '.*.his'
```

Toggle between uncorrected and motion-compensated reconstruction to appreciate the improvement:

![Blurred vs mc.gif](../../documentation/docs/ExternalData/Blurred_vs_mc.gif){w=400 alt="blurred vs motion compensation image"}

The 4D vector field is constructed with phase 50% as a reference. Modify the reference image to reconstruct other phases, such as the time-average position.
````
`````
