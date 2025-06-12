# 4DROOSTER: Total variation-regularized 3D + time reconstruction

RTK provides a tool to reconstruct a 3D + time image which does not require explicit motion estimation. Instead, it uses total variation regularization along both space and time. The implementation is based on a paper that we have published ([article](http://www.creatis.insa-lyon.fr/site/fr/publications/MORY-14)). You should read the article to understand the basics of the algorithm before trying to use the software.

The algorithm requires a set of projection images with the associated RTK geometry, the respiratory phase of each projection image and a motion mask in which the region of the space where movement is expected is set to 1 and the rest is set to 0. Each piece of data is described in more details below and can be downloaded using [Girder](https://data.kitware.com/#collection/5a7706878d777f0649e04776). It is assumed that the breathing motion is periodic, which implies that the mechanical state of the chest depends only on the respiratory phase.

## Projection images

This example is illustrated with a set of projection images of the [POPI patient](https://github.com/open-vv/popi-model/blob/master/popi-model.md). You can [download the projections](https://data.kitware.com/api/v1/item/5be99af88d777f2179a2e144/download) and the required tables of the Elekta database, [FRAME.DBF](https://data.kitware.com/api/v1/item/5be99a068d777f2179a2cf4f/download) and [IMAGE.DBF](https://data.kitware.com/api/v1/item/5be99a078d777f2179a2cf65/download). The dataset is first used to reconstruct a blurry image:

```
# Convert Elekta database to RTK geometry
rtkelektasynergygeometry \
  -o geometry.rtk \
  -f FRAME.DBF \
  -i IMAGE.DBF \
  -u 1.3.46.423632.141000.1169042526.68

# Reconstruct a 3D volume from all projection images.
# We implicitly assume that the patient's chest is static, which is wrong.
# Therefore the reconstruction will be blurry. This blurry reconstruction is
# not required for 4D ROOSTER, but there is usually no other way to generate
# a motion mask, which is required. Additionally, the blurry reconstruction
# can be used to initialize the 4D ROOSTER reconstruction.
rtkfdk \
  -p . \
  -r .*.his \
  -o fdk.mha \
  -g geometry.rtk \
  --hann 0.5 \
  --pad 1.0 \
  --size 160 \
  --spacing 2
```

You should obtain something like that with [VV](http://vv.creatis.insa-lyon.fr/):

![Blurred](../../documentation/docs/ExternalData/Blurred.jpg){w=600px alt="Blurred image"}

## Motion mask

The next piece of data is a 3D motion mask: a volume that contains only zeros, where no movement is expected to occur, and ones, where movement is expected. Typically, during breathing, the whole chest moves, so it is safe to use the [patient mask](https://data.kitware.com/api/v1/item/5be99a418d777f2179a2ddf4/download) (red+green). However, restricting the movement to a smaller region, e.g. the rib cage, can help reconstruct this region more accurately and mitigate artifacts, so we might want to use the [rib cage mask](https://data.kitware.com/api/v1/item/5be99a2c8d777f2179a2dc4f/download) (red):

![Mm](../../documentation/docs/ExternalData/MotionMask.jpg){w=400px alt="Motion mask"}

## Respiratory signal

The 4D ROOSTER algorithm requires that we associate each projection image with the instant of the respiratory cycle at which it has been acquired. We used the Amsterdam shroud solution of Lambert Zijp (described [here](http://www.creatis.insa-lyon.fr/site/fr/publications/RIT-12a)) which is implemented in RTK

```
rtkamsterdamshroud --path . \
                   --regexp '.*.his' \
                   --output shroud.mha \
                   --unsharp 650
rtkextractshroudsignal --input shroud.mha \
                       --output signal.txt \
                       --phase sphase.txt
```

to get the phase signal. Note that the phase must go from 0 to 1 where 0.3 corresponds to 30% in the respiratory cycle, i.e., frame 3 if you have a 10-frames 4D reconstruction or frame 6 if you have a 20-frames 4D reconstruction. The [resulting phase](https://data.kitware.com/api/v1/item/5be99af98d777f2179a2e160/download) is in green on top of the blue respiratory signal and the detected end-exhale peaks:

![Signal](../../documentation/docs/ExternalData/Signal.jpg){w=800px alt="Phase signal"}

## ROOSTER for conebeam CT reconstruction

We now have all the pieces to perform a 3D + time reconstruction. The algorithm will perform "niter" iterations of the main loop, and run three inner loops at each iteration:

*   conjugate gradient reconstruction with "cgiter" iterations
*   total-variation regularization in space, with parameter "gamma_space" (the higher, the more regularized) and "tviter" iterations
*   total-variation regularization in time, with parameter "gamma_time" (the higher, the more regularized) and "tviter" iterations

The number of iterations suggested here should work in most situations. The parameters gamma_space and gamma_time, on the other hand, must be adjusted carefully for each type of datasets (and, unfortunately, for each resolution). Unlike in analytical reconstruction methods like FDK, with 4D ROOSTER it is also very important that the reconstruction volume contains the whole patient's chest. You should therefore adapt the "size", "spacing" and "origin" parameters carefully, based on what you have observed on the blurry FDK reconstruction.

```
# Reconstruct from all projection images with 4D ROOSTER
rtkfourdrooster \
  -p . \
  -r .*.his \
  -o rooster.mha \
  -g geometry.rtk \
  --signal sphase.txt \
  --motionmask MotionMask.mha \
  --gamma_time 0.0001 \
  --gamma_space 0.0001 \
  --niter 30 \
  --cgiter 4 \
  --tviter 10 \
  --spacing 2 \
  --size 160 \
  --frames 5
```

Depending on the resolution you choose, and even on powerful computers, the reconstruction time can range from a few minutes (very low resolution, typically 32³ * 5, only for tests) to many hours (standard resolution, typically 256³ * 10). To speed things up, it is recommended to use the CUDA version of the forward and back projectors by running this command instead:

```
# Reconstruct from all projection images with 4D ROOSTER, using CUDA forward and back projectors
rtkfourdrooster \
  -p . \
  -r .*.his \
  -o rooster.mha \
  -g geometry.rtk \
  --signal sphase.txt \
  --motionmask MotionMask.mha \
  --fp CudaRayCast \
  --bp CudaVoxelBased \
  --gamma_time 0.0001 \
  --gamma_space 0.0001 \
  --niter 30 \
  --cgiter 4 \
  --tviter 10 \
  --spacing 2 \
  --size 160 \
  --frames 5
```

With a recent GPU, this should allow you to perform a standard resolution reconstruction in less than one hour.

Note that the reconstructed volume in this example does not fully contain the attenuating object, causing hyper-attenuation artifacts on the borders of the result. To avoid these artifacts, reconstruct a larger volume (--size 256) should be fine. Note that you will have to resize your motion mask as well, as 3D the motion mask is expected to have the same size, spacing and origin as the first 3 dimensions of the 4D output.

## Motion-Aware 4D Rooster

4D ROOSTER doesn't require explicit motion information, but can take advantage of it to guide TV-regularization if motion information is available. Please refer to the tutorial on Motion-Compensated FDK to learn how to obtain a valid 4D Displacement Vector Field (DVF). Once you have it, simply adding the option --dvf "4D_DVF_Filename" to the list of rtkfourdrooster arguments will run MA-ROOSTER instead of 4D ROOSTER.

```
# Reconstruct from all projection images with MA ROOSTER
rtkfourdrooster \
  -p . \
  -r .*.his \
  -o rooster.mha \
  -g geometry.rtk \
  --signal sphase.txt \
  --motionmask MotionMask.mha \
  --gamma_time 0.0001 \
  --gamma_space 0.0001 \
  --niter 30 \
  --cgiter 4 \
  --tviter 10 \
  --spacing 2 \
  --size 160 \
  --frames 5 \
  --dvf deformationField_4D.mhd
```

Making use of the motion information adds to computation time. If you have compiled RTK with RTK_USE_CUDA = ON and have a working and CUDA-enabled nVidia GPU, it will automatically be used to speed up that part of the process.

The article in which the theoretical foundations of MA-ROOSTER are presented (link to be added) contains results obtained on two patients. If you wish to reproduce these results, you can download all the necessary data here:

*   Original projections, log-transformed projections with the table removed, motion mask, respiratory signal, and transform matrices to change from CT to CBCT coordinates and back
    *   [Patient 1](https://data.kitware.com/api/v1/item/5be97d688d777f2179a28e39/download)
    *   [Patient 2](https://data.kitware.com/api/v1/item/5be99df68d777f2179a2e904/download)
*   Inverse-consistent 4D Displacement Vector Fields, to the end-exhale phase and from the end-exhale phase
    *   [Patient 1](https://data.kitware.com/api/v1/item/5be989e68d777f2179a29e95/download)
    *   [Patient 2](https://data.kitware.com/api/v1/item/5be9a1388d777f2179a2f44d/download)

Extract the data of each patient in a separate folder. From the folder containing the data of patient 1, run the following command line:

```
# Reconstruct patient 1 with MA ROOSTER
rtkfourdrooster \
  -p . \
  -r correctedProjs.mha \
  -o marooster.mha \
  -g geom.xml \
  --signal cphase.txt \
  --motionmask dilated_resampled_mm.mhd \
  --gamma_time 0.0002 \
  --gamma_space 0.00005 \
  --niter 10 \
  --cgiter 4 \
  --tviter 10 \
  --spacing "1, 1, 1, 1" \
  --size "220, 280, 370, 10" \
  --origin "-140, -140, -75, 0" \
  --frames 10 \
  --dvf toPhase50_4D.mhd \
  --idvf fromPhase50_4D.mhd
```

From the folder containing the data of patient 2, run the following command line (only the size and origin parameters are different):

```
# Reconstruct patient 2 with MA ROOSTER
rtkfourdrooster \
  -p . \
  -r correctedProjs.mha \
  -o marooster.mha \
  -g geom.xml \
  --signal cphase.txt \
  --motionmask dilated_resampled_mm.mhd \
  --gamma_time 0.0002 \
  --gamma_space 0.00005 \
  --niter 10 \
  --cgiter 4 \
  --tviter 10 \
  --spacing "1, 1, 1, 1" \
  --size "285, 270, 307, 10" \
  --origin "-167.5, -135, -205, 0" \
  --frames 10 \
  --dvf toPhase50_4D.mhd \
  --idvf fromPhase50_4D.mhd \
```

Note the option "--idvf", which allows to provide the inverse DVF. It is used to inverse warp the 4D reconstruction after the temporal regularization. MA-ROOSTER will work with and without the inverse DVF, and yield almost the same results in both cases. Not using the inverse DVF is approximately two times slower, as it requires MA-ROOSTER to perform the inverse warping by an iterative method.

Again, if you have a CUDA-enabled GPU (in this case with at least 3 GB of VRAM), and have compiled RTK with RTK_USE_CUDA = ON, you can add the "--bp CudaVoxelBased" and "--fp CudaRayCast" to speed up the computation by performing the forward and back projections on the GPU.

You do not need the 4D planning CT data to perform the MA-ROOSTER reconstructions. It is only required to compute the DVFs, which can be downloaded above. We do provide it anyway, in case you want to use your own method, or the one described in Motion-Compensated FDK, to extract a DVF from it:

*   4D planning CT
    *   [Patient 1](https://data.kitware.com/api/v1/item/5be98bd28d777f2179a2a279/download)
    *   [Patient 2](https://data.kitware.com/api/v1/item/5be9a1918d777f2179a2f568/download)
