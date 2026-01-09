# Motion-compensated 4D conjugate gradient

Please start by reading the [main documentation page on 3D + time reconstruction](../../documentation/docs/3d_time.md) if you have not read it already. It gives the necessary background to understand the 3D + time reconstruction tools, including this one.

RTK provides several tools to reconstruct a 3D + time image using a known motion estimate. Motion-compensated 4D conjugate gradient is one of them.

The algorithm requires a set of projection images with the associated RTK geometry, the respiratory phase of each projection image and a 3D + time displacement vector field (DVF) representing the estimated motion. Each piece of data is described in more details below. It is assumed that the breathing motion is periodic, which implies that the position of the chest depends only on the respiratory phase.

## Projection images

This example is illustrated with a dataset of a patient used in the [MA-ROOSTER paper](https://hal.science/hal-01375767). You can [download the projections](https://data.kitware.com/api/v1/item/5be97d688d777f2179a28e39/download), which also contain the RTK geometry file.

## Respiratory signal

The algorithm requires that we associate each projection image with its position in the respiratory cycle when it has been acquired. The cphase.txt is already zipped together with the projections, but you can visit [this page](../rtkamsterdamshroud/README.md) to learn how to generate it from the projections using the Amsterdam Shroud.

## 4D displacement vector field (DVF)

The forward and inverse 4D DVFs for this patient can be downloaded [here](https://data.kitware.com/api/v1/item/5be989e68d777f2179a29e95/download). Please refer to the tutorial on [Motion-Compensated FDK](../rtkfdk/README.md) if you wish to learn how to obtain a valid 4D DVF.
Copy paste the projections and DVFs in the same folder, then from that folder, run

```
# Reconstruct with motion-compensated 4D conjugate gradient
rtkmcfourdconjugategradient \
-p . \
-r correctedProjs.mha \
-o mc_fourdcg.mha \
-g geom.xml \
--signal cphase.txt \
--niter 2 \
--spacing "1, 1, 1, 1" \
--size "220, 280, 370, 10" \
--origin "-140, -140, -75, 0" \
--frames 10 \
--dvf toPhase50_4D.mhd \
--idvf fromPhase50_4D.mhd \
--cudacg
```

Compiling RTK with `RTK_USE_CUDA=ON` and having a working and CUDA-enabled nVidia GPU is mandatory for motion-compensated 4D conjugate gradient, and so is the `--cudacg` option, as the filters for [warped back projection](https://www.openrtk.org/Doxygen/classrtk_1_1WarpProjectionStackToFourDImageFilter.html) and [warped forward projection](https://www.openrtk.org/Doxygen/classrtk_1_1WarpFourDToProjectionStackImageFilter.html) are only implemented using CUDA. If one of these conditions is missing, the algorithm does not crash, but reverts to CPU-based non-motion-compensated forward and back projections.
Note the option `--idvf` to provide the inverse DVF. It is also mandatory, as it is needed to perform the warped forward projections.
