# 4D FDK

Please start by reading the [main documentation page on 3D + time reconstruction](../../documentation/docs/3d_time.md) if you have not read it already. It gives the necessary background to understand the 3D + time reconstruction tools, including this one. You can skip the sections on known non-rigid motion and deformation vector fields as they are not used in 4D FDK.

RTK provides several tools to reconstruct a 3D + time image without regularization. 4D FDK is one of them.

The algorithm requires a set of projection images with the associated RTK geometry, and the respiratory phase of each projection image. Each piece of data is described in more detail below and can be downloaded using [Girder](https://data.kitware.com/#collection/5a7706878d777f0649e04776). It is assumed that the breathing motion is periodic, which implies that the position of the chest depends on the respiratory phase only.

## Projection images

This example is illustrated with a set of projection images of the [POPI patient](https://github.com/open-vv/popi-model/blob/master/popi-model.md). You can [download the projections](https://data.kitware.com/api/v1/item/5be99af88d777f2179a2e144/download) and the required tables of the Elekta database, [FRAME.DBF](https://data.kitware.com/api/v1/item/5be99a068d777f2179a2cf4f/download) and [IMAGE.DBF](https://data.kitware.com/api/v1/item/5be99a078d777f2179a2cf65/download). See the [Elekta page](../rtkelektasynergygeometry/README.md) to find out how to produce a geometry file with them.

## Respiratory signal

The 4D FDK algorithm requires that we associate each projection image with its position in the respiratory cycle when it has been acquired.
See [this page](../rtkamsterdamshroud/README.md) to learn how to generate it from the projections using the Amsterdam Shroud.

## 4D FDK for cone-beam CT reconstruction

We now have all the pieces to perform a 3D + time reconstruction. The algorithm performs one FDK per frame, and for each frame, it uses a single projection per respiratory/cardiac cycle, and discards the rest. It might therefore be even faster than a regular 3D FDK using all projections.
```
# Reconstruct from all projection images with 4D FDK
rtkfourdfdk \
  -p . \
  -r .*.his \
  -o fourd_fdk.mha \
  -g geometry.rtk \
  --signal sphase.txt \
  --spacing 2 \
  --size 256 \
  --frames 5
```

To speed things up, it is recommended to use the CUDA version of the forward and back projectors by running this command instead:

```
# Reconstruct from all projection images with 4D FDK, using CUDA forward and back projectors
rtkfourdfdk \
  -p . \
  -r .*.his \
  -o fourd_fdk.mha \
  -g geometry.rtk \
  --signal sphase.txt \
  --hardware cuda \
  --spacing 2 \
  --size 256 \
  --frames 5
```

Note that the reconstructed volume in this example does not fully contain the attenuating object, causing hyper-attenuation artifacts at the borders of the result.
