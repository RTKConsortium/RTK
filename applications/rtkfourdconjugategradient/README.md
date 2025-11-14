# 4D conjugate gradient

Please start by reading the [main documentation page on 3D + time reconstruction](../../documentation/docs/3d_time.md) if you have not read it already. It gives the necessary background to understand the 3D + time reconstruction tools, including this one. You can skip the sections on known non-rigid motion and deformation vector fields as they are not used in 4D conjugate gradient.

RTK provides several tools to reconstruct a 3D + time image without regularization. 4D conjugate gradient is one of them.

The algorithm requires a set of projection images with the associated RTK geometry, and the respiratory phase of each projection image. Each piece of data is described in more detail below and can be downloaded using [Girder](https://data.kitware.com/#collection/5a7706878d777f0649e04776). It is assumed that the breathing motion is periodic, which implies that the position of the chest depends on the respiratory phase only.

## Projection images

This example is illustrated with a set of projection images of the [POPI patient](https://github.com/open-vv/popi-model/blob/master/popi-model.md). You can [download the projections](https://data.kitware.com/api/v1/item/5be99af88d777f2179a2e144/download) and the required tables of the Elekta database, [FRAME.DBF](https://data.kitware.com/api/v1/item/5be99a068d777f2179a2cf4f/download) and [IMAGE.DBF](https://data.kitware.com/api/v1/item/5be99a078d777f2179a2cf65/download). See the [Elekta page](../rtkelektasynergygeometry/README.md) to find out how to produce a geometry file with them.

## Respiratory signal

The 4D conjugate gradient algorithm requires that we associate each projection image with its position in the respiratory cycle when it has been acquired.
See [this page](../rtkamsterdamshroud/README.md) to learn how to generate it from the projections using the Amsterdam Shroud.

## 4D conjugate gradient for cone-beam CT reconstruction

We now have all the pieces to perform a 3D + time reconstruction. The algorithm will perform `niterations` iterations, each of which requires a forward and a back projection of the full set of projections.
```
# Reconstruct from all projection images with 4D conjugate gradient
rtkfourdconjugategradient \
  -p . \
  -r .*.his \
  -o fourd_cg.mha \
  -g geometry.rtk \
  --signal sphase.txt \
  --niterations 30 \
  --spacing 2 \
  --size 160 \
  --frames 5
```

Compared to FDK, which only performs a single back projection, it can therefore be expected to take approximately 2 x niterations longer. To speed things up, it is recommended to use the CUDA version of the forward and back projectors by running this command instead:

```
# Reconstruct from all projection images with 4D conjugate gradient, using CUDA forward and back projectors
rtkfourdconjugategradient \
  -p . \
  -r .*.his \
  -o fourd_cg.mha \
  -g geometry.rtk \
  --signal sphase.txt \
  --fp CudaRayCast \
  --bp CudaVoxelBased \
  --niterations 30 \
  --spacing 2 \
  --size 160 \
  --frames 5
```

Note that the reconstructed volume in this example does not fully contain the attenuating object, causing hyper-attenuation artifacts at the borders of the result. To avoid these artifacts, reconstructing a larger volume with `--size 256` should help.
