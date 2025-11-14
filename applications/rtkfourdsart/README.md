# 4D SART

Please start by reading the [main documentation page on 3D + time reconstruction](../../documentation/docs/3d_time.md) if you have not read it already. It gives the necessary background to understand the 3D + time reconstruction tools, including this one. You can skip the sections on known non-rigid motion and deformation vector fields as they are not used in 4D SART.

RTK provides several tools to reconstruct a 3D + time image without regularization. 4D SART is one of them. It is similar to [4D conjugate gradient](../rtkfourdconjugategradient/README.md) in the sense that it minimizes the same objective function, but with a different method.

The algorithm requires a set of projection images with the associated RTK geometry, and the respiratory phase of each projection image. Each piece of data is described in more detail below and can be downloaded using [Girder](https://data.kitware.com/#collection/5a7706878d777f0649e04776). It is assumed that the breathing motion is periodic, which implies that the position of the chest depends on the respiratory phase only.

## Projection images

This example is illustrated with a set of projection images of the [POPI patient](https://github.com/open-vv/popi-model/blob/master/popi-model.md). You can [download the projections](https://data.kitware.com/api/v1/item/5be99af88d777f2179a2e144/download) and the required tables of the Elekta database, [FRAME.DBF](https://data.kitware.com/api/v1/item/5be99a068d777f2179a2cf4f/download) and [IMAGE.DBF](https://data.kitware.com/api/v1/item/5be99a078d777f2179a2cf65/download). See the [Elekta page](../rtkelektasynergygeometry/README.md) to find out how to produce a geometry file with them.

## Respiratory signal

The 4D SART algorithm requires that we associate each projection image with its position in the respiratory cycle when it has been acquired.
See [this page](../rtkamsterdamshroud/README.md) to learn how to generate it from the projections using the Amsterdam Shroud.

## 4D SART for cone-beam CT reconstruction

We now have all the pieces to perform a 3D + time reconstruction. The algorithm will perform `niterations` iterations. 4D SART forward projects `nprojpersubset` projections, back projects them and updates the volume, then moves on to the next subset of projections. In RTK iterative reconstruction algorithms, we count one iteration when the full set of projections has been forward and back projected. 4D SART therefore performs several updates of the reconstructed volume in a single iteration.
```
# Reconstruct from all projection images with 4D SART
rtkfourdsart \
  -p . \
  -r .*.his \
  -o fourd_sart.mha \
  -g geometry.rtk \
  --signal sphase.txt \
  --niterations 5 \
  --nprojpersubset 10 \
  --spacing 2 \
  --size 160 \
  --frames 5
```

To speed things up, it is recommended to use the CUDA version of the forward and back projectors by running this command instead:

```
# Reconstruct from all projection images with 4D SART, using CUDA forward and back projectors
rtkfourdsart \
  -p . \
  -r .*.his \
  -o fourd_sart.mha \
  -g geometry.rtk \
  --signal sphase.txt \
  --fp CudaRayCast \
  --bp CudaVoxelBased \
  --niterations 5 \
  --nprojpersubset 10 \
  --spacing 2 \
  --size 160 \
  --frames 5
```

Note that the reconstructed volume in this example does not fully contain the attenuating object, causing hyper-attenuation artifacts at the borders of the result. To avoid these artifacts, reconstructing a larger volume with `--size 256` should help.
