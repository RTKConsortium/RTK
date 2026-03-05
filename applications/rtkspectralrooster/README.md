# Two-step spectral CT reconstruction

This page describes an application for two-step spectral CT reconstruction. Please start by reading the [general documentation on spectral CT](../../documentation/docs/multi_energy.md).

Once measured projections have been decomposed into material projections, any reconstruction algorithm can be used to reconstruct each material. In RTK, simply splitting the material projections file (saved as an `itkVectorImage`) into one set of projections per material, and then using `rtkfdk`, `rtksart` or `rtkconjugategradient` will work.

# Spectral ROOSTER

RTK provides a command line application to jointly reconstruct all materials at once, allowing to apply spatial regularization in each material, but also regularization along the material dimension, as reconstructed material volumes sometimes share the same spatial structure.
Since `rtkspectralrooster` uses the [ROOSTER filter](https://www.openrtk.org/Doxygen/classrtk_1_1FourDROOSTERConeBeamReconstructionFilter.html), originally designed to reconstruct 3D+time volume sequences, the material dimension is referred to as time in the command-line help and parameter names.

## Dataset

The data used in this documentation page can be downloaded using [Girder](https://data.kitware.com/#collection/5a7706878d777f0649e04776/folder/69a005bdf94d0dc14645a412).
It is a simulation through a cylinder of water with two inserts, one empty, the other filled with cortical bone.

## Command line application

```
# Reconstruct a set of material volumes from all material projections,
# applying Total Nuclear Variation regularization

rtkspectralrooster \
  -o cl_spectralrooster.mha \
  -i mat_recon.mha \
  -p decomposed.mha \
  -g geometry.xml \
  --niter 4 \
  --cgiter 2 \
  --tviter 2 \
  --gamma_tnv 1
```

`cl_spectralrooster.mha` should look similar to `recon_ref.mha`. You can check it with [VV](http://vv.creatis.insa-lyon.fr/) using the overlay feature. Note that `recon_ref.mha` is a 4D image, while `cl_spectralrooster.mha` is a VectorImage, so their data is not organized the same way on disk, but VV will open both a 4D images and overlay them correctly.
To speed things up, it is recommended to use the CUDA version of the forward and back projectors, and by using CUDA in the conjugate gradient descent algorithm, by running this command instead:

```
# Reconstruct a set of material volumes from all material projections,
# applying Total Nuclear Variation regularization, using CUDA accelerations

rtkspectralrooster \
  -o cl_spectralrooster.mha \
  -i mat_recon.mha \
  -p decomposed.mha \
  -g geometry.xml \
  --niter 4 \
  --cgiter 2 \
  --tviter 2 \
  --gamma_tnv 1
  --cudacg \
  --fp CudaRayCast \
  --bp CudaVoxelBased
```
