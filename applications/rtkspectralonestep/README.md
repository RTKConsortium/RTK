# One-step spectral CT reconstruction

Please start by reading the [general documentation on spectral CT](../../documentation/docs/multi_energy.md).

RTK provides a command line application to perform one-step reconstruction. It uses the measured projections, the incident spectrum, the material attenuations and the detector response for the spectral part, and the geometry and an initial reconstructed volume for the tomographic part.
Providing a mask of the region of the reconstructed volume where the image object is expected to be found, and outside of which there should be nothing, helps the algorithm converge.
The method is described in [this paper](https://ieeexplore.ieee.org/document/7979598).

## Dataset

The data used in this documentation page can be downloaded using [Girder](https://data.kitware.com/#collection/5a7706878d777f0649e04776/folder/69a005bdf94d0dc14645a412).
It is a simulation through a cylinder of water with two inserts, one empty, the other filled with cortical bone.

## Command line application

```
# One-step spectral reconstruction

rtkspectralonestep \
  -s counts.mha \
  --incident spectrum.mha \
  -a mat_basis.mha \
  -d drm.mha \
  -g geometry.xml \
  -i decomposed_init.mha \
  -o cl_onestep.mha \
  -t 20,40,60,80,100 \
  --mask mask.mha \
  -n 20
```

`cl_onestep.mha` should look similar to `recon_ref.mha`. You can check it with [VV](http://vv.creatis.insa-lyon.fr/) using the overlay feature. Note that `recon_ref.mha` is a 4D image, while `cl_onestep.mha` is a VectorImage, so their data is not organized the same way on disk, but VV will open both a 4D images and overlay them correctly.

To speed things up, it is recommended to use the CUDA version of the forward and back projectors by running this command instead:

```
# One-step spectral reconstruction using CUDA accelerations

rtkspectralonestep \
  -s counts.mha \
  --incident spectrum.mha \
  -a mat_basis.mha \
  -d drm.mha \
  -g geometry.xml \
  -i decomposed_init.mha \
  -o cl_onestep.mha \
  -t 20,40,60,80,100 \
  --mask mask.mha \
  -n 20 \
  -f CudaRayCast \
  -b CudaVoxelBased
```
