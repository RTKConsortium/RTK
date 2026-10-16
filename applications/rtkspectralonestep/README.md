# One-step spectral CT reconstruction

Please start by reading the [general documentation on spectral CT](../../documentation/docs/multi_energy.md).

RTK provides a command line application to perform one-step reconstruction. It uses the measured projections, the incident spectrum, the material attenuations and the detector response for the spectral part, and the geometry and an initial reconstructed volume for the tomographic part. Providing a mask of the region of the reconstructed volume where the image object is expected to be found, and outside of which there should be nothing, helps the algorithm converge. The method is described in [this paper](https://ieeexplore.ieee.org/document/7979598).

## Dataset

The data used in this documentation page can be downloaded using [Girder](https://data.kitware.com/#collection/5a7706878d777f0649e04776/folder/699fdb4af94d0dc14645a411).
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
  -i zero_recon.mha \
  -o cl_onestep.mha \
  -t 20,40,60,80,100 \
  --mask mask.mha \
  -n 7
```

`cl_onestep.mha` should look similar to `recon_ref.mha` available on [Girder](https://data.kitware.com/#collection/5a7706878d777f0649e04776/folder/699fdb4af94d0dc14645a411).

To speed things up, it is recommended to use the CUDA version of the forward and back projectors by running this command instead:

```
# One-step spectral reconstruction using CUDA accelerations

rtkspectralonestep \
  -s counts.mha \
  --incident spectrum.mha \
  -a mat_basis.mha \
  -d drm.mha \
  -g geometry.xml \
  -i zero_recon.mha \
  -o cl_onestep.mha \
  -t 20,40,60,80,100 \
  --mask mask.mha \
  -n 7 \
  -f CudaRayCast \
  -b CudaVoxelBased
```
