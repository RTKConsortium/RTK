# Spectral simplex decomposition

This page describes an application for two-step spectral CT reconstruction. Please start by reading the [general documentation on spectral CT](../../documentation/docs/multi_energy.md).

RTK provides a command line application to decompose the measured projections into one set of projections per basis (material) function. It uses the incident spectrum, the material attenuations and the detector response. The method is described in [Schlomka et al, PMB, 2008](https://iopscience.iop.org/article/10.1088/0031-9155/53/15/002).

## Dataset

The data used in this documentation page can be downloaded using [Girder](https://data.kitware.com/#collection/5a7706878d777f0649e04776/folder/699fdb4af94d0dc14645a411).
It is a simulation through a cylinder of water with two inserts, one empty, the other filled with cortical bone.

## Command line application

```
# Decompose the measured projections (photon counts on the spectral detector)
# into material projections, using the incident spectrum,
# the material attenuations and the detector response.

rtkspectralsimplexdecomposition \
  -o cl_decomposed.mha \
  -i decomposed_init.mha \
  -s counts.mha \
  -d drm.mha \
  --incident spectrum.mha \
  -a mat_basis.mha \
  -t 20,40,60,80,100

```

`cl_decomposed.mha` should be identical to `decomposed.mha` available on [Girder](https://data.kitware.com/#collection/5a7706878d777f0649e04776/folder/699fdb4af94d0dc14645a411).

After decomposition, you can reconstruct the two components of the image one by one with

```
rtkfdk -p . -r '^cl_decomposed.mha$' -o fdk0.mha --component 0 -g geometry.xml -v
rtkfdk -p . -r '^cl_decomposed.mha$' -o fdk1.mha --component 1 -g geometry.xml -v
```
