# Two-step spectral CT reconstruction

This page describes an application for two-step spectral CT reconstruction. Please start by reading the [general documentation on spectral CT](../../documentation/docs/multi_energy.md).

# Spectral simplex decomposition

RTK provides a command line application to decompose the measured projections into one set of projections per material assumed compose the imaged object. It uses the incident spectrum, the material attenuations and the detector response.
The method is described in [this paper](https://iopscience.iop.org/article/10.1088/0031-9155/53/15/002).

## Dataset

The data used in this documentation page can be downloaded using [Girder](https://data.kitware.com/#collection/5a7706878d777f0649e04776/folder/69a005bdf94d0dc14645a412).
It is a simulation through a cylinder of water with two inserts, one empty, the other filled with cortical bone.

## Command line application

```
# Decompose the measured projections (photon counts on the spectral detector)
# into material projections, using the incident spectrum,
# the material attenuations and the detector response.

rtkspectralsimplexdecomposition \
  -o cl_decomposed.mha \
  -i zero_decomposed.mha \
  -s counts.mha \
  -d drm.mha \
  --incident no_vector_spectrum.mha \
  -a mat_basis.mha \
  -t 20,40,60,80,100

```

`cl_decomposed.mha` should be identical to `decomposed.mha`. You can check it with [VV](http://vv.creatis.insa-lyon.fr/) using the overlay feature.
