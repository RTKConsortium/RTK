# Two-step spectral CT reconstruction

This page describes an application for two-step spectral CT reconstruction. Please start by reading the [general documentation on spectral CT](../../documentation/docs/multi_energy.md).

# Spectral forward model

RTK provides a command line application to simulate the measured projections in spectral CT from the decomposed projections, the incident spectrum, the material attenuations and the detector response.
It is the "forward operator" in the [inverse problem](https://en.wikipedia.org/wiki/Inverse_problem) that RTK solves to obtain the decomposed projections from the measured projections.

## Dataset

The data used in this documentation page can be downloaded using [Girder](https://data.kitware.com/#collection/5a7706878d777f0649e04776/folder/69a005bdf94d0dc14645a412).
It is a simulation through a cylinder of water with two inserts, one empty, the other filled with cortical bone.

## Command line application

```
# Simulate the measured projections (photon counts on the spectral detector)
# from the decomposed projections, the incident spectrum,
# the material attenuations and the detector response.

rtkspectralforwardmodel \
  -o cl_counts.mha \
  -i decomposed.mha \
  -d drm.mha \
  --incident no_vector_spectrum.mha \
  -a mat_basis.mha \
  -t 20,40,60,80,100
```

`cl_counts.mha` should be identical to `counts.mha`. You can check it with [VV](http://vv.creatis.insa-lyon.fr/) using the overlay feature.
