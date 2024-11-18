# Spectral forward model

This page describes an application for spectral CT simulation. Please start by reading the [general documentation on spectral CT](../../documentation/docs/multi_energy.md).

RTK provides a command line application to simulate the measured projections in spectral CT from the decomposed projections, the incident spectrum, the material attenuations and the detector response. It is the "forward operator" in the [inverse problem](https://en.wikipedia.org/wiki/Inverse_problem) that RTK solves to obtain the decomposed projections from the measured projections.

## Dataset

The data used in this documentation page can be downloaded using [Girder](https://data.kitware.com/#collection/5a7706878d777f0649e04776/folder/699fdb4af94d0dc14645a411).
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
  --incident spectrum.mha \
  -a mat_basis.mha \
  -t 20,40,60,80,100
```

`cl_counts.mha` should be identical to `counts.mha` available on [Girder](https://data.kitware.com/#collection/5a7706878d777f0649e04776/folder/699fdb4af94d0dc14645a411).
