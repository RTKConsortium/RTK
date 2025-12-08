# Spectral CT reconstruction

RTK can reconstruct from spectral CT acquisitions, either by a one-step or a two-steps method. Both methods support fast-switching.
This page provides a reminder of the concepts underlying spectral tomography, and describes the tools available in RTK.

## Spectral CT specific concepts

### Spectral CT acquisitions

Spectral acquisitions require a specific detector, able to measure the energy of each incoming photon, and increment a counter in the corresponding energy band, called a "bin".
In regular CT, projections contain one value per pixel. In spectral CT, projections contain as many values per pixel as the number of bins.

Fast-switching means changing the X-Ray tube settings very fast during the acquisition, typically to alternate between a low and and high voltage from one projection to the next. Low and high kV projections are therefore intertwined, which maximizes their spatial and temporal consistency.

Spectral and fast-switching approaches can be combined, as one only requires a specific detector and the other only requires a modification of the tube settings.

### Material decomposition
Regular CT reconstructs a volume of the X-ray attenuation coefficients of the imaged object.
Spectral reconstruction assumes the imaged object is made of a few materials (typically three) and reconstructs one volume per material: each volume is the density map of that material in the imaged object. This means that dual/spectral projections must be decomposed into material projections and then reconstructed.
This can be performed in two separate steps (two-steps method), or merged into a single, larger reconstruction problem (one-step method).

## Spectral CT specific input data

Multi-energy reconstruction requires specific input data:
- The measured projections have "number of bins" values per pixel
- The reconstructed volume have "number of materials" values per voxel
- The incident spectrum contains the number of photons emitted at each energy, towards each detector pixel
- The detector response contains the probabilities, for a photon of energy E, to be measured as having energy U
- The material attenuations describe, for each material assumed to compose the imaged object, its linear attenuation coefficient at each energy
- The thresholds are the lower energy bounds of bins
The format of each of these inputs is detailed below.

### VectorImage and Image of Vectors
Instead of adding a dimension, it was decided to handle the multi-energy aspect of spectral CT by using the vector types in ITK.

### Measured projections
Measured projections are either itk::Image<itk::Vector<float, nbins>, 3> (nbins known at compile-time) or itk::VectorImage<float, 3> (nbins determined at runtime). The filters and applications accept both formats.

### Reconstructed volume
The reconstructed volume is either itk::Image<itk::Vector<float, nbins>, 3> or itk::VectorImage<float, 3>. The filters and applications accept both formats.

### Incident spectrum
Incident spectrum is an itk::Image<float, 3>. The first dimension is the energy of the emitted photons in keV, the next two dimensions are the two dimensions of the detector.
In one-step, the incident spectrum is looped upon: the iterator returns to the beginning of the spectrum data when it reaches its end, usually after each projection, in which case the same incident spectrum is used for all projections.
But the size of the last dimension, i.e. the number of rows in the spectrum, may be an integer multiple of the number of rows in projections. A single itk::Image<float, 3> image spectrum can therefore contain several different spectra, stacked vertically.
With two spectra, this means odd projections will use one spectrum and even ones will use the other spectrum, effectively handling fast-switching.

### Detector response
Due to imperfect electronics in the detector, the energy it measures is not always the actual energy of the incoming photon. The detector response matrix models that.
It is passed to the spectral reconstruction applications as an itk::Image<float, 2>. The first dimension is the actual energy of an incident photon, the second dimension is the measured energy, and the pixel values are the probabilities of measurements.
In the one-step application, it is binned according to the thresholds, and passed to the [one-step reconstruction filter](https://www.openrtk.org/Doxygen/classrtk_1_1MechlemOneStepSpectralReconstructionFilter.html) as a vnl_matrix of size (nbins, nenergies).

### Material attenuations
Material attenuations is an itk::Image<float, 2>. The first dimension is the materials, the second is the energy, and the pixel values are the linear attenuation coefficients of the materials at each energy.

### Thresholds
Spectral detectors compare the measured energy to a set of pre-defined values, called thresholds. Each threshold is the lower bound of an energy bin. If the thresholds are (20;40;60;80;100), a photon with measured energy 45keV will pass the threshold checks for 20 and 40keV, and thus be counted in bin `[40;60]`. The last bin has no upper bound.
Thresholds are passed to the spectral reconstruction applications as a command-line argument, like spacing or size.
