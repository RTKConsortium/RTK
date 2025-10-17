# ImageQuality

This page summarizes the existing and future solutions in RTK for improving image quality of cone-beam (CB) CT images. It is based on discussions at the RTK meeting on image quality.

## X-ray source imperfections

- **Geometric blurring** can be corrected by the scatter glare correction detailed in the detector imperfections section.
- **Exposure fluctuations** from projection to projection are common. They can be corrected by [rtk::I0EstimationProjectionFilter](https://www.openrtk.org/Doxygen/classrtk_1_1I0EstimationProjectionFilter.html), which automatically estimates a constant I0 (intensity without object) per projection using a histogram analysis. This filter only works if there are pixels in each projection that measure x-rays without object (except maybe a few projections using a recursive least-square (RLS) algorithm). The filter does not have any parameter except the bitshift template value for the reduction of the histogram size. It is implemented for integer pixel types only.
- **Focal spot motion** cannot be corrected currently. It would require geometric calibration for each acquired projection using auto calibration.


## Detector imperfections

- **Variations of the flat field image and the dark field image** are known (due to temperature changes or ghosting, see, e.g., [Siewerdsen and Jaffray, Med Phys, 1999]). Acquisition of these two images before and after the acquisition is the best solution when possible. There is no other solution in RTK except for the automatic detection of the constant I0 value (see fluctuations of the source exposure).
- **Lag** corresponds to a short-term temporal effect of the detector (a few projections). The [thesis of Starman](https://stacks.stanford.edu/file/druid:dj434tf8306/Starman_Jared_thesis_withTitlePage-augmented.pdf) (2010) gives a good overview of the problem and solutions he proposed. [rtk::LagCorrectionImageFilter](https://www.openrtk.org/Doxygen/classrtk_1_1LagCorrectionImageFilter.html) implements equation 2.1 of his PhD thesis. The `a` and `b` parameters must be calibrated for a given system; some RTK users have done this and could share their scripts upon request via the RTK user mailing list. A CUDA version of this filter is being implemented in Louvain-La-Neuve (Belgium). The solution in chapter 3 of Starman's thesis might also be investigated in future works.
- **Scatter glare** is the point spread function (PSF) of the detector. The solution described in [Poludniowski et al, PMB, 2011] is implemented in [rtk::ScatterGlareCorrectionImageFilter](https://www.openrtk.org/Doxygen/classrtk_1_1ScatterGlareCorrectionImageFilter.html). The a and b parameters must be calibrated for a given system; some RTK users have done this and could share their scripts upon request via the RTK user mailing list.


## Beam hardening

- The acquired data may be linearized for a given material using [rtk::LookupTableImageFilter](https://www.openrtk.org/Doxygen/classrtk_1_1LookupTableImageFilter.html) as explained, e.g., in Fig. 1 of [Brooks and Di Chiro, PMB, 1976]. There are several solutions to compute the lookup table:
    - Compute it from the knowledge of the spectrum of the x-ray source and the detector response,
    - Measure the attenuation for several thicknesses of the material of interest,
    - Do a tomography of a homogeneous object (e.g., a cylinder) and find the lookup table using the [rtk::WaterPrecorrectionImageFilter](https://www.openrtk.org/Doxygen/classrtk_1_1WaterPrecorrectionImageFilter.html) which implements [Kachelriess et al, Med Phys, 2006].
    - Estimate the p and q parameters of Equation 1 in [Ohnesorge et al, Eur Radiol, 1999], which comes down to a two-parameter beam-hardening correction and convert it to a lookup table.
- The algorithm of [Kyriakou et al, Med Phys, 2010] may easily be implemented from the existing code in RTK.


## Scatter

- The patient may be approximated by a constant per projection in a first-order approximation. This is done using [Boellaard et al, Med Phys, 1997] in [rtk::BoellaardScatterCorrectionImageFilter](https://www.openrtk.org/Doxygen/classrtk_1_1BoellaardScatterCorrectionImageFilter.html).
- The [Ohnesorge et al, Eur Radiol, 1999] algorithm is being implemented in Louvain-La-Neuve (Belgium).
- The adaptive scatter kernel superposition [Sun and Star-Lack, PMB, 2010] will be implemented in Louvain-La-Neuve (Belgium).
- Two solutions using a prior scatter-free CT have been investigated by RTK users:
    - [Park et al, Med Phys, 2015] have implemented the algorithm of [Niu et al, Med Phys, 2010]. This solution will be studied in Lyon (France) as well.
    - Monte Carlo based scatter correction is investigated in Lyon (France).


## Statistical noise

- RTK has a fast 2D median filter for projection images for a few kernel dimensions, see [rtk::ConditionalMedianImageFilter](https://www.openrtk.org/Doxygen/classrtk_1_1ConditionalMedianImageFilter.html). A GPU version of the median filter will be developed in Salzburg (Austria).
- Median filters do not preserve edges (see [Arias-Castro and Donoho, Annals of Statistics, 2009]). A multi-pass median filter is required, which might be investigated in Louvain-La-Neuve (Belgium) in the future.
- The [Savitzky–Golay filter](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter) is a promising solution that will be investigated in Louvain-La-Neuve (Belgium) in the future. This solution also provides derivatives of the image.


## Truncated projection images

- The [rtk::FFTRampImageFilter](http://www.openrtk.org/Doxygen/classrtk_1_1FFTRampImageFilter.html) implements the heuristic solution of [Ohnesorge et al, Med Phys, 2000]. The parameter `TruncationCorrection` must be adjusted.
- Exact reconstruction based on differentiated backprojection and inverse Hilbert filtering (see, e.g., [Noo et al, PMB, 2004]) is investigated in Lyon (France).


## Geometric calibration

- RTK allows the description of the CBCT geometry using all 9 degrees-of-freedom (DOF).
- The 9 DOF are not accounted for in all filters, e.g., the ramp filter does not account for the in-plane rotation.
- Several users have developed 9 DOF calibration.