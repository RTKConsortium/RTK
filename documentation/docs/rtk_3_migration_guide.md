# RTK v3 migration guide

This guide documents the changes needed to migrate from RTK 2.7.0.post1 to
RTK 3.0.

## Build dependencies and CMake configuration

RTK is compatible with ITK main. The precompiled RTK Python wheels are built
against ITK 5.4.8.

### CUDA builds with CudaCommon

RTK is compatible with the latest CudaCommon release. Update CudaCommon to
that release when migrating CUDA builds. The configuration header is now
`itkCudaCommonConfiguration.h` instead of `cudaCommonConfiguration.h`.

For local Python builds, install ITK's and CudaCommon's wrapping into the same
Python packages folder before building RTK. Use
`utilities/rtk_install_python_packages.sh` with `ITK_DIR` pointing to ITK's
build tree and `DEST_DIR` pointing to that packages folder, as described in
the [installation instructions](../../INSTALLATION.md#installing-the-python-wrapping-into-a-python-environment).

### Application builds against ITK main

ITK main contains the unreleased ITK 6 development version. When building an
RTK application against ITK main, use module interface targets and explicit
factory registration. The ITK 5 configuration used with ITK 5.4.8 remains:

```cmake
find_package(ITK REQUIRED COMPONENTS RTK)
include(${ITK_USE_FILE})
add_executable(MyReconstruction MyReconstruction.cxx)
target_link_libraries(MyReconstruction PRIVATE ${ITK_LIBRARIES})
```

For builds against ITK main, use:

```cmake
find_package(ITK REQUIRED COMPONENTS RTK ITKImageIO)
itk_generate_factory_registration()
add_executable(MyReconstruction MyReconstruction.cxx)
target_link_libraries(MyReconstruction PRIVATE
  ITK::RTKModule
  ITK::ITKImageIO
  ITK::ITKFFTImageFilterInit
)
```

Call `itk_generate_factory_registration()` before adding executables. The
ImageIO and FFT factory targets enable image readers, writers, and FFT
implementations used by reconstruction filters. Projects supporting both
ITK 5.4.8 and ITK main should select the configuration by `ITK_VERSION_MAJOR`, as in
[FirstReconstruction](../../examples/FirstReconstruction/CMakeLists.txt).

## Python application options use hyphens

Multi-word options in Python applications now use hyphens to match the C++
applications and the GNU coding standards:
`--outputevery` →`--output-every`,
`--iterationfilename` → `--iteration-file-name`,
`--hide_volume` → `--hide-volume`.

## Use `--size` instead of `--dimension`

As of RTK 3.0, the `--dimension` argument in applications is deprecated and
will be removed in a future release.  Please use the `--size` argument instead.
This update aligns RTK with upstream ITK's member name, where `--size` more
accurately reflects the expected parameter (e.g., image size) and avoids
confusion with spatial dimensions (e.g., 2D vs 3D). Note that if `--dimension`
is given, a deprecation message is shown and `--size` is ignored.

## Python: `version()` replaced by `__version__`

In the Python wrappers the previous `version()` call has been replaced by the
module attribute `__version__`.

## Forbild phantom file format only

The legacy `rtk::GeometricPhantomFileReader` and its undocumented configuration
file format have been removed starting with RTK 3.0. Everything it could do is
also achievable with the only file format for geometric phantoms accepted in
RTK 3.0, the [Forbild phantom file format](Phantom.md). The flag `--forbild`
has also been removed from command line tools as it is now the default and only
option.

The `SetIsForbildConfigFile()`, `GetIsForbildConfigFile()` and
`IsForbildConfigFileOn/Off()` methods have also been removed from
`rtk::DrawGeometricPhantomImageFilter` and
`rtk::ProjectGeometricPhantomImageFilter`. `SetConfigFile()` and the
`--phantomfile` application option remain, but they now only accept Forbild
files.

The generic `--config` command-line option has also been removed from all C++
applications, including the phantom applications; see the next section.

## Remove config option from c++ applications

The `--config` command-line option has been removed from all RTK ggo-based
applications as it was not known to be used in practice.

## Removed command-line applications

The following command-line applications have been removed and have no Python
counterparts:

- `rtkfdktwodweights`
- `rtklastdimensionl0gradientdenoising`
- `rtklut`
- `rtkparkershortscanweighting`
- `rtkramp`
- `rtkrayboxintersection`
- `rtkrayellipsoidintersection`
- `rtkrayquadricintersection`
- `rtkwangdisplaceddetectorweighting`
- `rtkwarpedbackprojectsequence`
- `rtkwarpedforwardprojectsequence`

## Removed command-line options

Two additional command-line options have been removed:

- `rtkfourdfdk`: `--lowmem` / `-l` has been removed because it was unused.
- `rtki0estimation`: `--output` / `-o` has been removed because it was not
  implemented.

## Default branch renamed from master to main

The InsightSoftwareConsortium has decided to rename its default branches from `master` to `main` (https://github.com/InsightSoftwareConsortium/ITK/issues/4732) following [GitHub's move](https://github.com/github/renaming). RTK's default branch has been renamed from `master` to `main` too. Here is what to do:

- GitHub forks: see [Matt McCormick's post](https://discourse.itk.org/t/itk-git-repository-primary-branch-name-transition-from-master-to-main/7569) to change your fork's default repository.
- Your local clones should be updated as suggested by GitHub (if `origin` is the name of https://github.com/RTKConsortium/RTK.git):

```
git branch -m master main
git fetch origin
git branch -u origin/main main
git remote set-head origin -a
```

## Python class instantiations: required explicit template arguments

When selecting a wrapped instantiation with `rtk.Filter[...]`, supply the
full template argument list. Python template indexing does not infer C++
default template arguments. For `I = itk.Image[itk.F, 3]` and
`CI = itk.CudaImage[itk.F, 3]` (in CUDA builds), update the arguments as follows:

| Class | Previous arguments | RTK 3 arguments |
| --- | --- | --- |
| `BoellaardScatterCorrectionImageFilter` | `[I]` | `[I, I]` |
| `ConjugateGradientConeBeamReconstructionFilter` | `[I]` | `[I, I, I]` |
| `CudaForwardProjectionImageFilter` | `[CI]` | `[CI, CI]` |
| `DisplacedDetectorForOffsetFieldOfViewImageFilter` | `[I]` | `[I, I]` |
| `DisplacedDetectorImageFilter` | `[I]` | `[I, I]` |
| `FDKConeBeamReconstructionFilter` | `[I]` | `[I, I, itk.F]` |
| `FDKVarianceReconstructionFilter` | `[I]` | `[I, I, itk.F]` |
| `FDKWeightProjectionFilter` | `[I]` | `[I, I]` |
| `IterativeConeBeamReconstructionFilter` | `[I]` | `[I, I]` |
| `ParkerShortScanImageFilter` | `[I]` | `[I, I]` |

For example, replace `rtk.FDKConeBeamReconstructionFilter[I].New()` with
`rtk.FDKConeBeamReconstructionFilter[I, I, itk.F].New()`.

Classes with a single template parameter, including
`ADMMTotalVariationConeBeamReconstructionFilter`, `AmsterdamShroudImageFilter`,
and `PhaseGatingImageFilter`, use `[I]`. The ADMM CUDA template change is
described in the next section.

## Template parameters removed from ADMM total variation and Laplacian filters

The unused second template parameter has been removed from several classes.
Code using the previous two-argument form must be updated:

- `ADMMTotalVariationConeBeamReconstructionFilter<TOutputImage, TGradientOutputImage>` → `ADMMTotalVariationConeBeamReconstructionFilter<TOutputImage>`
- `ADMMTotalVariationConjugateGradientOperator<TOutputImage, TGradientOutputImage>` → `ADMMTotalVariationConjugateGradientOperator<TOutputImage>`
- `LaplacianImageFilter<TOutputImage, TGradientImage>` → `LaplacianImageFilter<TOutputImage>`

`rtk::CudaLaplacianImageFilter` now derives from
`rtk::LaplacianImageFilter<itk::CudaImage<float, 3>>`. In Python,
`ADMMTotalVariationConeBeamReconstructionFilter` now takes a single image
template argument.

## Removed `m_Normalize` from `rtk::CudaRayCastBackProjectionImageFilter`

Most users intuitively expect this for the backprojector matched to `rtk::CudaForwardProjectionFilter`. The filter was originally implemented with the normalization for SART and OSEM reconstruction but the normalization is now done in the reconstruction filters so it can be removed.

## CUDA forward / back projector clipping

The CUDA forward projector `rtk::CudaForwardProjectionFilter` now only accumulates attenuation values for ray samples located in-between the source and detector. This matches the behavior of the CPU forward projector `rtk::JosephForwardProjectionImageFilter`. The same clipping behavior is also applied to the adjoint operator `rtk::rtkCudaRayCastBackProjectionImageFilter`.

## CUDA forward / ray-cast step size default

The CUDA forward projector and the CUDA ray-cast backprojector now use the minimum voxel spacing of the input volume as the ray step when StepSize is not set. The previous default value was 1.

## Image type itk::Image for incident spectrum in spectral code

Different formats were used across the spectral code for the incident spectrum, 2D itk::VectorImage and 3D itk::Image. All applications and filters now accept the two formats but handling of the `itk::VectorImage` format will be removed in the future and is marked as `ITK_FUTURE_LEGACY_REMOVE`.

## Attenuation map required for JosephAttenuated projectors

The Joseph attenuated forward and back projectors now require the attenuation
map input to be set. `rtk::JosephForwardAttenuatedProjectionImageFilter` and
`rtk::JosephBackAttenuatedProjectionImageFilter` throw an exception if input 2
is missing. In the Python applications, selecting `--fp JosephAttenuated` or
`--bp JosephAttenuated` without `--attenuationmap` now raises an error.

## FDK now requires a rotation axis close to Y

`rtk::FDKConeBeamReconstructionFilter` now throws if the rotation axis deviates
from the Y axis by more than `AngularTolerance`, which defaults to 30 degrees.
The tolerance can be changed with `SetAngularTolerance()` and
`GetAngularTolerance()`. Geometries that were previously accepted but are not
actually FDK-compatible must be corrected, or the tolerance must be increased
if the approximation is acceptable.

## Displaced detector filters now reject unsupported geometries

`rtk::DisplacedDetectorImageFilter` and
`rtk::DisplacedDetectorForOffsetFieldOfViewImageFilter` now throw when the
geometry is parallel, i.e. when the source-to-detector distance is zero.
Disable the filter with `SetDisable(true)` or `--nodisplaced` when this check is
not applicable. `rtk::DisplacedDetectorForOffsetFieldOfViewImageFilter` also
throws when manual offsets have been set with `SetOffsets()`; use
`rtk::DisplacedDetectorImageFilter` for independent projections in that case.

## rtkdualenergy applications and associated code removed

Historically, several applications had been developed to handle material decomposition of specific dual-energy spectral CT acquisitions with fast switching. `rtkdualenergysimplexdecomposition` and `rtkdualenergyforwardmodel` and the associated code in `DualEnergyNegativeLogLikelihood` and `SpectralForwardModelImageFilter` have been removed as one can do the same with `rtkspectralsimplexdecomposition` and `rtkspectralforwardmodel`.

The dual-energy-only API was also removed from
`rtk::SimplexSpectralProjectionsDecompositionImageFilter` and
`rtk::SpectralForwardModelImageFilter`:

- `SetInputSecondIncidentSpectrum()` (both overloads)
- `GetInputSecondIncidentSpectrum()`
- `SetIsSpectralCT()` and `GetIsSpectralCT()`

## rtkmcrooster and rtkmotioncompensatedfourdconjugategradient have been removed

Motion-compensated 4D reconstruction potentially avoids some interpolations during reconstruction but the benefit was too small for publication. The code was found to be unusable 10 years after its development and it was decided to remove it as it had not been used in the past 10 years.

The removed code includes the following classes and their Python wrappers:

- `rtk::MotionCompensatedFourDConjugateGradientConeBeamReconstructionFilter`
- `rtk::MotionCompensatedFourDROOSTERConeBeamReconstructionFilter`
- `rtk::MotionCompensatedFourDReconstructionConjugateGradientOperator`
- `rtk::WarpFourDToProjectionStackImageFilter`
- `rtk::WarpProjectionStackToFourDImageFilter`

## Positions now use `itk::Point` instead of `itk::Vector`

APIs that represent physical *positions* were changed from `itk::Vector<double,3>` to `itk::Point<double,3>`. This makes a clear distinction between points (locations) and vectors (directions or offsets).
Projection iterators (`rtkProjectionsRegionConstIteratorRayBased*`), Joseph forward/back projectors, ray-box filters, convex/quadric shapes, Ora geometry readers, Forbild phantom readers, and test/phantom helpers now expect `PointType` for positions (source, pixel, box corners, shape centers).

## Elekta projection cropping can now be disabled explicitly

`rtk::ProjectionsReader` retains its default crop of four pixels at each
boundary in the first two dimensions for Elekta raw projections. Other formats
default to no cropping. Previously, an explicit crop size of zero was
indistinguishable from the default and was overridden for Elekta data.

The reader now distinguishes unset crop sizes from explicit zeros. Setting
either `SetLowerBoundaryCropSize()` or `SetUpperBoundaryCropSize()` overrides
the Elekta default; an unset opposite boundary then defaults to zero. In C++
applications, use `--lowercrop 0 --uppercrop 0` to disable cropping.

Python applications using `SetProjectionsReaderFromArgParse()` always pass
explicit zero crop sizes when no crop options are supplied. They therefore
now default to **no cropping**, unlike the C++ applications. To preserve the
previous Elekta cropping in Python applications, pass:

```bash
--lowercrop 4,4,0 --uppercrop 4,4,0
```

## Common noise options in applications generating projections

Noise options are now provided similarly in all command line applications generating projections via `rtknoise_section.ggo` and the Python helper `rtknoise_group.py` (for example: `--poisson <I0>,<muref>`, `--gaussian <std>`). The legacy `--noise` option in `rtkprojectshepploganphantom` has been removed.

## `ProjectionsReader` for vector images

`ProjectionsReader` can now accept an image of vectors (i.e. a vector field), either represented by an `itk::VectorImage` or an `itk::Image<itk::Vector<>, 3>` as template output type. This change has triggered a change in its API: its member `m_VectorComponent` is now an `int` instead of an `unsigned int` and its value is `-1` by default. If its member `m_FileNames[0]` is a vector image, its template output type a scalar image and `m_VectorComponent` has not been set to a valid component, `ProjectionsReader` now triggers an exception instead of silently selecting the first component in previous versions.

## `rtkspectraldenoiseprojections` deleted

Thanks to the previous change, the command line application `rtkprojections` can now process vector images with cropping, conditional median filtering and binning options. The dedicated command line application `rtkspectraldenoiseprojections` was therefore obsolete and has been deleted.

## `rtk::AdditiveGaussianNoiseImageFilter` removed

`rtk::AdditiveGaussianNoiseImageFilter` and its `NormalVariateNoiseFunctor` have
been removed. Use `itk::AdditiveGaussianNoiseImageFilter` from ITK instead,
which provides the same `SetMean()` and `SetStandardDeviation()` API. The ITK
filter also provides `SetSeed()` and `GetSeed()` through its
`itk::NoiseBaseImageFilter` base class. Set a seed explicitly to control random
initialization; the generated noise may differ from the previous RTK filter.
In Python, replace
`rtk.AdditiveGaussianNoiseImageFilter[...]` with
`itk.AdditiveGaussianNoiseImageFilter[...]`. The RTK applications that add
Gaussian noise (`rtkdrawgeometricphantom`, `rtkdrawshepploganphantom`) already
use the ITK filter.
