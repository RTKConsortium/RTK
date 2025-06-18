#include "rtkTest.h"
#include "rtkConfiguration.h"
#include "rtkMacro.h"
#include "rtkSimplexSpectralProjectionsDecompositionImageFilter.h"
#include "rtkSpectralForwardModelImageFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkRayEllipsoidIntersectionImageFilter.h"
#include <itkImageFileReader.h>
#include <itkCastImageFilter.h>

/**
 * \file rtkdecomposespectralprojectionstest.cxx
 *
 * \brief Functional test for the filters performing spectral forward model and spectral projections' material
 * decomposition
 *
 * This test generates analytical projections of a small phantom made of cylinder of water,
 * iodine and gadolinium, computes the expected photon counts in each detector bin in the
 * noiseless case, and performs a material decomposition on the photon counts to recover
 * the analytical projections.
 *
 * \author Cyril Mory
 */

int
main(int argc, char * argv[])
{
  if (argc < 4)
  {
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << "  incident_spectrum  detector_response  material_attenuations" << std::endl;
    return EXIT_FAILURE;
  }

  using PixelValueType = float;
  constexpr unsigned int Dimension = 3;
  using OutputImageType = itk::Image<PixelValueType, Dimension>;

  using DecomposedProjectionsType = itk::VectorImage<PixelValueType, Dimension>;
  using MeasuredProjectionsType = itk::VectorImage<PixelValueType, Dimension>;

  using IncidentSpectrumImageType = itk::Image<PixelValueType, Dimension>;
  using IncidentSpectrumReaderType = itk::ImageFileReader<IncidentSpectrumImageType>;

  using DetectorResponseImageType = itk::Image<PixelValueType, Dimension - 1>;
  using DetectorResponseReaderType = itk::ImageFileReader<DetectorResponseImageType>;

  using MaterialAttenuationsImageType = itk::Image<PixelValueType, Dimension - 1>;
  using MaterialAttenuationsReaderType = itk::ImageFileReader<MaterialAttenuationsImageType>;

  // Cast filters to convert between vector image types
  using CastDecomposedProjectionsFilterType =
    itk::CastImageFilter<DecomposedProjectionsType, itk::Image<itk::Vector<PixelValueType, 3>, Dimension>>;
  using CastMeasuredProjectionFilterType =
    itk::CastImageFilter<MeasuredProjectionsType, itk::Image<itk::Vector<PixelValueType, 6>, Dimension>>;

  // Read all inputs
  IncidentSpectrumReaderType::Pointer incidentSpectrumReader = IncidentSpectrumReaderType::New();
  incidentSpectrumReader->SetFileName(argv[1]);
  incidentSpectrumReader->Update();

  DetectorResponseReaderType::Pointer detectorResponseReader = DetectorResponseReaderType::New();
  detectorResponseReader->SetFileName(argv[2]);
  detectorResponseReader->Update();

  MaterialAttenuationsReaderType::Pointer materialAttenuationsReader = MaterialAttenuationsReaderType::New();
  materialAttenuationsReader->SetFileName(argv[3]);
  materialAttenuationsReader->Update();

#if FAST_TESTS_NO_CHECKS
  constexpr unsigned int NumberOfProjectionImages = 1;
#else
  constexpr unsigned int NumberOfProjectionImages = 64;
#endif

  // Constant image source for the analytical projections calculation
  using ConstantImageSourceType = rtk::ConstantImageSource<OutputImageType>;

  ConstantImageSourceType::Pointer projectionsSource = ConstantImageSourceType::New();
  auto                             origin = itk::MakePoint(-255., -0.5, -255.);
#if FAST_TESTS_NO_CHECKS
  auto spacing = itk::MakeVector(504., 504., 504.);
  auto size = itk::MakeSize(2, 1, NumberOfProjectionImages);
#else
  auto spacing = itk::MakeVector(8., 1., 1.);
  auto size = itk::MakeSize(64, 1, NumberOfProjectionImages);
#endif
  projectionsSource->SetOrigin(origin);
  projectionsSource->SetSpacing(spacing);
  projectionsSource->SetSize(size);
  projectionsSource->SetConstant(0.);

  // Initialize the multi-materials projections
  DecomposedProjectionsType::Pointer decomposed = DecomposedProjectionsType::New();
  decomposed->SetVectorLength(3);
  decomposed->SetOrigin(origin);
  decomposed->SetSpacing(spacing);
  DecomposedProjectionsType::RegionType region;
  DecomposedProjectionsType::IndexType  index;
  index.Fill(0);
  region.SetSize(size);
  region.SetIndex(index);
  decomposed->SetRegions(region);
  decomposed->Allocate();

  // Geometry object
  using GeometryType = rtk::ThreeDCircularProjectionGeometry;
  GeometryType::Pointer geometry = GeometryType::New();
  for (unsigned int noProj = 0; noProj < NumberOfProjectionImages; noProj++)
    geometry->AddProjection(600., 1200., noProj * 360. / NumberOfProjectionImages);

  // Generate 3 phantoms, one per material
  using REIType = rtk::RayEllipsoidIntersectionImageFilter<OutputImageType, OutputImageType>;
  REIType::Pointer rei;
  rei = REIType::New();
  for (unsigned int material = 0; material < 3; material++)
  {
    auto semiprincipalaxis = itk::MakeVector(10., 10., 10.);
    auto center = itk::MakeVector(15., 0., 15.);
    rei->SetAngle(0.);
    if (material == 2) // water
      rei->SetDensity(1.);
    else // iodine and gadolinium
      rei->SetDensity(0.01);
    rei->SetCenter(center);
    rei->SetAxis(semiprincipalaxis);

    // Compute analytical projections through them
    rei->SetInput(projectionsSource->GetOutput());
    rei->SetGeometry(geometry);
    TRY_AND_EXIT_ON_ITK_EXCEPTION(rei->Update());

    // Merge these projections into the multi-material projections image
    itk::ImageRegionConstIterator<OutputImageType> inIt(rei->GetOutput(), rei->GetOutput()->GetLargestPossibleRegion());
    itk::ImageRegionIterator<DecomposedProjectionsType> outIt(decomposed, decomposed->GetLargestPossibleRegion());
    outIt.GoToBegin();
    while (!outIt.IsAtEnd())
    {
      itk::VariableLengthVector<PixelValueType> vector = outIt.Get();
      vector[material] = inIt.Get();
      outIt.Set(vector);
      ++inIt;
      ++outIt;
    }
  }

  // Generate a set of zero-filled photon count projections
  MeasuredProjectionsType::Pointer measuredProjections = MeasuredProjectionsType::New();
  measuredProjections->CopyInformation(decomposed);
  measuredProjections->SetVectorLength(6);
  measuredProjections->SetRegions(region);
  measuredProjections->Allocate();

  // Generate the thresholds vector
  itk::VariableLengthVector<unsigned int> thresholds;
  thresholds.SetSize(7);
  thresholds[0] = 25;
  thresholds[1] = 40;
  thresholds[2] = 55;
  thresholds[3] = 70;
  thresholds[4] = 85;
  thresholds[5] = 100;
  thresholds[6] = 180;

  // Apply the forward model to the multi-material projections
  using SpectralForwardFilterType =
    rtk::SpectralForwardModelImageFilter<DecomposedProjectionsType, MeasuredProjectionsType, IncidentSpectrumImageType>;
  SpectralForwardFilterType::Pointer forward = SpectralForwardFilterType::New();
  forward->SetInputDecomposedProjections(decomposed);
  forward->SetInputMeasuredProjections(measuredProjections);
  forward->SetInputIncidentSpectrum(incidentSpectrumReader->GetOutput());
  forward->SetDetectorResponse(detectorResponseReader->GetOutput());
  forward->SetMaterialAttenuations(materialAttenuationsReader->GetOutput());
  forward->SetThresholds(thresholds);
  forward->SetIsSpectralCT(true);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(forward->Update())

  // Generate a set of decomposed projections as input for the simplex
  DecomposedProjectionsType::Pointer initialDecomposedProjections = DecomposedProjectionsType::New();
  initialDecomposedProjections->CopyInformation(decomposed);
  initialDecomposedProjections->SetRegions(region);
  initialDecomposedProjections->SetVectorLength(3);
  initialDecomposedProjections->Allocate();
  DecomposedProjectionsType::PixelType initPixel;
  initPixel.SetSize(3);
  initPixel[0] = 0.1;
  initPixel[1] = 0.1;
  initPixel[2] = 10;
  initialDecomposedProjections->FillBuffer(initPixel);

  // Create and set the simplex filter to perform the decomposition
  using SimplexFilterType = rtk::SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionsType,
                                                                                    MeasuredProjectionsType,
                                                                                    IncidentSpectrumImageType>;
  SimplexFilterType::Pointer simplex = SimplexFilterType::New();
  simplex->SetInputDecomposedProjections(initialDecomposedProjections);
  simplex->SetInputMeasuredProjections(forward->GetOutput());
  simplex->SetInputIncidentSpectrum(incidentSpectrumReader->GetOutput());
  simplex->SetDetectorResponse(detectorResponseReader->GetOutput());
  simplex->SetMaterialAttenuations(materialAttenuationsReader->GetOutput());
  simplex->SetThresholds(thresholds);
  simplex->SetNumberOfIterations(10000);
  forward->SetIsSpectralCT(true);

  std::cout << "\n\n****** Case 1: User-provided initial values ******" << std::endl;

  TRY_AND_EXIT_ON_ITK_EXCEPTION(simplex->Update())
  CheckVectorImageQuality<DecomposedProjectionsType>(simplex->GetOutput(), decomposed, 0.0001, 15, 2.0);

  std::cout << "\n\n****** Case 2: Heuristically-determined initial values ******" << std::endl;

  simplex->SetGuessInitialization(true);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(simplex->Update())
  CheckVectorImageQuality<DecomposedProjectionsType>(simplex->GetOutput(), decomposed, 0.0001, 15, 2.0);

  std::cout << "\n\n****** Case 3: Fixed-length vector image inputs ******" << std::endl;

  // measuredProjections has been consumed by forward, which is InPlace. Reallocate it
  measuredProjections->SetRegions(region);
  measuredProjections->Allocate();

  typename CastDecomposedProjectionsFilterType::Pointer castDecomposedProjections =
    CastDecomposedProjectionsFilterType::New();
  typename CastMeasuredProjectionFilterType::Pointer castMeasuredProjections = CastMeasuredProjectionFilterType::New();
  castDecomposedProjections->SetInput(decomposed);
  castMeasuredProjections->SetInput(measuredProjections);
  forward->SetInputDecomposedProjections(castDecomposedProjections->GetOutput());
  forward->SetInputMeasuredProjections(castMeasuredProjections->GetOutput());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(forward->Update())

  typename CastDecomposedProjectionsFilterType::Pointer castDecomposedProjections2 =
    CastDecomposedProjectionsFilterType::New();
  typename CastMeasuredProjectionFilterType::Pointer castMeasuredProjections2 = CastMeasuredProjectionFilterType::New();
  castDecomposedProjections->SetInput(initialDecomposedProjections);
  castMeasuredProjections->SetInput(forward->GetOutput());
  simplex->SetInputDecomposedProjections(castDecomposedProjections->GetOutput());
  simplex->SetInputMeasuredProjections(castMeasuredProjections->GetOutput());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(simplex->Update())

  CheckVectorImageQuality<DecomposedProjectionsType>(simplex->GetOutput(), decomposed, 0.0001, 15, 2.0);

#ifndef ITK_FUTURE_LEGACY_REMOVE
  std::cout << "\n\n****** Case 4: Legacy VectorImage type for incident spectrum ******" << std::endl;

  using VectorImageType = itk::VectorImage<PixelValueType, Dimension - 1>;
  using VectorSpectrumReaderType = itk::ImageFileReader<VectorImageType>;
  VectorSpectrumReaderType::Pointer vectorSpectrumReader = VectorSpectrumReaderType::New();
  vectorSpectrumReader->SetFileName(argv[4]);
  vectorSpectrumReader->Update();
  forward->SetInputIncidentSpectrum(vectorSpectrumReader->GetOutput());
  simplex->SetInputIncidentSpectrum(vectorSpectrumReader->GetOutput());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(simplex->Update())
  CheckVectorImageQuality<DecomposedProjectionsType>(simplex->GetOutput(), decomposed, 0.0001, 15, 2.0);
#endif

  std::cout << "\n\nTest PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
