#include "rtkTest.h"
#include "rtkConfiguration.h"
#include "rtkMacro.h"
#include "rtkSimplexSpectralProjectionsDecompositionImageFilter.h"
#include "rtkSpectralForwardModelImageFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkRayEllipsoidIntersectionImageFilter.h"
#include <itkImageFileReader.h>

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

  using DecomposedProjectionType = itk::VectorImage<PixelValueType, Dimension>;

  using MeasuredProjectionsType = itk::VectorImage<PixelValueType, Dimension>;

  using IncidentSpectrumImageType = itk::Image<PixelValueType, Dimension>;
  using IncidentSpectrumReaderType = itk::ImageFileReader<IncidentSpectrumImageType>;

  using DetectorResponseImageType = itk::Image<PixelValueType, Dimension - 1>;
  using DetectorResponseReaderType = itk::ImageFileReader<DetectorResponseImageType>;

  using MaterialAttenuationsImageType = itk::Image<PixelValueType, Dimension - 1>;
  using MaterialAttenuationsReaderType = itk::ImageFileReader<MaterialAttenuationsImageType>;

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
  ConstantImageSourceType::PointType   origin;
  ConstantImageSourceType::SizeType    size;
  ConstantImageSourceType::SpacingType spacing;

  ConstantImageSourceType::Pointer projectionsSource = ConstantImageSourceType::New();
  origin[0] = -255.;
  origin[1] = -0.5;
  origin[2] = -255.;
#if FAST_TESTS_NO_CHECKS
  size[0] = 2;
  size[1] = 1;
  size[2] = NumberOfProjectionImages;
  spacing[0] = 504.;
  spacing[1] = 504.;
  spacing[2] = 504.;
#else
  size[0] = 64;
  size[1] = 1;
  size[2] = NumberOfProjectionImages;
  spacing[0] = 8.;
  spacing[1] = 1.;
  spacing[2] = 1.;
#endif
  projectionsSource->SetOrigin(origin);
  projectionsSource->SetSpacing(spacing);
  projectionsSource->SetSize(size);
  projectionsSource->SetConstant(0.);

  // Initialize the multi-materials projections
  DecomposedProjectionType::Pointer decomposed = DecomposedProjectionType::New();
  decomposed->SetVectorLength(3);
  decomposed->SetOrigin(origin);
  decomposed->SetSpacing(spacing);
  DecomposedProjectionType::RegionType region;
  DecomposedProjectionType::IndexType  index;
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
    REIType::VectorType semiprincipalaxis, center;
    semiprincipalaxis.Fill(10.);
    center.Fill(0.);
    //    center[0] = (material-1) * 15;
    //    center[2] = (material-1) * 15;
    center[0] = 15;
    center[2] = 15;
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
    itk::ImageRegionIterator<DecomposedProjectionType> outIt(decomposed, decomposed->GetLargestPossibleRegion());
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
    rtk::SpectralForwardModelImageFilter<DecomposedProjectionType, MeasuredProjectionsType, IncidentSpectrumImageType>;
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
  DecomposedProjectionType::Pointer initialDecomposedProjections = DecomposedProjectionType::New();
  initialDecomposedProjections->CopyInformation(decomposed);
  initialDecomposedProjections->SetRegions(region);
  initialDecomposedProjections->SetVectorLength(3);
  initialDecomposedProjections->Allocate();
  DecomposedProjectionType::PixelType initPixel;
  initPixel.SetSize(3);
  initPixel[0] = 0.1;
  initPixel[1] = 0.1;
  initPixel[2] = 10;
  initialDecomposedProjections->FillBuffer(initPixel);

  // Create and set the simplex filter to perform the decomposition
  using SimplexFilterType = rtk::SimplexSpectralProjectionsDecompositionImageFilter<DecomposedProjectionType,
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
  CheckVectorImageQuality<DecomposedProjectionType>(simplex->GetOutput(), decomposed, 0.0001, 15, 2.0);

  std::cout << "\n\n****** Case 2: Heuristically-determined initial values ******" << std::endl;

  simplex->SetGuessInitialization(true);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(simplex->Update())
  CheckVectorImageQuality<DecomposedProjectionType>(simplex->GetOutput(), decomposed, 0.0001, 15, 2.0);

#ifndef ITK_FUTURE_LEGACY_REMOVE
  std::cout << "\n\n****** Case 3: Legacy VectorImage type for incident spectrum ******" << std::endl;

  using VectorImageType = itk::VectorImage<PixelValueType, Dimension - 1>;
  using VectorSpectrumReaderType = itk::ImageFileReader<VectorImageType>;
  VectorSpectrumReaderType::Pointer vectorSpectrumReader = VectorSpectrumReaderType::New();
  vectorSpectrumReader->SetFileName(argv[4]);
  vectorSpectrumReader->Update();
  forward->SetInputIncidentSpectrum(vectorSpectrumReader->GetOutput());
  simplex->SetInputIncidentSpectrum(vectorSpectrumReader->GetOutput());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(simplex->Update())
  CheckVectorImageQuality<DecomposedProjectionType>(simplex->GetOutput(), decomposed, 0.0001, 15, 2.0);
#endif

  std::cout << "\n\nTest PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
