#include "rtkTest.h"
#include "rtkMacro.h"
#include "rtkConditionalMedianImageFilter.h"
#include "rtkProjectionsReader.h"

#include <itkBinShrinkImageFilter.h>
#include <itkExtractImageFilter.h>
#include <itkImageFileReader.h>
#include <itkVectorImage.h>
#include <itkVectorIndexSelectionCastImageFilter.h>
#include <itkImageFileWriter.h>

#include <vector>

/**
 * \file rtkprojectionsreadervectorimagetest.cxx
 *
 * \brief Test rtk::ProjectionsReader with an itk::VectorImage output and with the
 * extraction of one component of a vector image followed by preprocessing.
 */

int
rtkprojectionsreadervectorimagetest(int argc, char * argv[])
{
  if (argc < 2)
  {
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << " vectorProjectionImage " << std::endl;
    return EXIT_FAILURE;
  }

  using VectorImageType = itk::VectorImage<float, 3>;

  auto referenceReader = itk::ImageFileReader<VectorImageType>::New();
  referenceReader->SetFileName(argv[1]);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(referenceReader->Update());

  auto projectionsReader = rtk::ProjectionsReader<VectorImageType>::New();
  projectionsReader->SetFileNames(std::vector<std::string>{ argv[1] });
  TRY_AND_EXIT_ON_ITK_EXCEPTION(projectionsReader->Update());

  CheckVariableLengthVectorImageQuality<VectorImageType>(
    projectionsReader->GetOutput(), referenceReader->GetOutput(), 1e-10, 1e10, 1.);

  using MedianType = rtk::ConditionalMedianImageFilter<VectorImageType>;
  auto                         directMedian = MedianType::New();
  MedianType::MedianRadiusType radius;
  radius[0] = 1;
  radius[1] = 1;
  radius[2] = 0;

  directMedian->SetRadius(radius);
  directMedian->SetThresholdMultiplier(1.);
  directMedian->SetInput(referenceReader->GetOutput());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(directMedian->Update());

  auto denoisingProjectionsReader = rtk::ProjectionsReader<VectorImageType>::New();
  denoisingProjectionsReader->SetFileNames(std::vector<std::string>{ argv[1] });
  denoisingProjectionsReader->SetMedianRadius(radius);
  denoisingProjectionsReader->SetConditionalMedianThresholdMultiplier(1.);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(denoisingProjectionsReader->Update());

  CheckVariableLengthVectorImageQuality<VectorImageType>(
    denoisingProjectionsReader->GetOutput(), directMedian->GetOutput(), 1e-10, 1e10, 1.);

  VectorImageType::RegionType sliceRegion = denoisingProjectionsReader->GetOutput()->GetLargestPossibleRegion();
  sliceRegion.SetSize(2, 0);
  using SliceImageType = itk::VectorImage<float, 2>;
  auto extract = itk::ExtractImageFilter<VectorImageType, SliceImageType>::New();
  extract->SetInput(referenceReader->GetOutput());
  extract->SetExtractionRegion(sliceRegion);
  extract->SetDirectionCollapseToIdentity();
  TRY_AND_EXIT_ON_ITK_EXCEPTION(extract->Update());
  itk::WriteImage(extract->GetOutput(), "slice_rtkprojectionsreadervectorimagetest.mha");

  projectionsReader->SetFileNames(std::vector<std::string>{ "slice_rtkprojectionsreadervectorimagetest.mha",
                                                            "slice_rtkprojectionsreadervectorimagetest.mha" });
  TRY_AND_EXIT_ON_ITK_EXCEPTION(projectionsReader->UpdateLargestPossibleRegion());

  auto referenceSeriesReader = itk::ImageSeriesReader<VectorImageType>::New();
  referenceSeriesReader->SetFileNames(std::vector<std::string>{ "slice_rtkprojectionsreadervectorimagetest.mha",
                                                                "slice_rtkprojectionsreadervectorimagetest.mha" });
  TRY_AND_EXIT_ON_ITK_EXCEPTION(referenceSeriesReader->Update());

  CheckVariableLengthVectorImageQuality<VectorImageType>(
    projectionsReader->GetOutput(), referenceSeriesReader->GetOutput(), 1e-10, 1e10, 1.);

  // Extraction of one component followed by conditional median and binning
  using ScalarImageType = itk::Image<float, 3>;
  constexpr unsigned int component = 1;
  auto                   selection = itk::VectorIndexSelectionCastImageFilter<VectorImageType, ScalarImageType>::New();
  selection->SetInput(referenceReader->GetOutput());
  selection->SetIndex(component);

  using ScalarMedianType = rtk::ConditionalMedianImageFilter<ScalarImageType>;
  auto scalarMedian = ScalarMedianType::New();
  scalarMedian->SetInput(selection->GetOutput());
  scalarMedian->SetRadius(radius);
  scalarMedian->SetThresholdMultiplier(1.);

  using BinType = itk::BinShrinkImageFilter<ScalarImageType, ScalarImageType>;
  auto                       bin = BinType::New();
  BinType::ShrinkFactorsType shrinkFactors;
  shrinkFactors[0] = 2;
  shrinkFactors[1] = 1;
  shrinkFactors[2] = 1;
  bin->SetInput(scalarMedian->GetOutput());
  bin->SetShrinkFactors(shrinkFactors);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(bin->Update());

  auto componentProjectionsReader = rtk::ProjectionsReader<ScalarImageType>::New();
  componentProjectionsReader->SetFileNames(std::vector<std::string>{ argv[1] });
  componentProjectionsReader->SetVectorComponent(component);
  componentProjectionsReader->SetMedianRadius(radius);
  componentProjectionsReader->SetConditionalMedianThresholdMultiplier(1.);
  componentProjectionsReader->SetShrinkFactors(shrinkFactors);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(componentProjectionsReader->Update());

  CheckImageQuality<ScalarImageType>(componentProjectionsReader->GetOutput(), bin->GetOutput(), 1e-10, 1e10, 1.);

  itksys::SystemTools::RemoveFile("slice_rtkprojectionsreadervectorimagetest.mha");
  std::cout << "\n\nTest PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
