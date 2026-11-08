#include "rtkTest.h"
#include "rtkMacro.h"
#include "rtkConditionalMedianImageFilter.h"
#include "rtkProjectionsReader.h"

#include <itkExtractImageFilter.h>
#include <itkImageFileReader.h>
#include <itkVectorImage.h>
#include <itkImageFileWriter.h>

#include <vector>

/**
 * \file rtkprojectionsreadervectorimagetest.cxx
 *
 * \brief Test rtk::ProjectionsReader with an itk::VectorImage output.
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

  itksys::SystemTools::RemoveFile("slice_rtkprojectionsreadervectorimagetest.mha");
  std::cout << "\n\nTest PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
