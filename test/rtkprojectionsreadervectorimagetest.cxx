#include "rtkTest.h"
#include "rtkMacro.h"
#include "rtkConditionalMedianImageFilter.h"
#include "rtkProjectionsReader.h"

#include <itkImageFileReader.h>
#include <itkVectorImage.h>

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
    projectionsReader->GetOutput(), referenceReader->GetOutput(), 1e-5, 100, 2000.0);

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
    denoisingProjectionsReader->GetOutput(), directMedian->GetOutput(), 0.04, 60, 2000.0);

  std::cout << "\n\nTest PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
