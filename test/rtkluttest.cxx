#include "rtkTest.h"
#include "rtkProjectionsReader.h"
#include "rtkMacro.h"
#include <itkRegularExpressionSeriesFileNames.h>
#include <itkCastImageFilter.h>

/**
 * \file rtkluttest.cxx
 *
 * \brief Test rtk::LookupTableImageFilter
 *
 * This test float and double lookup table filters. It compares the output of
 * the Elekta lookup table with the values of the same lut casted to float and
 * double.
 *
 * \author Simon Rit
 */

int
main(int argc, char * argv[])
{
  if (argc < 2)
  {
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << " ElektaProjections " << std::endl;
    return EXIT_FAILURE;
  }

  // Elekta projections reader
  using ShortImageType = itk::Image<unsigned short, 3>;
  auto                     r = rtk::ProjectionsReader<ShortImageType>::New();
  std::vector<std::string> fileNames;
  fileNames.emplace_back(argv[1]);
  r->SetFileNames(fileNames);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(r->Update());

  /***** Float *****/
  using FloatImageType = itk::Image<float, 3>;
  using ShortFloatLUTType = rtk::ElektaSynergyLookupTableImageFilter<FloatImageType>;
  auto sflut = ShortFloatLUTType::New();
  sflut->SetInput(r->GetOutput());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(sflut->Update());

  auto fCast = itk::CastImageFilter<ShortImageType, FloatImageType>::New();
  fCast->SetInput(r->GetOutput());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(fCast->Update());

  auto flCast = itk::CastImageFilter<ShortFloatLUTType::LookupTableType, itk::Image<float, 1>>::New();
  flCast->SetInput(sflut->GetLookupTable());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(flCast->Update());

  auto flut = rtk::LookupTableImageFilter<FloatImageType, FloatImageType>::New();
  flut->SetInput(fCast->GetOutput());
  flut->SetLookupTable(flCast->GetOutput());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(flut->Update());

  CheckImageQuality<FloatImageType>(flut->GetOutput(), sflut->GetOutput(), 1.6e-7, 100, 2.0);

  /***** Double *****/
  using DoubleImageType = itk::Image<float, 3>;
  using ShortDoubleLUTType = rtk::ElektaSynergyLookupTableImageFilter<DoubleImageType>;
  auto sdlut = ShortDoubleLUTType::New();
  sdlut->SetInput(r->GetOutput());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(sdlut->Update());

  auto dCast = itk::CastImageFilter<ShortImageType, DoubleImageType>::New();
  dCast->SetInput(r->GetOutput());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(dCast->Update());

  auto dlCast = itk::CastImageFilter<ShortDoubleLUTType::LookupTableType, itk::Image<float, 1>>::New();
  dlCast->SetInput(sdlut->GetLookupTable());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(dlCast->Update());

  auto dlut = rtk::LookupTableImageFilter<DoubleImageType, DoubleImageType>::New();
  dlut->SetInput(dCast->GetOutput());
  dlut->SetLookupTable(dlCast->GetOutput());
  TRY_AND_EXIT_ON_ITK_EXCEPTION(dlut->Update());

  CheckImageQuality<DoubleImageType>(dlut->GetOutput(), sdlut->GetOutput(), 1.6e-7, 100, 2.0);

  // If all succeed
  std::cout << "\n\nTest PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
