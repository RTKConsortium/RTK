#include "rtkTest.h"
#include "rtkMacro.h"
#include <itkImageFileReader.h>
#include "rtkGetNewtonUpdateImageFilter.h"
#include <itkCSVArray2DFileReader.h>

/**
 * \file rtknewtonupdatetest.cxx
 *
 * \brief Test for the filter rtkNewtonUpdateImageFilter
 *
 * This test reads gradient and hessian, runs rtkNewtonUpdateImageFilter
 * to get the update computed by Newton's method, and compares its outputs
 * to the expected one (computed with Matlab)
 *
 * \author Cyril Mory
 */

int
main(int argc, char * argv[])
{
  if (argc < 4)
  {
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << " gradient.mha hessian.mha reference.mha" << std::endl;
    return EXIT_FAILURE;
  }

  // Define types
  constexpr unsigned int nMaterials = 3;
  using dataType = double;
  using TGradient = itk::Image<itk::Vector<dataType, nMaterials>, 3>;
  using THessian = itk::Image<itk::Vector<dataType, nMaterials * nMaterials>, 3>;

  // Define, instantiate, set and update readers
  using GradientReaderType = itk::ImageFileReader<TGradient>;
  auto gradientReader = GradientReaderType::New();
  gradientReader->SetFileName(argv[1]);
  gradientReader->Update();

  using HessianReaderType = itk::ImageFileReader<THessian>;
  auto hessianReader = HessianReaderType::New();
  hessianReader->SetFileName(argv[2]);
  hessianReader->Update();

  using OutputReaderType = itk::ImageFileReader<TGradient>;
  auto outputReader = OutputReaderType::New();
  outputReader->SetFileName(argv[3]);
  outputReader->Update();

  // Create the filter
  using NewtonUpdateFilterType = rtk::GetNewtonUpdateImageFilter<TGradient, THessian>;
  auto newtonUpdate = NewtonUpdateFilterType::New();

  // Set its inputs
  newtonUpdate->SetInputGradient(gradientReader->GetOutput());
  newtonUpdate->SetInputHessian(hessianReader->GetOutput());

  // Update the filter
  TRY_AND_EXIT_ON_ITK_EXCEPTION(newtonUpdate->Update());

  // 2. Compare read projections
  CheckVectorImageQuality<TGradient>(newtonUpdate->GetOutput(), outputReader->GetOutput(), 1.e-9, 200, 2000.0);

  // If all succeed
  std::cout << "\n\nTest PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
