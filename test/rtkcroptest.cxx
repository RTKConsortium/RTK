#include "rtkTestConfiguration.h"
#include "rtkCudaCropImageFilter.h"

/**
 * \file rtkcroptest.cxx
 * \brief Functional test for the classes performing crop filtering
 * \author Julien Jomier
 */
int
main(int, char **)
{
  constexpr unsigned int Dimension = 3;
  using PixelType = float;
  using ImageType = itk::CudaImage<PixelType, Dimension>;

  auto image = ImageType::New();
  image->SetRegions(itk::MakeSize(50, 50, 50));
  image->Allocate();
  image->FillBuffer(12.3);

  using CropImageFilter = rtk::CudaCropImageFilter;
  auto crop = CropImageFilter::New();
  crop->SetInput(image);
  crop->SetUpperBoundaryCropSize(itk::MakeSize(1, 1, 1));
  crop->SetLowerBoundaryCropSize(itk::MakeSize(10, 10, 10));

  try
  {
    crop->Update();
  }
  catch (itk::ExceptionObject & err)
  {
    std::cerr << err << std::endl;
    exit(EXIT_FAILURE);
  }

  ImageType::IndexType index;
  index.Fill(2);

  if (itk::Math::abs(crop->GetOutput()->GetPixel(index) - 12.3) > 0.0001)
  {
    std::cout << "Output should be 12.3. Value Computed = " << crop->GetOutput()->GetPixel(index) << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Done!" << std::endl;
  std::cout << "\n\nTest PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
