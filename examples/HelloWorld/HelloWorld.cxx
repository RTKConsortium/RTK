#include <rtkFDKBackProjectionImageFilter.h>

int main(int , char **)
{
  // Define the type of pixel and the image dimension
  using OutputPixelType = float;
  const unsigned int Dimension = 3;

  // Define the type of image
  using OutputImageType = itk::Image< OutputPixelType, Dimension >;

  // Define and allocate the FDK Back Projection Filter
  using BPType = rtk::FDKBackProjectionImageFilter<OutputImageType, OutputImageType>;
  BPType::Pointer p = BPType::New();

  std::cout << "RTK Hello World!" << std::endl;

  return 0;
}

