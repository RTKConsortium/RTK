#include <rtkFDKBackProjectionImageFilter.h>

int main(int argc, char **argv)
{
  // Define the type of pixel and the image dimension
  typedef float OutputPixelType;
  const unsigned int Dimension = 3;

  // Define the type of image
  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;

  // Define and allocate the FDK Back Projection Filter
  typedef rtk::FDKBackProjectionImageFilter<OutputImageType, OutputImageType> BPType;
  BPType::Pointer p = BPType::New();

  std::cout << "RTK Hello World!" << std::endl;

  return 0;
}

