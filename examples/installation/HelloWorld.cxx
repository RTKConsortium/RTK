#include <itkFDKBackProjectionImageFilter.h>

int main(int argc, char **argv)
{
  typedef float OutputPixelType;
  const unsigned int Dimension = 3;
  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;
  typedef itk::FDKBackProjectionImageFilter<OutputImageType, OutputImageType> BPType;
  BPType::Pointer p = BPType::New();
  return 0;
}

