#include "rtkTestConfiguration.h"
#include "rtkCudaCropImageFilter.h"

/**
 * \file rtkcroptest.cxx
 * \brief Functional test for the classes performing crop filtering
 * \author Julien Jomier
 */
int main(int , char** )
{
  const unsigned int Dimension = 3;
  typedef float PixelType;
  typedef itk::CudaImage< PixelType, Dimension > ImageType;

  ImageType::Pointer image = ImageType::New();
  ImageType::RegionType region;
  ImageType::SizeType size;
  size[0] = 50;
  size[1] = 50;
  size[2] = 50;
  region.SetSize(size);
  image->SetRegions(region);
  image->Allocate();
  image->FillBuffer(12.3);

  typedef rtk::CudaCropImageFilter CropImageFilter;
  CropImageFilter::Pointer crop = CropImageFilter::New();
  crop->SetInput(image);
 
  ImageType::SizeType upCropSize, lowCropSize;
  for(unsigned int i=0; i<ImageType::ImageDimension; i++)
    {
    lowCropSize[i] = 1;
    upCropSize[i]  = 10;
    }
  crop->SetUpperBoundaryCropSize(upCropSize);
  crop->SetLowerBoundaryCropSize(lowCropSize);
 
  try
    {
    crop->Update();
    }
 catch( itk::ExceptionObject & err )                                   
    { 
    std::cerr << err << std::endl;                                      
    exit(EXIT_FAILURE);                                                 
    }
    
  ImageType::IndexType index;
  index.Fill(2);
  
  if(fabs(crop->GetOutput()->GetPixel(index)-12.3)>0.0001)
    {
    std::cout << "Output should be 12.3. Value Computed = " 
              << crop->GetOutput()->GetPixel(index) << std::endl;
    return EXIT_FAILURE;
    }
  
  std::cout << "Done!" << std::endl;
  std::cout << "\n\nTest PASSED! " << std::endl;

  return EXIT_SUCCESS;
}
