#include <itkImageRegionConstIterator.h>
#include <itkStreamingImageFilter.h>
#include <itkCudaImage.h>
#include "itkInPlaceImageTestFilter.h"

int main(int, char** )
{
  itk::MultiThreader::SetGlobalMaximumNumberOfThreads( 1 ); 
  
  const unsigned int Dimension = 3;
  typedef float                                    OutputPixelType;
  typedef itk::CudaImage< OutputPixelType, Dimension > OutputImageType;
  //typedef itk::Image< OutputPixelType, Dimension > OutputImageType;

  OutputImageType::Pointer cudaImage = OutputImageType::New();
  OutputImageType::SizeType size;
  size[0] = size[1] = size[2] = 30;
  OutputImageType::RegionType region;
  region.SetSize(size);
  cudaImage->SetRegions(region);
  cudaImage->Allocate();
  cudaImage->FillBuffer(3.0);
  
  cudaImage->UpdateBuffers();
  cudaImage->GetCudaDataManager()->SetCPUBufferDirty();

  // FOV
  typedef itk::InPlaceImageTestFilter<OutputImageType, OutputImageType> FilterType;
  FilterType::Pointer filter = FilterType::New();
  filter->SetInput(cudaImage);
  filter->Update();
  OutputImageType::IndexType ind;
  ind.Fill(10);
  std::cout << filter->GetOutput()->GetPixel(ind) << std::endl;

  // FOV
  filter->GetOutput()->ReleaseData();
  filter->Update();
  std::cout << filter->GetOutput()->GetPixel(ind) << std::endl;

  std::cout << "Test PASSED! " << std::endl;
}
