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

  OutputImageType::Pointer cudaImage = OutputImageType::New();
  OutputImageType::SizeType size;
  size[0] = size[1] = size[2] = 30;
  OutputImageType::RegionType region;
  region.SetSize(size);
  cudaImage->SetRegions(region);
  cudaImage->Allocate();
  cudaImage->FillBuffer(3.0);
  cudaImage->UpdateBuffers();

  // FOV
  typedef itk::InPlaceImageTestFilter<OutputImageType, OutputImageType> FilterType;
  FilterType::Pointer filter = FilterType::New();
  filter->SetInput(cudaImage);
  
  // Test the streamer
  typedef itk::StreamingImageFilter<OutputImageType,OutputImageType> StreamerType;
  StreamerType::Pointer streamer = StreamerType::New();
  streamer->SetInput(filter->GetOutput());
  streamer->SetNumberOfStreamDivisions(2);
  streamer->Update();

  OutputImageType::IndexType ind;
  ind.Fill(10);

  std::cout << streamer->GetOutput()->GetPixel(ind) << std::endl;

  std::cout << "Test PASSED! " << std::endl;
}
