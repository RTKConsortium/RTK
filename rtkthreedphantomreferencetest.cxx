#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif
//#include <iostream>
#include "rtkthreedphantomreferencetest_ggo.h"
#include "rtkGgoFunctions.h"
#include <fstream>
#include "itkRandomImageSource.h"
#include "itkImageFileWriter.h"
#include "itkImageFileReader.h"
#include "itkRawImageIO.h"
#include "itkImageRegionConstIterator.h"
#include "itkSheppLoganPhantomFilter.h"
#include "itkDrawQuadricFunctor.h"

int main(int argc, char* argv[])
{
  GGO(rtkthreedphantomreferencetest, args_info);
  const unsigned int Dimension = 3;
  typedef float                                                          OutputPixelType;
  typedef itk::Image< OutputPixelType, Dimension >                       OutputImageType;
  typedef OutputImageType::PixelType                                     PixelType;
  typedef double                                                         ErrorType;
  typedef itk::DrawQuadricFunctor<OutputImageType, OutputImageType>      DQType;
  typedef itk::SheppLoganPhantomFilter<OutputImageType, OutputImageType> SLPType;
  typedef itk::ImageRegionConstIterator<OutputImageType>                 ImageIteratorType;
  typedef itk::ConstantImageSource< OutputImageType >                    ConstantImageSourceType;

  // Geometry
  if(args_info.verbose_flag)
    std::cout << "Reading geometry information from "
              << args_info.geometry_arg
              << "..."
              << std::endl;

  itk::ThreeDCircularProjectionGeometryXMLFileReader::Pointer geometryReader;
  geometryReader = itk::ThreeDCircularProjectionGeometryXMLFileReader::New();
  geometryReader->SetFilename(args_info.geometry_arg);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( geometryReader->GenerateOutputInformation() )
  // Constant Image Sources whether for the reference and test objects
  ConstantImageSourceType::Pointer constantImageSourceRef = ConstantImageSourceType::New();
  // Constant Image Source for the reference object, using ggo parameters
  rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkthreedphantomreferencetest>(constantImageSourceRef, args_info);
  // Reader, obtaining the reconstruction test file.
  itk::ImageFileReader<OutputImageType>::Pointer reader;
  reader = itk::ImageFileReader<OutputImageType>::New();
  reader->SetFileName(args_info.test_arg);

  TRY_AND_EXIT_ON_ITK_EXCEPTION( reader->Update() )

  // Create a reference object (in this case a 3D phantom reference).
  DQType::Pointer dq = DQType::New();
  dq->SetNumberOfThreads(1);
  dq->SetInput(constantImageSourceRef->GetOutput());
  dq->SetConfigFile(args_info.phantomfile_arg);

  TRY_AND_EXIT_ON_ITK_EXCEPTION( dq->Update() )

  ImageIteratorType itTest( reader->GetOutput(), reader->GetOutput()->GetBufferedRegion() );
  ImageIteratorType itRef( dq->GetOutput(), dq->GetOutput()->GetBufferedRegion() );

  ErrorType TestError     = 0.;
  ErrorType TestTolerance = 0.005;
  ErrorType EnerError     = 0.;

  itTest.GoToBegin();
  itRef.GoToBegin();
  while( !itTest.IsAtEnd() )
    {
    PixelType TestVal = itTest.Get();
    PixelType RefVal = itRef.Get();
    if( TestVal != RefVal )
      {
        TestError += abs(RefVal - TestVal);
        EnerError += pow((RefVal - TestVal), 2.0);
      }
    ++itTest;
    ++itRef;
    }
  // Error per Pixel
  ErrorType ErrorPerPixel = TestError/pow(args_info.dimension_arg[0], 3.0);
  std::cout << "\nError per Pixel = " << ErrorPerPixel << std::endl;
  // MSE
  ErrorType MSE = EnerError/pow(args_info.dimension_arg[0], 3.0);
  std::cout << "MSE = " << MSE << std::endl;
  // PSNR
  ErrorType PSNR = 20*log10(2.0) - 10*log10(MSE);
  std::cout << "PSNR = " << PSNR << "dB" << std::endl;
  // QI
  ErrorType QI = (2.0-ErrorPerPixel)/2.0;
  std::cout << "QI = " << QI << std::endl;
  // Checking results
  if (ErrorPerPixel > TestTolerance)
  {
    std::cerr << "Test Failed, Error per pixel not valid!" << std::endl;
    return EXIT_FAILURE;
  }
  if (PSNR < 25.0)
  {
    std::cerr << "Test Failed, PSNR not valid!" << std::endl;
    return EXIT_FAILURE;
  }
  // Write out the TEST object
  itk::ImageFileWriter<OutputImageType>::Pointer writer;
  writer = itk::ImageFileWriter<OutputImageType>::New();
  writer->SetInput(reader->GetOutput());
  writer->SetFileName(args_info.output_arg[1]);

  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() )

  // Write out the REFERENCE object
  writer->SetInput(dq->GetOutput());
  writer->SetFileName(args_info.output_arg[0]);

  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() )

  std::cerr << "Test PASSED! " << std::endl;
  return EXIT_SUCCESS;
}
