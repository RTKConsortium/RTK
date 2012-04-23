#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif
//#include <iostream>
#include "rtkthreedphantomreferencetest2_ggo.h"
#include "rtkGgoFunctions.h"
#include <fstream>
#include "itkRandomImageSource.h"
#include "itkImageFileWriter.h"
#include "itkImageFileReader.h"
#include "itkRawImageIO.h"
#include "itkImageRegionConstIterator.h"
#include "itkSheppLoganPhantomFilter.h"
#include "itkDrawQuadricFunctor.h"

#include "itkProjectionsReader.h"
#include "itkFDKConeBeamReconstructionFilter.h"
#include <itkRegularExpressionSeriesFileNames.h>

int main(int argc, char* argv[])
{
  GGO(rtkthreedphantomreferencetest2, args_info);
  const unsigned int Dimension = 3;
  typedef float                                                          OutputPixelType;
  typedef itk::Image< OutputPixelType, Dimension >                       OutputImageType;
  typedef OutputImageType::PixelType                                     PixelType;
  typedef double                                                         ErrorType;
  typedef itk::DrawQuadricFunctor<OutputImageType, OutputImageType>      DQType;
  typedef itk::SheppLoganPhantomFilter<OutputImageType, OutputImageType> SLPType;
  typedef itk::ImageRegionConstIterator<OutputImageType>                 ImageIteratorType;
  typedef itk::ConstantImageSource< OutputImageType >                    ConstantImageSourceType;

  // Generate file names
  itk::RegularExpressionSeriesFileNames::Pointer names = itk::RegularExpressionSeriesFileNames::New();
  names->SetDirectory(args_info.path_arg);
  names->SetNumericSort(false);
  names->SetRegularExpression(args_info.regexp_arg);
  names->SetSubMatch(0);

  if(args_info.verbose_flag)
    std::cout << "Regular expression matches "
              << names->GetFileNames().size()
              << " file(s)..."
              << std::endl;

  // Projections reader
  typedef itk::ProjectionsReader< OutputImageType > ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileNames( names->GetFileNames() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( reader->GenerateOutputInformation() );

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
  rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkthreedphantomreferencetest2>(constantImageSourceRef, args_info);

#define SET_FELDKAMP_OPTIONS(f) \
    f->SetInput( 0, constantImageSourceRef->GetOutput() ); \
    f->SetInput( 1, reader->GetOutput() ); \
    f->SetGeometry( geometryReader->GetOutputObject() ); \
    f->GetRampFilter()->SetTruncationCorrection(args_info.pad_arg); \
    f->GetRampFilter()->SetHannCutFrequency(args_info.hann_arg);

  // FDK reconstruction filtering
  itk::ImageToImageFilter<OutputImageType, OutputImageType>::Pointer feldkamp;
  typedef itk::FDKConeBeamReconstructionFilter< OutputImageType > FDKCPUType;
  feldkamp = FDKCPUType::New();
  SET_FELDKAMP_OPTIONS( static_cast<FDKCPUType*>(feldkamp.GetPointer()) );

  TRY_AND_EXIT_ON_ITK_EXCEPTION( feldkamp->Update() )

  // Create a reference object (in this case a 3D phantom reference).
  DQType::Pointer dq = DQType::New();
  dq->SetNumberOfThreads(1);
  dq->SetInput(constantImageSourceRef->GetOutput());
  dq->SetConfigFile(args_info.phantomfile_arg);
  try
  {
    dq->Update();
  }
  catch ( itk::ExceptionObject & excp)
  {
    std::cerr << "\nError while creating reference object" << std::endl;
    std::cerr << "\nException caught: " << excp << std::endl;
  }

  ImageIteratorType itTest( feldkamp->GetOutput(), feldkamp->GetOutput()->GetBufferedRegion() );
  ImageIteratorType itRef( dq->GetOutput(), dq->GetOutput()->GetBufferedRegion() );

  ErrorType TestError     = 0.;
  ErrorType TestTolerance = 0.005;
  ErrorType EnerError     = 0.;

  itTest.GoToBegin();
  itRef.GoToBegin();

  while( !itRef.IsAtEnd() )
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
  typedef itk::ImageFileWriter<  OutputImageType > WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg[1] );
  writer->SetInput( feldkamp->GetOutput() );

  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() )

  // Write out the REFERENCE object
  writer->SetInput(dq->GetOutput());
  writer->SetFileName(args_info.output_arg[0]);

  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() )

  std::cerr << "Test PASSED! " << std::endl;
  return EXIT_SUCCESS;
}




