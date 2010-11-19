#include "rtkfdk_ggo.h"
#include "rtkGgoFunctions.h"
#include "rtkConfiguration.h"

#include "itkThreeDCircularProjectionGeometryXMLFile.h"
#include "itkProjectionsReader.h"
#include "itkFFTRampImageFilter.h"
#include "itkFDKWeightProjectionFilter.h"

#include "itkFDKBackProjectionImageFilter.h"
#ifdef CUDA_FOUND
#  include "itkCudaFDKBackProjectionImageFilter.h"
#endif

#include <itkImageFileWriter.h>
#include <itkRegularExpressionSeriesFileNames.h>
#include <itkTimeProbe.h>
#include <itkStreamingImageFilter.h>

int main(int argc, char * argv[])
{
  GGO(rtkfdk, args_info);

  typedef float OutputPixelType;
  const unsigned int Dimension = 3;

  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;

  // Generate file names
  itk::RegularExpressionSeriesFileNames::Pointer names = itk::RegularExpressionSeriesFileNames::New();
  names->SetDirectory(args_info.path_arg);
  names->SetNumericSort(false);
  names->SetRegularExpression(args_info.regexp_arg);
  names->SetSubMatch(0);

  // Projections reader
  typedef itk::ProjectionsReader< OutputImageType > ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileNames( names->GetFileNames() );
  reader->GenerateOutputInformation();

  // Geometry
  itk::ThreeDCircularProjectionGeometryXMLFileReader::Pointer geometryReader = itk::ThreeDCircularProjectionGeometryXMLFileReader::New();
  geometryReader->SetFilename(args_info.geometry_arg);
  geometryReader->GenerateOutputInformation();

  // Weight projections according to fdk algorithm
  typedef itk::FDKWeightProjectionFilter< OutputImageType > WeightFilterType;
  WeightFilterType::Pointer weightFilter = WeightFilterType::New();
  weightFilter->SetInput( reader->GetOutput() );
  weightFilter->SetSourceToDetectorDistance( geometryReader->GetOutputObject()->GetSourceToDetectorDistance() );

  // Ramp filter
  typedef itk::FFTRampImageFilter<OutputImageType> RampFilterType;
  RampFilterType::Pointer rampFilter = RampFilterType::New();
  rampFilter->SetInput( weightFilter->GetOutput() );
  rampFilter->SetTruncationCorrection(args_info.pad_arg);

  // Streaming filter
  typedef itk::StreamingImageFilter<OutputImageType, OutputImageType> StreamerType;
  StreamerType::Pointer streamer = StreamerType::New();
  streamer->SetInput( rampFilter->GetOutput() );
  streamer->SetNumberOfStreamDivisions( 1 + reader->GetOutput()->GetLargestPossibleRegion().GetNumberOfPixels() / (1024*1024*4) );

  // Try to do all 2D pre-processing
  try {
    streamer->Update();
  } catch( itk::ExceptionObject & err ) {
    std::cerr << "ExceptionObject caught !" << std::endl;
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
  }

  // Create reconstructed volume
  OutputImageType::Pointer tomography = rtk::CreateImageFromGgo<OutputImageType>(args_info);

  // Backprojection
  typedef itk::FDKBackProjectionImageFilter<OutputImageType, OutputImageType> BackProjectionFilterType;
  BackProjectionFilterType::Pointer bpFilter;
  if(!strcmp(args_info.hardware_arg, "cpu"))
    bpFilter = BackProjectionFilterType::New();
  else if(!strcmp(args_info.hardware_arg, "cuda"))
    {
#ifdef CUDA_FOUND
    bpFilter = itk::CudaFDKBackProjectionImageFilter::New();
#else
    std::cerr << "The program has not been compiled with cuda option" << std::endl;
    return EXIT_FAILURE;
#endif
    }
  bpFilter->SetInput( 0, tomography );
  bpFilter->SetInput( 1, streamer->GetOutput() );
  bpFilter->SetGeometry( geometryReader->GetOutputObject() );
  bpFilter->SetInPlace( true );

  // Write
  typedef itk::ImageFileWriter<  OutputImageType >  WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg );
  writer->SetInput( bpFilter->GetOutput() );

  try {
    writer->Update();
  } catch( itk::ExceptionObject & err ) {
    std::cerr << "ExceptionObject caught !" << std::endl;
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
