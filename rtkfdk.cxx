#include "rtkfdk_ggo.h"
#include "rtkGgoFunctions.h"
#include "rtkConfiguration.h"

#include "itkThreeDCircularProjectionGeometryXMLFile.h"
#include "itkProjectionsReader.h"
#include "itkDisplacedDetectorImageFilter.h"
#include "itkFDKWeightProjectionFilter.h"
#include "itkFFTRampImageFilter.h"

#include "itkFDKBackProjectionImageFilter.h"
#if CUDA_FOUND
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

  if(args_info.verbose_flag)
    std::cout << "Regular expression matches "
              << names->GetFileNames().size()
              << " file(s)..."
              << std::endl;

  // Projections reader
  typedef itk::ProjectionsReader< OutputImageType > ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileNames( names->GetFileNames() );
  reader->GenerateOutputInformation();

  itk::TimeProbe readerProbe;
  if(!args_info.lowmem_flag)
    {
    if(args_info.verbose_flag)
      std::cout << "Reading... " << std::flush;
    try {
      readerProbe.Start();
      reader->Update();
      readerProbe.Stop();
    } catch( itk::ExceptionObject & err ) {
      std::cerr << "ExceptionObject caught !" << std::endl;
      std::cerr << err << std::endl;
      return EXIT_FAILURE;
    }
    if(args_info.verbose_flag)
      std::cout << "It took " << readerProbe.GetMeanTime() << ' ' << readerProbe.GetUnit() << std::endl;
    }

  // Geometry
  if(args_info.verbose_flag)
    std::cout << "Reading geometry information from "
              << args_info.geometry_arg
              << "..."
              << std::endl;
  itk::ThreeDCircularProjectionGeometryXMLFileReader::Pointer geometryReader = itk::ThreeDCircularProjectionGeometryXMLFileReader::New();
  geometryReader->SetFilename(args_info.geometry_arg);
  geometryReader->GenerateOutputInformation();

  // Displaced detector weighting
  typedef itk::DisplacedDetectorImageFilter< OutputImageType > DDFType;
  DDFType::Pointer ddf = DDFType::New();
  ddf->SetInput( reader->GetOutput() );
  ddf->SetGeometry( geometryReader->GetOutputObject() );

  // Weight projections according to fdk algorithm
  typedef itk::FDKWeightProjectionFilter< OutputImageType > WeightFilterType;
  WeightFilterType::Pointer weightFilter = WeightFilterType::New();
  weightFilter->SetInput( ddf->GetOutput() );
  weightFilter->SetSourceToDetectorDistance( geometryReader->GetOutputObject()->GetSourceToDetectorDistance() );
  weightFilter->SetInPlace(false); //SR: there seems to be a bug in ITK: incompatibility between InPlace and streaming?

  // Ramp filter
  typedef itk::FFTRampImageFilter<OutputImageType> RampFilterType;
  RampFilterType::Pointer rampFilter = RampFilterType::New();
  rampFilter->SetInput( weightFilter->GetOutput() );
  rampFilter->SetTruncationCorrection(args_info.pad_arg);
  rampFilter->SetHannCutFrequency(args_info.hann_arg);
  
  // Streaming filter
  typedef itk::StreamingImageFilter<OutputImageType, OutputImageType> StreamerType;
  StreamerType::Pointer streamer = StreamerType::New();
  streamer->SetInput( rampFilter->GetOutput() );
  streamer->SetNumberOfStreamDivisions( geometryReader->GetOutputObject()->GetMatrices().size() );

  // Try to do all 2D pre-processing
  itk::TimeProbe streamerProbe;
  if(!args_info.lowmem_flag)
    {
    try
      {
      if(args_info.verbose_flag)
        std::cout << "Weighting and filtering projections... " << std::flush;
      streamerProbe.Start();
      streamer->Update();
      streamerProbe.Stop();
      if(args_info.verbose_flag)
        std::cout << "It took " << streamerProbe.GetMeanTime() << ' ' << readerProbe.GetUnit() << std::endl;
      }
    catch( itk::ExceptionObject & err )
      {
      std::cerr << "ExceptionObject caught !" << std::endl;
      std::cerr << err << std::endl;
      return EXIT_FAILURE;
      }
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
#if CUDA_FOUND
    bpFilter = itk::CudaFDKBackProjectionImageFilter::New();
#else
    std::cerr << "The program has not been compiled with cuda option" << std::endl;
    return EXIT_FAILURE;
#endif
    }
  bpFilter->SetInput( 0, tomography );
  bpFilter->SetUpdateProjectionPerProjection(args_info.lowmem_flag);
  if(args_info.lowmem_flag)
    bpFilter->SetInput( 1, rampFilter->GetOutput() );
  else
    bpFilter->SetInput( 1, streamer->GetOutput() );
  bpFilter->SetGeometry( geometryReader->GetOutputObject() );
  bpFilter->SetInPlace( true );


  // SR: this appears to trigger 2 updates in cuda mode with the lowmem option
  //     and an off-centered geometry. No clue why... Disable this update
  //     until the problem is understood and solved.
  if(!args_info.lowmem_flag)
    {
    if(args_info.verbose_flag)
      std::cout << "Backprojecting using "
                << args_info.hardware_arg
                << "... "  << std::flush;

    itk::TimeProbe bpProbe;
    try {
      bpProbe.Start();
      bpFilter->Update();
      bpProbe.Stop();
    } catch( itk::ExceptionObject & err ) {
      std::cerr << "ExceptionObject caught !" << std::endl;
      std::cerr << err << std::endl;
      return EXIT_FAILURE;
    }

    if(args_info.verbose_flag)
      std::cout << "It took " << bpProbe.GetMeanTime() << ' ' << readerProbe.GetUnit() << std::endl;
    }

  // Write
  typedef itk::ImageFileWriter<  OutputImageType >  WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg );
  writer->SetInput( bpFilter->GetOutput() );

  if(args_info.verbose_flag)
    std::cout << "Writing... " << std::flush;
  itk::TimeProbe writerProbe;
  try {
    writerProbe.Start();
    writer->Update();
    writerProbe.Stop();
  } catch( itk::ExceptionObject & err ) {
    std::cerr << "ExceptionObject caught !" << std::endl;
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
  }

  if(args_info.verbose_flag)
    std::cout << "It took " << writerProbe.GetMeanTime() << ' ' << readerProbe.GetUnit() << std::endl;

  return EXIT_SUCCESS;
}
