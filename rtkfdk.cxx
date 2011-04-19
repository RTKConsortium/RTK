#include "rtkfdk_ggo.h"
#include "rtkGgoFunctions.h"
#include "rtkConfiguration.h"

#include "itkThreeDCircularProjectionGeometryXMLFile.h"
#include "itkProjectionsReader.h"
#include "itkDisplacedDetectorImageFilter.h"
#include "itkParkerShortScanImageFilter.h"
#include "itkFDKWeightProjectionFilter.h"

#include "itkFFTRampImageFilter.h"
#include "itkFDKBackProjectionImageFilter.h"
#if CUDA_FOUND
#  include "itkCudaFFTRampImageFilter.h"
#  include "itkCudaFDKBackProjectionImageFilter.h"
#endif

#include <itkRegularExpressionSeriesFileNames.h>
#include <itkTimeProbe.h>
#include <itkStreamingImageFilter.h>
#include <itkImageFileWriter.h>

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
    try
      {
      readerProbe.Start();
      reader->Update();
      readerProbe.Stop();
      }
    catch( itk::ExceptionObject & err )
      {
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
  itk::ThreeDCircularProjectionGeometryXMLFileReader::Pointer geometryReader;
  geometryReader = itk::ThreeDCircularProjectionGeometryXMLFileReader::New();
  geometryReader->SetFilename(args_info.geometry_arg);
  geometryReader->GenerateOutputInformation();

  // Displaced detector weighting
  typedef itk::DisplacedDetectorImageFilter< OutputImageType > DDFType;
  DDFType::Pointer ddf = DDFType::New();
  ddf->SetInput( reader->GetOutput() );
  ddf->SetGeometry( geometryReader->GetOutputObject() );

  // Short scan image filter
  typedef itk::ParkerShortScanImageFilter< OutputImageType > PSSFType;
  PSSFType::Pointer pssf = PSSFType::New();
  pssf->SetInput( ddf->GetOutput() );
  pssf->SetGeometry( geometryReader->GetOutputObject() );
  pssf->InPlaceOff();

  // Weight projections according to fdk algorithm
  typedef itk::FDKWeightProjectionFilter< OutputImageType > WeightFilterType;
  WeightFilterType::Pointer weightFilter = WeightFilterType::New();
  weightFilter->SetInput( pssf->GetOutput() );
  weightFilter->SetGeometry( geometryReader->GetOutputObject() );
  weightFilter->SetInPlace(false); //SR: there seems to be a bug in ITK:
                                   // incompatibility between InPlace and
                                   // streaming?

  // Ramp filter
  itk::ImageToImageFilter<OutputImageType, OutputImageType>::Pointer rampFilter;
  if(!strcmp(args_info.hardware_arg, "cuda") )
    {
#if !CUDA_FOUND
    std::cerr << "The program has not been compiled with cuda option" << std::endl;
    return EXIT_FAILURE;
#else
    typedef itk::CudaFFTRampImageFilter CUDARampFilterType;
    CUDARampFilterType::Pointer cudaRampFilter = CUDARampFilterType::New();
    cudaRampFilter->SetInput( weightFilter->GetOutput() );
    cudaRampFilter->SetTruncationCorrection(args_info.pad_arg);
    cudaRampFilter->SetHannCutFrequency(args_info.hann_arg);
    rampFilter = cudaRampFilter;
#endif
    }
  else
    {
    typedef itk::FFTRampImageFilter<OutputImageType> CPURampFilterType;
    CPURampFilterType::Pointer cpuRampFilter = CPURampFilterType::New();
    cpuRampFilter->SetInput( weightFilter->GetOutput() );
    cpuRampFilter->SetTruncationCorrection(args_info.pad_arg);
    cpuRampFilter->SetHannCutFrequency(args_info.hann_arg);
    rampFilter = cpuRampFilter;
    }

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

  // Create reconstructed image
  typedef itk::ConstantImageSource< OutputImageType > ConstantImageSourceType;
  ConstantImageSourceType::Pointer constantImageSource = ConstantImageSourceType::New();
  rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkfdk>(constantImageSource, args_info);

  // Backprojection
  typedef itk::FDKBackProjectionImageFilter<OutputImageType, OutputImageType> BackProjectionFilterType;
  BackProjectionFilterType::Pointer bpFilter;
  if(!strcmp(args_info.hardware_arg, "cpu") )
    bpFilter = BackProjectionFilterType::New();
  else if(!strcmp(args_info.hardware_arg, "cuda") )
    {
#if CUDA_FOUND
    bpFilter = itk::CudaFDKBackProjectionImageFilter::New();
#else
    std::cerr << "The program has not been compiled with cuda option" << std::endl;
    return EXIT_FAILURE;
#endif
    }
  bpFilter->SetInput( 0, constantImageSource->GetOutput() );
  bpFilter->SetUpdateProjectionPerProjection(args_info.lowmem_flag);
  if(args_info.lowmem_flag)
    bpFilter->SetInput( 1, rampFilter->GetOutput() );
  else
    bpFilter->SetInput( 1, streamer->GetOutput() );
  bpFilter->SetGeometry( geometryReader->GetOutputObject() );
  bpFilter->SetInPlace(false);

  // SR: this appears to trigger 2 updates in cuda mode with the lowmem option
  //     and an off-centered geometry. No clue why... Disable this update
  //     until the problem is understood and solved.
  if(!args_info.lowmem_flag && args_info.divisions_arg==1)
    {
    bpFilter->SetInPlace( true );
    if(args_info.verbose_flag)
      std::cout << "Backprojecting using "
                << args_info.hardware_arg
                << "... "  << std::flush;

    itk::TimeProbe bpProbe;
    try
      {
      bpProbe.Start();
      bpFilter->Update();
      bpProbe.Stop();
      }
    catch( itk::ExceptionObject & err )
      {
      std::cerr << "ExceptionObject caught !" << std::endl;
      std::cerr << err << std::endl;
      return EXIT_FAILURE;
      }

    if(args_info.verbose_flag)
      std::cout << "It took " << bpProbe.GetMeanTime() << ' ' << readerProbe.GetUnit() << std::endl;
    }

  // Streaming depending on streaming capability of writer
  StreamerType::Pointer streamerBP = StreamerType::New();
  itk::ImageIOBase::Pointer imageIOBase;
  streamerBP->SetInput( bpFilter->GetOutput() );
  streamerBP->SetNumberOfStreamDivisions( args_info.divisions_arg );

  // Write
  typedef itk::ImageFileWriter<  OutputImageType > WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg );
  writer->SetInput( streamerBP->GetOutput() );
  
  if(args_info.verbose_flag)
    std::cout << "Writing... " << std::flush;
  itk::TimeProbe writerProbe;
  try
    {
    writerProbe.Start();
    writer->Update();
    writerProbe.Stop();
    }
  catch( itk::ExceptionObject & err )
    {
    std::cerr << "ExceptionObject caught !" << std::endl;
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
    }

  if(args_info.verbose_flag)
    std::cout << "It took " << writerProbe.GetMeanTime() << ' ' << readerProbe.GetUnit() << std::endl;

  return EXIT_SUCCESS;
}
