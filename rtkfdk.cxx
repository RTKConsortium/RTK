#include "rtkfdk_ggo.h"
#include "rtkGgoFunctions.h"
#include "rtkConfiguration.h"

#include "itkThreeDCircularProjectionGeometryXMLFile.h"
#include "itkProjectionsReader.h"
#include "itkDisplacedDetectorImageFilter.h"
#include "itkParkerShortScanImageFilter.h"
#include "itkFDKConeBeamReconstructionFilter.h"
#if CUDA_FOUND
# include "itkCudaFDKConeBeamReconstructionFilter.h"
#endif

#include <itkRegularExpressionSeriesFileNames.h>
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
  TRY_AND_EXIT_ON_ITK_EXCEPTION( reader->GenerateOutputInformation() );

  itk::TimeProbe readerProbe;
  if(!args_info.lowmem_flag)
    {
    if(args_info.verbose_flag)
      std::cout << "Reading... " << std::flush;
    readerProbe.Start();
    TRY_AND_EXIT_ON_ITK_EXCEPTION( reader->Update() )
    readerProbe.Stop();
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
  TRY_AND_EXIT_ON_ITK_EXCEPTION( geometryReader->GenerateOutputInformation() )

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

  // Create reconstructed image
  typedef itk::ConstantImageSource< OutputImageType > ConstantImageSourceType;
  ConstantImageSourceType::Pointer constantImageSource = ConstantImageSourceType::New();
  rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkfdk>(constantImageSource, args_info);

  // This macro sets options for fdk filter which I can not see how to do better
  // because TFFTPrecision is not the same, e.g. for CPU and CUDA (SR)
#define SET_FELDKAMP_OPTIONS(f) \
    f->SetInput( 0, constantImageSource->GetOutput() ); \
    f->SetInput( 1, reader->GetOutput() ); \
    f->SetGeometry( geometryReader->GetOutputObject() ); \
    f->GetRampFilter()->SetTruncationCorrection(args_info.pad_arg); \
    f->GetRampFilter()->SetHannCutFrequency(args_info.hann_arg);

  // FDK reconstruction filtering
  itk::ImageToImageFilter<OutputImageType, OutputImageType>::Pointer feldkamp;
  typedef itk::FDKConeBeamReconstructionFilter< OutputImageType > FDKCPUType;
  typedef itk::CudaFDKConeBeamReconstructionFilter                FDKCUDAType;
  if(!strcmp(args_info.hardware_arg, "cpu") )
    {
    feldkamp = FDKCPUType::New();
    SET_FELDKAMP_OPTIONS( static_cast<FDKCPUType*>(feldkamp.GetPointer()) );
    }
  else if(!strcmp(args_info.hardware_arg, "cuda") )
    {
#if CUDA_FOUND
    feldkamp = FDKCUDAType::New();
    SET_FELDKAMP_OPTIONS( static_cast<FDKCUDAType*>(feldkamp.GetPointer()) );
#else
    std::cerr << "The program has not been compiled with cuda option" << std::endl;
    return EXIT_FAILURE;
#endif
    }

  // Streaming depending on streaming capability of writer
  typedef itk::StreamingImageFilter<OutputImageType, OutputImageType> StreamerType;
  StreamerType::Pointer streamerBP = StreamerType::New();
  itk::ImageIOBase::Pointer imageIOBase;
  streamerBP->SetInput( feldkamp->GetOutput() );
  streamerBP->SetNumberOfStreamDivisions( args_info.divisions_arg );

  // Write
  typedef itk::ImageFileWriter<  OutputImageType > WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg );
  writer->SetInput( streamerBP->GetOutput() );

  if(args_info.verbose_flag)
    std::cout << "Reconstructing and writing... " << std::flush;
  itk::TimeProbe writerProbe;

  writerProbe.Start();
  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() );
  writerProbe.Stop();

  if(args_info.verbose_flag)
    {
    std::cout << "It took " << writerProbe.GetMeanTime() << ' ' << readerProbe.GetUnit() << std::endl;
    if(!strcmp(args_info.hardware_arg, "cpu") )
      static_cast<FDKCPUType* >(feldkamp.GetPointer())->PrintTiming(std::cout);
    else if(!strcmp(args_info.hardware_arg, "cuda") )
      static_cast<FDKCUDAType*>(feldkamp.GetPointer())->PrintTiming(std::cout);
    }

  return EXIT_SUCCESS;
}
