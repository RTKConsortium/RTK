#include "rtkramp_ggo.h"
#include "rtkMacro.h"

#include "itkProjectionsReader.h"
#include "itkFFTRampImageFilter.h"

#include <itkImageFileWriter.h>
#include <itkRegularExpressionSeriesFileNames.h>
#include <itkStreamingImageFilter.h>
#include <itkTimeProbe.h>

int main(int argc, char * argv[])
{
  GGO(rtkramp, args_info);

  typedef double OutputPixelType;
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

  TRY_AND_EXIT_ON_ITK_EXCEPTION( reader->Update() )

  // Ramp filter
  typedef itk::FFTRampImageFilter<OutputImageType> rampFilterType;
  rampFilterType::Pointer rampFilter = rampFilterType::New();
  rampFilter->SetInput( reader->GetOutput() );
  rampFilter->SetTruncationCorrection(args_info.pad_arg);
  rampFilter->SetHannCutFrequency(args_info.hann_arg);

  // Streaming filter
  typedef itk::StreamingImageFilter<OutputImageType, OutputImageType> StreamerType;
  StreamerType::Pointer streamer = StreamerType::New();
  streamer->SetInput( rampFilter->GetOutput() );
  streamer->SetNumberOfStreamDivisions( 1 + reader->GetOutput()->GetLargestPossibleRegion().GetNumberOfPixels() / (1024*1024*4) );

  itk::TimeProbe probe;
  probe.Start();
  TRY_AND_EXIT_ON_ITK_EXCEPTION( streamer->Update() )
  probe.Stop();
  std::cout << "The streamed ramp filter update took "
            << probe.GetMeanTime()
            << probe.GetUnit()
            << std::endl;

  // Write
  typedef itk::ImageFileWriter<  OutputImageType > WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg );
  writer->SetInput( streamer->GetOutput() );
  writer->UpdateOutputInformation();

  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() )

  return EXIT_SUCCESS;
}
