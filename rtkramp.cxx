#include "rtkramp_ggo.h"
#include "itkProjectionsReader.h"
#include "itkFFTRampImageFilter.h"

#include <itkImageFileWriter.h>
#include <itkRegularExpressionSeriesFileNames.h>
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

  // Ramp filter
  typedef itk::FFTRampImageFilter<OutputImageType> rampFilterType;
  rampFilterType::Pointer rampFilter = rampFilterType::New();
  rampFilter->SetInput( reader->GetOutput() );

  // Write
  typedef itk::ImageFileWriter<  OutputImageType >  WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg );
  writer->SetInput( rampFilter->GetOutput() );
  writer->UpdateOutputInformation();
  writer->SetNumberOfStreamDivisions( 1 + rampFilter->GetOutput()->GetLargestPossibleRegion().GetNumberOfPixels() / (1024*1024*4) );

  try {
    writer->Update();
  } catch( itk::ExceptionObject & err ) {
    std::cerr << "ExceptionObject caught !" << std::endl;
    std::cerr << err << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
