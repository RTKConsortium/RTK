#include "rtkprojections_ggo.h"
#include "rtkMacro.h"

#include "rtkProjectionsReader.h"

#include <itkImageFileWriter.h>
#include <itkRegularExpressionSeriesFileNames.h>
#include <itkTimeProbe.h>

int main(int argc, char * argv[])
{
  GGO(rtkprojections, args_info);

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
  typedef rtk::ProjectionsReader< OutputImageType > ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileNames( names->GetFileNames() );

  // Write
  typedef itk::ImageFileWriter<  OutputImageType > WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg );
  writer->SetInput( reader->GetOutput() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->UpdateOutputInformation() )
  writer->SetNumberOfStreamDivisions( 1 + reader->GetOutput()->GetLargestPossibleRegion().GetNumberOfPixels() / (1024*1024*4) );

  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() )

  return EXIT_SUCCESS;
}
