#include "rtkamsterdamshroud_ggo.h"
#include "rtkMacro.h"

#include "rtkProjectionsReader.h"
#include "rtkAmsterdamShroudImageFilter.h"

#include <itkImageFileWriter.h>
#include <itkRegularExpressionSeriesFileNames.h>
#include <itkTimeProbe.h>

int main(int argc, char * argv[])
{
  GGO(rtkamsterdamshroud, args_info);

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
  typedef rtk::ProjectionsReader< OutputImageType > ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileNames( names->GetFileNames() );
  
  // Amsterdam shroud
  typedef rtk::AmsterdamShroudImageFilter<OutputImageType> shroudFilterType;
  shroudFilterType::Pointer shroudFilter = shroudFilterType::New();
  shroudFilter->SetInput( reader->GetOutput() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( shroudFilter->UpdateOutputInformation() )

  // Write
  typedef itk::ImageFileWriter< shroudFilterType::OutputImageType > WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg );
  writer->SetInput( shroudFilter->GetOutput() );
  writer->SetNumberOfStreamDivisions(names->GetFileNames().size());

  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() )

  return EXIT_SUCCESS;
}
