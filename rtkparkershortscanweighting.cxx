#include "rtkparkershortscanweighting_ggo.h"
#include "rtkGgoFunctions.h"

#include "itkThreeDCircularProjectionGeometryXMLFile.h"
#include "itkProjectionsReader.h"
#include "itkParkerShortScanImageFilter.h"

#include <itkImageFileWriter.h>
#include <itkRegularExpressionSeriesFileNames.h>

int main(int argc, char * argv[])
{
  GGO(rtkparkershortscanweighting, args_info);

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

  // Short scan image filter
  typedef itk::ParkerShortScanImageFilter< OutputImageType > PSSFType;
  PSSFType::Pointer pssf = PSSFType::New();
  pssf->SetInput( reader->GetOutput() );
  pssf->SetGeometry( geometryReader->GetOutputObject() );
  pssf->SetNumberOfThreads(1);
  pssf->InPlaceOff();

  // Write
  typedef itk::ImageFileWriter<  OutputImageType > WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg );
  writer->SetInput( pssf->GetOutput() );
  writer->SetNumberOfStreamDivisions( args_info.divisions_arg );
  writer->Update();

  return EXIT_SUCCESS;
}
