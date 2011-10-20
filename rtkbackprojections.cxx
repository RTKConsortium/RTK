#include "rtkbackprojections_ggo.h"
#include "rtkGgoFunctions.h"

#include "itkProjectionsReader.h"
#include "itkThreeDCircularProjectionGeometryXMLFile.h"
#include "itkFDKBackProjectionImageFilter.h"

#include <itkRegularExpressionSeriesFileNames.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkTimeProbe.h>

int main(int argc, char * argv[])
{
  GGO(rtkbackprojections, args_info);

  typedef float OutputPixelType;
  const unsigned int Dimension = 3;

  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;

  // Geometry
  if(args_info.verbose_flag)
    std::cout << "Reading geometry information from "
              << args_info.geometry_arg
              << "..."
              << std::flush;
  itk::ThreeDCircularProjectionGeometryXMLFileReader::Pointer geometryReader;
  geometryReader = itk::ThreeDCircularProjectionGeometryXMLFileReader::New();
  geometryReader->SetFilename(args_info.geometry_arg);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( geometryReader->GenerateOutputInformation() )
  if(args_info.verbose_flag)
    std::cout << " done." << std::endl;

  // Create an empty volume
  typedef itk::ConstantImageSource< OutputImageType > ConstantImageSourceType;
  ConstantImageSourceType::Pointer constantImageSource = ConstantImageSourceType::New();
  rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkbackprojections>(constantImageSource, args_info);

  // Generate file names
  itk::RegularExpressionSeriesFileNames::Pointer names = itk::RegularExpressionSeriesFileNames::New();
  names->SetDirectory(args_info.path_arg);
  names->SetNumericSort(false);
  names->SetRegularExpression(args_info.regexp_arg);
  names->SetSubMatch(0);

  if(args_info.verbose_flag)
    std::cout << "Reading "
              << names->GetFileNames().size()
              << " projection file(s)..."
              << std::flush;

  // Projections reader
  itk::TimeProbe readerProbe;
  readerProbe.Start();
  typedef itk::ProjectionsReader< OutputImageType > ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileNames( names->GetFileNames() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( reader->Update() );
  readerProbe.Stop();
  if(args_info.verbose_flag)
    std::cout << " done in "
              << readerProbe.GetMeanTime() << ' ' << readerProbe.GetUnit()
              << '.' << std::endl;

  // Create back projection image filter
  if(args_info.verbose_flag)
    std::cout << "Backprojecting volume..." << std::flush;
  itk::TimeProbe bpProbe;
  itk::BackProjectionImageFilter<OutputImageType, OutputImageType>::Pointer bp;
  switch(args_info.method_arg)
  {
  case(method_arg_VoxelBasedBackProjection):
    bp = itk::BackProjectionImageFilter<OutputImageType, OutputImageType>::New();
    break;
  case(method_arg_FDKBackProjection):
    bp = itk::FDKBackProjectionImageFilter<OutputImageType, OutputImageType>::New();
    break;
  default:
    std::cerr << "Unhandled --method value." << std::endl;
    return EXIT_FAILURE;
  }
  bp->SetInput( constantImageSource->GetOutput() );
  bp->SetInput( 1, reader->GetOutput() );
  bp->SetGeometry( geometryReader->GetOutputObject() );
  bpProbe.Start();
  TRY_AND_EXIT_ON_ITK_EXCEPTION( bp->Update() )
  bpProbe.Stop();
  if(args_info.verbose_flag)
    std::cout << " done in "
              << bpProbe.GetMeanTime() << ' ' << bpProbe.GetUnit()
              << '.' << std::endl;

  // Write
  if(args_info.verbose_flag)
    std::cout << "Writing... " << std::flush;
  itk::TimeProbe writeProbe;
  typedef itk::ImageFileWriter<  OutputImageType > WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg );
  writer->SetInput( bp->GetOutput() );
  writeProbe.Start();
  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() );
  writeProbe.Stop();
  if(args_info.verbose_flag)
    std::cout << " done in "
              << writeProbe.GetMeanTime() << ' ' << writeProbe.GetUnit()
              << '.' << std::endl;

  return EXIT_SUCCESS;
}
