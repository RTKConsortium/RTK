#include "rtksart_ggo.h"
#include "rtkGgoFunctions.h"

#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkProjectionsReader.h"
#include "rtkSARTConeBeamReconstructionFilter.h"

#include <itkRegularExpressionSeriesFileNames.h>
#include <itkImageFileWriter.h>

int main(int argc, char * argv[])
{
  GGO(rtksart, args_info);

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
  typedef rtk::ProjectionsReader< OutputImageType > ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileNames( names->GetFileNames() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( reader->GenerateOutputInformation() );

  // Geometry
  if(args_info.verbose_flag)
    std::cout << "Reading geometry information from "
              << args_info.geometry_arg
              << "..."
              << std::endl;
  rtk::ThreeDCircularProjectionGeometryXMLFileReader::Pointer geometryReader;
  geometryReader = rtk::ThreeDCircularProjectionGeometryXMLFileReader::New();
  geometryReader->SetFilename(args_info.geometry_arg);
  TRY_AND_EXIT_ON_ITK_EXCEPTION( geometryReader->GenerateOutputInformation() )

  // Create input: either an existing volume read from a file or a blank image
  itk::ImageSource< OutputImageType >::Pointer inputFilter;
  if(args_info.input_given)
    {
    // Read an existing image to initialize the volume
    typedef itk::ImageFileReader<  OutputImageType > InputReaderType;
    InputReaderType::Pointer inputReader = InputReaderType::New();
    inputReader->SetFileName( args_info.input_arg );
    inputFilter = inputReader;
    }
  else
    {
    // Create new empty volume
    typedef rtk::ConstantImageSource< OutputImageType > ConstantImageSourceType;
    ConstantImageSourceType::Pointer constantImageSource = ConstantImageSourceType::New();
    rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtksart>(constantImageSource, args_info);
    inputFilter = constantImageSource;
    }

  // Construct selected backprojection filter
  rtk::BackProjectionImageFilter<OutputImageType, OutputImageType>::Pointer bp;
  switch(args_info.bp_arg)
  {
  case(bp_arg_VoxelBasedBackProjection):
    bp = rtk::BackProjectionImageFilter<OutputImageType, OutputImageType>::New();
    break;
  case(bp_arg_Joseph):
    bp = rtk::JosephBackProjectionImageFilter<OutputImageType, OutputImageType>::New();
    break;
  default:
    std::cerr << "Unhandled --bp value." << std::endl;
    return EXIT_FAILURE;
  }

  // SART reconstruction filter
  typedef rtk::SARTConeBeamReconstructionFilter< OutputImageType > SARTType;
  SARTType::Pointer sart = SARTType::New();
  sart->SetInput( inputFilter->GetOutput() );
  sart->SetInput(1, reader->GetOutput());
  sart->SetGeometry( geometryReader->GetOutputObject() );
  sart->SetNumberOfIterations( args_info.niterations_arg );
  sart->SetLambda( args_info.lambda_arg );
  sart->SetBackProjectionFilter( bp );
  sart->Update();

  // Write
  typedef itk::ImageFileWriter<  OutputImageType > WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg );
  writer->SetInput( sart->GetOutput() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() );

  return EXIT_SUCCESS;
}
