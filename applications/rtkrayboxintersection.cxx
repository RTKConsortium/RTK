#include "rtkrayboxintersection_ggo.h"
#include "rtkGgoFunctions.h"

#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkRayBoxIntersectionImageFilter.h"

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

int main(int argc, char * argv[])
{
  GGO(rtkrayboxintersection, args_info);

  typedef float OutputPixelType;
  const unsigned int Dimension = 3;

  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;

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

  // Create a stack of empty projection images
  typedef rtk::ConstantImageSource< OutputImageType > ConstantImageSourceType;
  ConstantImageSourceType::Pointer constantImageSource = ConstantImageSourceType::New();
  rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkrayboxintersection>(constantImageSource, args_info);

  // Adjust size according to geometry
  ConstantImageSourceType::SizeType sizeOutput;
  sizeOutput[0] = constantImageSource->GetSize()[0];
  sizeOutput[1] = constantImageSource->GetSize()[1];
  sizeOutput[2] = geometryReader->GetOutputObject()->GetGantryAngles().size();
  constantImageSource->SetSize( sizeOutput );

  // Input reader
  typedef itk::ImageFileReader<  OutputImageType > ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName( args_info.input_arg );
  reader->UpdateOutputInformation();

  // Create projection image filter
  typedef rtk::RayBoxIntersectionImageFilter<OutputImageType, OutputImageType> RBIType;
  RBIType::Pointer rbi = RBIType::New();
  rbi->SetInput( constantImageSource->GetOutput() );
  rbi->SetBoxFromImage( reader->GetOutput() );
  rbi->SetGeometry( geometryReader->GetOutputObject() );
  rbi->Update();

  // Write
  typedef itk::ImageFileWriter<  OutputImageType > WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg );
  writer->SetInput( rbi->GetOutput() );
  if(args_info.verbose_flag)
    std::cout << "Projecting and writing... " << std::flush;
  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() );

  return EXIT_SUCCESS;
}
