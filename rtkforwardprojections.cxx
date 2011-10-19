#include "rtkforwardprojections_ggo.h"
#include "rtkGgoFunctions.h"

#include "itkThreeDCircularProjectionGeometryXMLFile.h"
#include "itkJosephForwardProjectionImageFilter.h"
#include "itkRayCastInterpolatorForwardProjectionImageFilter.h"

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkTimeProbe.h>

int main(int argc, char * argv[])
{
  GGO(rtkforwardprojections, args_info);

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

  // Create a stack of empty projection images
  typedef itk::ConstantImageSource< OutputImageType > ConstantImageSourceType;
  ConstantImageSourceType::Pointer constantImageSource = ConstantImageSourceType::New();
  rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkforwardprojections>(constantImageSource, args_info);

  // Adjust size according to geometry
  ConstantImageSourceType::SizeType sizeOutput;
  sizeOutput[0] = constantImageSource->GetSize()[0];
  sizeOutput[1] = constantImageSource->GetSize()[1];
  sizeOutput[2] = geometryReader->GetOutputObject()->GetGantryAngles().size();
  constantImageSource->SetSize( sizeOutput );

  // Input reader
  if(args_info.verbose_flag)
    std::cout << "Reading input volume "
              << args_info.input_arg
              << "..."
              << std::flush;
  itk::TimeProbe readerProbe;
  typedef itk::ImageFileReader<  OutputImageType > ReaderType;
  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName( args_info.input_arg );
  readerProbe.Start();
  TRY_AND_EXIT_ON_ITK_EXCEPTION( reader->Update() )
  readerProbe.Stop();
  if(args_info.verbose_flag)
    std::cout << " done in "
              << readerProbe.GetMeanTime() << ' ' << readerProbe.GetUnit()
              << '.' << std::endl;

  // Create forward projection image filter
  if(args_info.verbose_flag)
    std::cout << "Projecting volume..." << std::flush;
  itk::TimeProbe projProbe;
  itk::ForwardProjectionImageFilter<OutputImageType, OutputImageType>::Pointer forwardProjection;
  switch(args_info.method_arg)
  {
  case(method_arg_Joseph):
    forwardProjection = itk::JosephForwardProjectionImageFilter<OutputImageType, OutputImageType>::New();
    break;
  case(method_arg_RayCastInterpolator):
    forwardProjection = itk::RayCastInterpolatorForwardProjectionImageFilter<OutputImageType, OutputImageType>::New();
    break;
  }
  forwardProjection->SetInput( constantImageSource->GetOutput() );
  forwardProjection->SetInput( 1, reader->GetOutput() );
  forwardProjection->SetGeometry( geometryReader->GetOutputObject() );
  projProbe.Start();
  TRY_AND_EXIT_ON_ITK_EXCEPTION( forwardProjection->Update() )
  projProbe.Stop();
  if(args_info.verbose_flag)
    std::cout << " done in "
              << projProbe.GetMeanTime() << ' ' << projProbe.GetUnit()
              << '.' << std::endl;

  // Write
  typedef itk::ImageFileWriter<  OutputImageType > WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg );
  writer->SetInput( forwardProjection->GetOutput() );
  if(args_info.verbose_flag)
    std::cout << "Projecting and writing... " << std::flush;
  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() );

  return EXIT_SUCCESS;
}
