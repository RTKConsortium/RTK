#include "rtkrayellipsoidintersection_ggo.h"
#include "rtkGgoFunctions.h"

#include "itkThreeDCircularProjectionGeometryXMLFile.h"
#include "itkRayEllipsoidIntersectionImageFilter.h"

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

int main(int argc, char * argv[])
{
  GGO(rtkrayellipsoidintersection, args_info);

  typedef float OutputPixelType;
  const unsigned int Dimension = 3;

  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;

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

  // Create a stack of empty projection images
  typedef itk::ConstantImageSource< OutputImageType > ConstantImageSourceType;
  ConstantImageSourceType::Pointer constantImageSource = ConstantImageSourceType::New();
  rtk::SetConstantImageSourceFromGgo<ConstantImageSourceType, args_info_rtkrayellipsoidintersection>(constantImageSource, args_info);

  // Adjust size according to geometry
  ConstantImageSourceType::SizeType sizeOutput;
  sizeOutput[0] = constantImageSource->GetSize()[0];
  sizeOutput[1] = constantImageSource->GetSize()[1];
  sizeOutput[2] = geometryReader->GetOutputObject()->GetGantryAngles().size();
  constantImageSource->SetSize( sizeOutput );

  // Create projection image filter
  typedef itk::RayEllipsoidIntersectionImageFilter<OutputImageType, OutputImageType> REIType;
  REIType::Pointer rei = REIType::New();

  rei->SetMultiplicativeConstant(args_info.mult_arg);
  if(args_info.axes_given>0) rei->SetSemiPrincipalAxisX(args_info.axes_arg[0]);
  if(args_info.axes_given>1) rei->SetSemiPrincipalAxisY(args_info.axes_arg[1]);
  if(args_info.axes_given>2) rei->SetSemiPrincipalAxisZ(args_info.axes_arg[2]);
  if(args_info.center_given>0) rei->SetCenterX(args_info.center_arg[0]);
  if(args_info.center_given>1) rei->SetCenterY(args_info.center_arg[1]);
  if(args_info.center_given>2) rei->SetCenterZ(args_info.center_arg[2]);
  if(args_info.rotation_given>0)
  {
    rei->SetRotate(true);
    rei->SetRotationAngle(args_info.rotation_arg[0]);
  }

  rei->SetInput( constantImageSource->GetOutput() );
  rei->SetGeometry( geometryReader->GetOutputObject() );
  rei->Update();

  // Write
  typedef itk::ImageFileWriter<  OutputImageType > WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( args_info.output_arg );
  writer->SetInput( rei->GetOutput() );
  if(args_info.verbose_flag)
    std::cout << "Projecting and writing... " << std::flush;
  TRY_AND_EXIT_ON_ITK_EXCEPTION( writer->Update() );

  return EXIT_SUCCESS;
}
