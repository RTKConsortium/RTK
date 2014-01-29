// RTK includes
#include "rtkConfiguration.h"
#include <rtkFDKBackProjectionImageFilter.h>
#include <rtkConstantImageSource.h>
#include <rtkThreeDCircularProjectionGeometry.h>
#include <rtkRayEllipsoidIntersectionImageFilter.h>
#include <rtkDisplacedDetectorImageFilter.h>
#include <rtkParkerShortScanImageFilter.h>
#include <rtkFDKConeBeamReconstructionFilter.h>

// ITK includes
#include <itkImageFileWriter.h>
#include <itkStreamingImageFilter.h>

int main(int , char **)
{
  // Defines the image type
  typedef itk::Image< float, 3 > ImageType;

  // Defines the RTK geometry object
  typedef rtk::ThreeDCircularProjectionGeometry GeometryType;
  GeometryType::Pointer geometry = GeometryType::New();

  // Projection matrices
  unsigned int numberOfProjections = 360;
  unsigned int firstAngle = 0;
  unsigned int angularArc = 360;
  unsigned int sid = 600; // source to isocenter distance in mm
  unsigned int sdd = 1200; // source to detector distance in mm
  int isox = 0; // X coordinate on the projection image of isocenter
  int isoy = 0; // Y coordinate on the projection image of isocenter

  for(unsigned int noProj=0; noProj<numberOfProjections; noProj++)
    {
    double angle = (float)firstAngle + (float)noProj * angularArc / (float)numberOfProjections;
    geometry->AddProjection(sid,
                            sdd,
                            angle,
                            isox,
                            isoy);
    }

  // Create a stack of empty projection images
  typedef rtk::ConstantImageSource< ImageType > ConstantImageSourceType;
  ConstantImageSourceType::Pointer constantImageSource = ConstantImageSourceType::New();
  ConstantImageSourceType::PointType origin;
  ConstantImageSourceType::SpacingType spacing;
  ConstantImageSourceType::SizeType sizeOutput;

  origin[0] = -127.;
  origin[1] = -127.;
  origin[2] = -127.;

  // Adjust size according to geometry
  sizeOutput[0] = 256;
  sizeOutput[1] = 256;
  sizeOutput[2] = numberOfProjections;

  spacing[0] = 1.;
  spacing[1] = 1.;
  spacing[2] = 1.;
  constantImageSource->SetOrigin( origin );
  constantImageSource->SetSpacing( spacing );
  constantImageSource->SetSize( sizeOutput );
  constantImageSource->SetConstant( 0. );
  
   // Create the projector
  typedef rtk::RayEllipsoidIntersectionImageFilter<ImageType, ImageType> REIType;
  REIType::Pointer rei = REIType::New();
  REIType::VectorType semiprincipalaxis, center;
  semiprincipalaxis.Fill(50.);
  center.Fill(0.);
  //Set GrayScale value, axes, center...
  rei->SetDensity(2.);
  rei->SetAngle(0.);
  rei->SetCenter(center);
  rei->SetAxis(semiprincipalaxis);
  rei->SetGeometry( geometry );
  rei->SetInput( constantImageSource->GetOutput() );
  rei->Update();

  // Create reconstructed image
  ConstantImageSourceType::Pointer constantImageSource2 = ConstantImageSourceType::New();

  // Adjust size according to geometry
  sizeOutput[0] = 256;
  sizeOutput[1] = 256;
  sizeOutput[2] = 256;

  constantImageSource2->SetOrigin( origin );
  constantImageSource2->SetSpacing( spacing );
  constantImageSource2->SetSize( sizeOutput );
  constantImageSource2->SetConstant( 0. );

  std::cout << "Performing reconstruction" << std::endl;

  // FDK reconstruction filtering
  typedef rtk::FDKConeBeamReconstructionFilter< ImageType > FDKCPUType;
  FDKCPUType::Pointer feldkamp = FDKCPUType::New();
  feldkamp->SetInput( 0, constantImageSource2->GetOutput() );
  feldkamp->SetInput( 1, rei->GetOutput() );
  feldkamp->SetGeometry( geometry );
  feldkamp->GetRampFilter()->SetTruncationCorrection(0.);
  feldkamp->GetRampFilter()->SetHannCutFrequency(0.0);
  feldkamp->Update();

  std::cout << "Writing output image" << std::endl;

  // Writer
  typedef itk::ImageFileWriter<ImageType> WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( "output.mha" );
  writer->SetUseCompression(true);
  writer->SetInput( feldkamp->GetOutput() );
  writer->Update();

  std::cout << "Done" << std::endl;

  return 0;
}

