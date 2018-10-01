// RTK includes
#include <rtkConstantImageSource.h>
#include <rtkThreeDCircularProjectionGeometryXMLFileWriter.h>
#include <rtkRayEllipsoidIntersectionImageFilter.h>
#include <rtkFDKConeBeamReconstructionFilter.h>
#include <rtkFieldOfViewImageFilter.h>

// ITK includes
#include <itkImageFileWriter.h>

int main(int argc, char **argv)
{
  if(argc<3)
    {
    std::cout << "Usage: FirstReconstruction <outputimage> <outputgeometry>" << std::endl;
    return EXIT_FAILURE;
    }

  // Defines the image type
  typedef itk::Image< float, 3 > ImageType;

  // Defines the RTK geometry object
  typedef rtk::ThreeDCircularProjectionGeometry GeometryType;
  GeometryType::Pointer geometry = GeometryType::New();
  unsigned int numberOfProjections = 360;
  double firstAngle = 0;
  double angularArc = 360;
  unsigned int sid = 600; // source to isocenter distance
  unsigned int sdd = 1200; // source to detector distance
  for(unsigned int noProj=0; noProj<numberOfProjections; noProj++)
    {
    double angle = firstAngle + noProj * angularArc / numberOfProjections;
    geometry->AddProjection(sid, sdd, angle);
    }

  // Write the geometry to disk
  rtk::ThreeDCircularProjectionGeometryXMLFileWriter::Pointer xmlWriter;
  xmlWriter = rtk::ThreeDCircularProjectionGeometryXMLFileWriter::New();
  xmlWriter->SetFilename(argv[2]);
  xmlWriter->SetObject(geometry);
  xmlWriter->WriteFile();

  // Create a stack of empty projection images
  typedef rtk::ConstantImageSource< ImageType > ConstantImageSourceType;
  ConstantImageSourceType::Pointer constantImageSource = ConstantImageSourceType::New();
  ConstantImageSourceType::PointType origin;
  ConstantImageSourceType::SpacingType spacing;
  ConstantImageSourceType::SizeType sizeOutput;

  origin[0] = -127;
  origin[1] = -127;
  origin[2] = 0.;

  sizeOutput[0] = 128;
  sizeOutput[1] = 128;
  sizeOutput[2] = numberOfProjections;

  spacing.Fill(2.);

  constantImageSource->SetOrigin(origin);
  constantImageSource->SetSpacing(spacing);
  constantImageSource->SetSize(sizeOutput);
  constantImageSource->SetConstant(0.);

   // Create projections of an ellipse
  typedef rtk::RayEllipsoidIntersectionImageFilter<ImageType, ImageType> REIType;
  REIType::Pointer rei = REIType::New();
  REIType::VectorType semiprincipalaxis, center;
  semiprincipalaxis.Fill(50.);
  center.Fill(0.);
  center[2]=10.;
  rei->SetDensity(2.);
  rei->SetAngle(0.);
  rei->SetCenter(center);
  rei->SetAxis(semiprincipalaxis);
  rei->SetGeometry(geometry);
  rei->SetInput(constantImageSource->GetOutput());

  // Create reconstructed image
  ConstantImageSourceType::Pointer constantImageSource2 = ConstantImageSourceType::New();
  sizeOutput.Fill(128);
  origin.Fill(-63.5);
  spacing.Fill(1.);
  constantImageSource2->SetOrigin(origin);
  constantImageSource2->SetSpacing(spacing);
  constantImageSource2->SetSize(sizeOutput);
  constantImageSource2->SetConstant(0.);

  // FDK reconstruction
  std::cout << "Reconstructing..." << std::endl;
  typedef rtk::FDKConeBeamReconstructionFilter< ImageType > FDKCPUType;
  FDKCPUType::Pointer feldkamp = FDKCPUType::New();
  feldkamp->SetInput(0, constantImageSource2->GetOutput());
  feldkamp->SetInput(1, rei->GetOutput());
  feldkamp->SetGeometry(geometry);
  feldkamp->GetRampFilter()->SetTruncationCorrection(0.);
  feldkamp->GetRampFilter()->SetHannCutFrequency(0.0);

  // Field-of-view masking
  typedef rtk::FieldOfViewImageFilter<ImageType, ImageType> FOVFilterType;
  FOVFilterType::Pointer fieldofview = FOVFilterType::New();
  fieldofview->SetInput(0, feldkamp->GetOutput());
  fieldofview->SetProjectionsStack(rei->GetOutput());
  fieldofview->SetGeometry(geometry);

  // Writer
  std::cout << "Writing output image..." << std::endl;
  typedef itk::ImageFileWriter<ImageType> WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName(argv[1]);
  writer->SetInput(fieldofview->GetOutput());
  writer->Update();

  std::cout << "Done!" << std::endl;
  return EXIT_SUCCESS;
}

