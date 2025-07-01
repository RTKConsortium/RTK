// RTK includes
#include <rtkConstantImageSource.h>
#include <rtkThreeDCircularProjectionGeometryXMLFileWriter.h>
#include <rtkRayEllipsoidIntersectionImageFilter.h>
#include <rtkFDKConeBeamReconstructionFilter.h>
#include <rtkFieldOfViewImageFilter.h>

// ITK includes
#include <itkImageFileWriter.h>

int
main(int argc, char ** argv)
{
  if (argc < 3)
  {
    std::cout << "Usage: FirstReconstruction <outputimage> <outputgeometry>" << std::endl;
    return EXIT_FAILURE;
  }

  // Defines the image type
  using ImageType = itk::Image<float, 3>;

  // Defines the RTK geometry object
  using GeometryType = rtk::ThreeDCircularProjectionGeometry;
  auto         geometry = GeometryType::New();
  unsigned int numberOfProjections = 360;
  double       firstAngle = 0;
  double       angularArc = 360;
  unsigned int sid = 600;  // source to isocenter distance
  unsigned int sdd = 1200; // source to detector distance
  for (unsigned int noProj = 0; noProj < numberOfProjections; noProj++)
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
  using ConstantImageSourceType = rtk::ConstantImageSource<ImageType>;
  auto constantImageSource = ConstantImageSourceType::New();
  constantImageSource->SetOrigin(itk::MakePoint(-127, -127, 0.));
  constantImageSource->SetSpacing(itk::MakeVector(2., 2., 2.));
  constantImageSource->SetSize(itk::MakeSize(128, 128, numberOfProjections));

  // Create projections of an ellipse
  using REIType = rtk::RayEllipsoidIntersectionImageFilter<ImageType, ImageType>;
  auto rei = REIType::New();
  rei->SetDensity(2.);
  rei->SetAngle(0.);
  rei->SetCenter(itk::MakePoint(0., 0., 10.));
  rei->SetAxis(itk::MakeVector(50., 50., 50.));
  rei->SetGeometry(geometry);
  rei->SetInput(constantImageSource->GetOutput());

  // Create reconstructed image
  auto constantImageSource2 = ConstantImageSourceType::New();
  constantImageSource2->SetOrigin(itk::MakePoint(-63.5, -63.5, -63.5));
  constantImageSource2->SetSpacing(itk::MakeVector(1., 1., 1.));
  constantImageSource2->SetSize(itk::MakeSize(128, 128, 128));
  constantImageSource2->SetConstant(0.);

  // FDK reconstruction
  std::cout << "Reconstructing..." << std::endl;
  using FDKCPUType = rtk::FDKConeBeamReconstructionFilter<ImageType>;
  auto feldkamp = FDKCPUType::New();
  feldkamp->SetInput(0, constantImageSource2->GetOutput());
  feldkamp->SetInput(1, rei->GetOutput());
  feldkamp->SetGeometry(geometry);
  feldkamp->GetRampFilter()->SetTruncationCorrection(0.);
  feldkamp->GetRampFilter()->SetHannCutFrequency(0.0);

  // Field-of-view masking
  using FOVFilterType = rtk::FieldOfViewImageFilter<ImageType, ImageType>;
  auto fieldofview = FOVFilterType::New();
  fieldofview->SetInput(0, feldkamp->GetOutput());
  fieldofview->SetProjectionsStack(rei->GetOutput());
  fieldofview->SetGeometry(geometry);

  // Writer
  std::cout << "Writing output image..." << std::endl;
  using WriterType = itk::ImageFileWriter<ImageType>;
  auto writer = WriterType::New();
  writer->SetFileName(argv[1]);
  writer->SetInput(fieldofview->GetOutput());
  writer->Update();

  std::cout << "Done!" << std::endl;
  return EXIT_SUCCESS;
}
