#include "rtkDrawGeometricPhantomImageFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkProjectGeometricPhantomImageFilter.h"
#include "rtkFDKConeBeamReconstructionFilter.h"
#include "rtkThreeDCircularProjectionGeometryXMLFileWriter.h"

int
main(int argc, char * argv[])
{
  if (argc < 2)
  {
    std::cout << "Usage: GeometricPhantom <phantomfile>" << std::endl;
    return EXIT_FAILURE;
  }

  constexpr unsigned int Dimension = 3;
  using OutputPixelType = float;
  using OutputImageType = itk::Image<OutputPixelType, Dimension>;

  constexpr unsigned int numberOfProjections = 60;
  constexpr double       angularArc = 360.;
  constexpr unsigned int sid = 600;  // source to isocenter distance
  constexpr unsigned int sdd = 1200; // source to detector distance
  constexpr double       scale = 2.;

  itk::Matrix<double, Dimension, Dimension> rotation;
  rotation[0][0] = 1.;
  rotation[1][2] = 1.;
  rotation[2][1] = 1.;

  // Set up the geometry for the projections
  auto geometry = rtk::ThreeDCircularProjectionGeometry::New();
  for (unsigned int x = 0; x < numberOfProjections; x++)
  {
    geometry->AddProjection(sid, sdd, x * angularArc / numberOfProjections);
  }
  rtk::WriteGeometry(geometry, "geometry.xml");

  // Create a constant image source for the tomography
  auto tomographySource = rtk::ConstantImageSource<OutputImageType>::New();
  tomographySource->SetOrigin(itk::MakePoint(-63, -63, -63));
  tomographySource->SetSpacing(itk::MakeVector(2., 2., 2.));
  tomographySource->SetSize(itk::MakeSize(64, 64, 64));

  // Create a constant image source for the projections
  auto projectionsSource = rtk::ConstantImageSource<OutputImageType>::New();
  projectionsSource->SetOrigin(itk::MakePoint(-63., -63., -63.));
  projectionsSource->SetSpacing(itk::MakeVector(2., 2., 2.));
  projectionsSource->SetSize(itk::MakeSize(64, 64, numberOfProjections));

  // Project the geometric phantom image
  auto pgp = rtk::ProjectGeometricPhantomImageFilter<OutputImageType, OutputImageType>::New();
  pgp->SetInput(projectionsSource->GetOutput());
  pgp->SetGeometry(geometry);
  pgp->SetPhantomScale(scale);
  pgp->SetRotationMatrix(rotation);
  pgp->SetConfigFile(argv[1]);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(itk::WriteImage(pgp->GetOutput(), "projections.mha"));

  // Draw the geometric phantom image
  auto dgp = rtk::DrawGeometricPhantomImageFilter<OutputImageType, OutputImageType>::New();
  dgp->SetInput(tomographySource->GetOutput());
  dgp->SetPhantomScale(scale);
  dgp->SetRotationMatrix(rotation);
  dgp->SetConfigFile(argv[1]);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(itk::WriteImage(dgp->GetOutput(), "ref.mha"));

  // Perform FDK reconstruction filtering
  auto feldkamp = rtk::FDKConeBeamReconstructionFilter<OutputImageType>::New();
  feldkamp->SetInput(0, tomographySource->GetOutput());
  feldkamp->SetInput(1, pgp->GetOutput());
  feldkamp->SetGeometry(geometry);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(itk::WriteImage(feldkamp->GetOutput(), "fdk.mha"));

  return EXIT_SUCCESS;
}
