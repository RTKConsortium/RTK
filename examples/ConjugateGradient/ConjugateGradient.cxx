#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkConjugateGradientConeBeamReconstructionFilter.h"
#include "rtkIterationCommands.h"
#include "rtkProjectGeometricPhantomImageFilter.h"

#ifdef RTK_USE_CUDA
#  include <itkCudaImage.h>
#endif
#include <itkImageFileWriter.h>

int
main(int argc, char * argv[])
{
  using OutputPixelType = float;
  constexpr unsigned int Dimension = 3;

#ifdef RTK_USE_CUDA
  using OutputImageType = itk::CudaImage<OutputPixelType, Dimension>;
#else
  using OutputImageType = itk::Image<OutputPixelType, Dimension>;
#endif

  constexpr unsigned int numberOfProjections = 180;
  constexpr double       angularArc = 360.;
  constexpr unsigned int sid = 600;  // source to isocenter distance
  constexpr unsigned int sdd = 1200; // source to detector distance
  constexpr double       scale = 2.;
  std::string            configFileName = "Thorax";
  if (argc > 1 && argv[1] && argv[1][0] != '\0')
  {
    configFileName = argv[1];
  }
  constexpr unsigned int niterations = 10;

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

  // Create a constant image source for the projections
  auto projectionsSource = rtk::ConstantImageSource<OutputImageType>::New();
  projectionsSource->SetOrigin(itk::MakePoint(-127., -127., -127.));
  projectionsSource->SetSpacing(itk::MakeVector(2., 2., 2.));
  projectionsSource->SetSize(itk::MakeSize(128, 128, numberOfProjections));

  // Project the geometric phantom image
  auto pgp = rtk::ProjectGeometricPhantomImageFilter<OutputImageType, OutputImageType>::New();
  pgp->SetInput(projectionsSource->GetOutput());
  pgp->SetGeometry(geometry);
  pgp->SetPhantomScale(scale);
  pgp->SetConfigFile(configFileName);
  pgp->SetRotationMatrix(rotation);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(itk::WriteImage(pgp->GetOutput(), "projections.mha"));

  // Create new empty volume
  auto conjugateGradientSource = rtk::ConstantImageSource<OutputImageType>::New();
  conjugateGradientSource->SetOrigin(itk::MakePoint(-127., -127., -127.));
  conjugateGradientSource->SetSpacing(itk::MakeVector(2., 2., 2.));
  conjugateGradientSource->SetSize(itk::MakeSize(128, 128, 128));

  // Set the forward and back projection filters to be used
  using ConjugateGradientFilterType = rtk::ConjugateGradientConeBeamReconstructionFilter<OutputImageType>;
  auto conjugategradient = ConjugateGradientFilterType::New();

  conjugategradient->SetInputVolume(conjugateGradientSource->GetOutput());
  conjugategradient->SetInputProjectionStack(pgp->GetOutput());
  conjugategradient->SetGeometry(geometry);
  conjugategradient->SetNumberOfIterations(niterations);
#ifdef RTK_USE_CUDA
  conjugategradient->SetCudaConjugateGradient(true);
  conjugategradient->SetForwardProjectionFilter(ConjugateGradientFilterType::FP_CUDARAYCAST);
  conjugategradient->SetBackProjectionFilter(ConjugateGradientFilterType::BP_CUDAVOXELBASED);
#else
  conjugategradient->SetForwardProjectionFilter(ConjugateGradientFilterType::FP_JOSEPH);
  conjugategradient->SetBackProjectionFilter(ConjugateGradientFilterType::BP_VOXELBASED);
#endif

  using VerboseIterationCommandType = rtk::VerboseIterationCommand<ConjugateGradientFilterType>;
  auto verboseIterationCommand = rtk::VerboseIterationCommand<ConjugateGradientFilterType>::New();
  conjugategradient->AddObserver(itk::AnyEvent(), verboseIterationCommand);

  TRY_AND_EXIT_ON_ITK_EXCEPTION(conjugategradient->Update())

  // Write
  TRY_AND_EXIT_ON_ITK_EXCEPTION(itk::WriteImage(conjugategradient->GetOutput(), "conjugategradient.mha"));

  return EXIT_SUCCESS;
}
