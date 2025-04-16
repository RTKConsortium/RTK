#include <itkImageFileWriter.h>
#include <itkMultiplyImageFilter.h>
#include <itkExpImageFilter.h>
#include <itkShotNoiseImageFilter.h>
#include <itkThresholdImageFilter.h>
#include <itkLogImageFilter.h>

#include <rtkConstantImageSource.h>
#include <rtkSheppLoganPhantomFilter.h>

using ImageType = itk::Image<float, 3>;

// Constant parameters
const float I0 = 1e4;     // Number of photons before attenuation
const float mu = 0.01879; // mm^-1, attenuation coefficient of water at 75 keV

int
main()
{
  // Simulate a Shepp Logan projection
  auto geometry = rtk::ThreeDCircularProjectionGeometry::New();
  geometry->AddProjection(1000., 0., 0.);

  auto source = rtk::ConstantImageSource<ImageType>::New();
  source->SetSize(itk::MakeSize(64, 64, 1));
  source->SetSpacing(itk::MakeVector(2., 2., 2.));
  source->SetOrigin(itk::MakePoint(-63., -63., 0.));

  auto shepp = rtk::SheppLoganPhantomFilter<ImageType, ImageType>::New();
  shepp->SetInput(source->GetOutput());
  shepp->SetGeometry(geometry);
  shepp->SetPhantomScale(70.);

  // Use ITK to add pre-log Poisson noise
  auto multiply = itk::MultiplyImageFilter<ImageType>::New();
  multiply->SetInput(shepp->GetOutput());
  multiply->SetConstant(-mu);

  auto exp = itk::ExpImageFilter<ImageType, ImageType>::New();
  exp->SetInput(multiply->GetOutput());

  auto multiply2 = itk::MultiplyImageFilter<ImageType>::New();
  multiply2->SetInput(exp->GetOutput());
  multiply2->SetConstant(I0);

  auto poisson = itk::ShotNoiseImageFilter<ImageType>::New();
  poisson->SetInput(multiply2->GetOutput());

  auto threshold = itk::ThresholdImageFilter<ImageType>::New();
  threshold->SetInput(poisson->GetOutput());
  threshold->SetLower(1.);
  threshold->SetOutsideValue(1.);

  auto multiply3 = itk::MultiplyImageFilter<ImageType>::New();
  multiply3->SetInput(threshold->GetOutput());
  multiply3->SetConstant(1. / I0);

  auto log = itk::LogImageFilter<ImageType, ImageType>::New();
  log->SetInput(multiply3->GetOutput());

  auto multiply4 = itk::MultiplyImageFilter<ImageType>::New();
  multiply4->SetInput(log->GetOutput());
  multiply4->SetConstant(-1. / mu);

  itk::WriteImage(multiply4->GetOutput(), "projection.mha");

  return 0;
}
