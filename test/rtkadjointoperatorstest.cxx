#include "rtkMacro.h"
#include "rtkTest.h"
#include "itkRandomImageSource.h"
#include "rtkConstantImageSource.h"
#include "rtkJosephBackProjectionImageFilter.h"
#include "rtkJosephForwardProjectionImageFilter.h"
#include "rtkJosephBackAttenuatedProjectionImageFilter.h"
#include "rtkJosephForwardAttenuatedProjectionImageFilter.h"
#ifdef RTK_USE_CUDA
  #include "rtkCudaForwardProjectionImageFilter.h"
  #include "rtkCudaRayCastBackProjectionImageFilter.h"
#endif

/**
 * \file rtkadjointoperatorstest.cxx
 *
 * \brief Tests whether forward and back projectors are matched
 *
 * This test generates a random volume "v" and a random set of projections "p",
 * and compares the scalar products <Rv , p> and <v, R* p>, where R is either the
 * Joseph forward projector or the Cuda ray cast forward projector,
 * and R* is either the Joseph back projector or the Cuda ray cast back projector.
 * If R* is indeed the adjoint of R, these scalar products are equal.
 *
 * \author Cyril Mory
 */

int main(int, char** )
{
  const unsigned int Dimension = 3;

#ifdef USE_CUDA
  typedef float                                        OutputPixelType;
  typedef itk::CudaImage< OutputPixelType, Dimension > OutputImageType;
#else
  typedef double                                   OutputPixelType;
  typedef itk::Image< OutputPixelType, Dimension > OutputImageType;
#endif

#if FAST_TESTS_NO_CHECKS
  const unsigned int NumberOfProjectionImages = 3;
#else
  const unsigned int NumberOfProjectionImages = 180;
#endif

  // Random image sources
  typedef itk::RandomImageSource< OutputImageType > RandomImageSourceType;
  RandomImageSourceType::Pointer randomVolumeSource  = RandomImageSourceType::New();
  RandomImageSourceType::Pointer randomProjectionsSource = RandomImageSourceType::New();

  // Constant sources
  typedef rtk::ConstantImageSource< OutputImageType > ConstantImageSourceType;
  ConstantImageSourceType::Pointer constantVolumeSource = ConstantImageSourceType::New();
  ConstantImageSourceType::Pointer constantProjectionsSource = ConstantImageSourceType::New();
  ConstantImageSourceType::Pointer constantAttenuationSource = ConstantImageSourceType::New();
  
  // Image meta data
  RandomImageSourceType::PointType origin;
  RandomImageSourceType::SizeType size;
  RandomImageSourceType::SpacingType spacing;

  // Volume metadata
  origin[0] = -127.;
  origin[1] = -127.;
  origin[2] = -127.;
#if FAST_TESTS_NO_CHECKS
  size[0] = 2;
  size[1] = 2;
  size[2] = 2;
  spacing[0] = 252.;
  spacing[1] = 252.;
  spacing[2] = 252.;
#else
  size[0] = 64;
  size[1] = 64;
  size[2] = 64;
  spacing[0] = 4.;
  spacing[1] = 4.;
  spacing[2] = 4.;
#endif
  randomVolumeSource->SetOrigin( origin );
  randomVolumeSource->SetSpacing( spacing );
  randomVolumeSource->SetSize( size );
  randomVolumeSource->SetMin( 0. );
  randomVolumeSource->SetMax( 1. );
  randomVolumeSource->SetNumberOfThreads(2); //With 1, it's deterministic

  constantVolumeSource->SetOrigin( origin );
  constantVolumeSource->SetSpacing( spacing );
  constantVolumeSource->SetSize( size );
  constantVolumeSource->SetConstant( 0. );

  constantAttenuationSource->SetOrigin( origin );
  constantAttenuationSource->SetSpacing( spacing );
  constantAttenuationSource->SetSize( size );
  constantAttenuationSource->SetConstant( 0.0154 );

  // Projections metadata
  origin[0] = -255.;
  origin[1] = -255.;
  origin[2] = -255.;
#if FAST_TESTS_NO_CHECKS
  size[0] = 2;
  size[1] = 2;
  size[2] = NumberOfProjectionImages;
  spacing[0] = 504.;
  spacing[1] = 504.;
  spacing[2] = 504.;
#else
  size[0] = 64;
  size[1] = 64;
  size[2] = NumberOfProjectionImages;
  spacing[0] = 8.;
  spacing[1] = 8.;
  spacing[2] = 8.;
#endif
  randomProjectionsSource->SetOrigin( origin );
  randomProjectionsSource->SetSpacing( spacing );
  randomProjectionsSource->SetSize( size );
  randomProjectionsSource->SetMin( 0. );
  randomProjectionsSource->SetMax( 100. );
  randomProjectionsSource->SetNumberOfThreads(2); //With 1, it's deterministic

  constantProjectionsSource->SetOrigin( origin );
  constantProjectionsSource->SetSpacing( spacing );
  constantProjectionsSource->SetSize( size );
  constantProjectionsSource->SetConstant( 0. );

  // Update all sources
  TRY_AND_EXIT_ON_ITK_EXCEPTION( randomVolumeSource->Update() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( constantVolumeSource->Update() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( randomProjectionsSource->Update() );
  TRY_AND_EXIT_ON_ITK_EXCEPTION( constantProjectionsSource->Update() );

  // Geometry object
  typedef rtk::ThreeDCircularProjectionGeometry GeometryType;
  GeometryType::Pointer geometry = GeometryType::New();
  for(unsigned int noProj=0; noProj<NumberOfProjectionImages; noProj++)
    geometry->AddProjection(600., 1200., noProj*360./NumberOfProjectionImages);

  for (unsigned int panel = 0; panel<2; panel++)
    {
    if (panel==0)
      std::cout << "\n\n****** Testing with flat panel ******" << std::endl;
    else
      {
      std::cout << "\n\n****** Testing with cylindrical panel ******" << std::endl;
      geometry->SetRadiusCylindricalDetector(200);
      }

      std::cout << "\n\n****** Joseph Forward projector ******" << std::endl;

      typedef rtk::JosephForwardProjectionImageFilter<OutputImageType, OutputImageType> JosephForwardProjectorType;
      JosephForwardProjectorType::Pointer fw = JosephForwardProjectorType::New();
      fw->SetInput(0, constantProjectionsSource->GetOutput());
      fw->SetInput(1, randomVolumeSource->GetOutput());
      fw->SetGeometry( geometry );
      TRY_AND_EXIT_ON_ITK_EXCEPTION( fw->Update() );

      std::cout << "\n\n****** Joseph Back projector ******" << std::endl;

      typedef rtk::JosephBackProjectionImageFilter<OutputImageType, OutputImageType> JosephBackProjectorType;
      JosephBackProjectorType::Pointer bp = JosephBackProjectorType::New();
      bp->SetInput(0, constantVolumeSource->GetOutput());
      bp->SetInput(1, randomProjectionsSource->GetOutput());
      bp->SetGeometry( geometry.GetPointer() );

      TRY_AND_EXIT_ON_ITK_EXCEPTION( bp->Update() );

      CheckScalarProducts<OutputImageType, OutputImageType>(randomVolumeSource->GetOutput(), bp->GetOutput(), randomProjectionsSource->GetOutput(), fw->GetOutput());
      std::cout << "\n\nTest PASSED! " << std::endl;

      typedef itk::Image<itk::Vector<OutputPixelType, 3>, Dimension> VectorImageType;
      VectorImageType::Pointer vectorRandomProjections = VectorImageType::New();
      VectorImageType::Pointer vectorConstantProjections = VectorImageType::New();
      VectorImageType::Pointer vectorRandomVolume = VectorImageType::New();
      VectorImageType::Pointer vectorConstantVolume = VectorImageType::New();
      vectorRandomProjections->CopyInformation(randomProjectionsSource->GetOutput());
      vectorRandomProjections->SetRegions(randomProjectionsSource->GetOutput()->GetLargestPossibleRegion());
      vectorRandomProjections->Allocate();
      itk::ImageRegionIterator<VectorImageType> outIt1(vectorRandomProjections, vectorRandomProjections->GetLargestPossibleRegion());
      itk::ImageRegionIterator<OutputImageType> inIt1(randomProjectionsSource->GetOutput(), randomProjectionsSource->GetOutput()->GetLargestPossibleRegion());
      itk::Vector<OutputPixelType, 3> temp;
      while(!outIt1.IsAtEnd())
        {
        temp[0] = inIt1.Get();
        temp[1] = inIt1.Get();
        temp[2] = inIt1.Get();
        outIt1.Set(temp);
        ++outIt1;
        ++inIt1;
        }

      constantVolumeSource->Update();
      vectorConstantVolume->CopyInformation(constantVolumeSource->GetOutput());
      vectorConstantVolume->SetRegions(constantVolumeSource->GetOutput()->GetLargestPossibleRegion());
      vectorConstantVolume->Allocate();
      itk::ImageRegionIterator<VectorImageType> outIt2(vectorConstantVolume, vectorConstantVolume->GetLargestPossibleRegion());
      itk::ImageRegionIterator<OutputImageType> inIt2(constantVolumeSource->GetOutput(), constantVolumeSource->GetOutput()->GetLargestPossibleRegion());
      while(!outIt2.IsAtEnd())
        {
        temp[0] = inIt2.Get();
        temp[1] = inIt2.Get();
        temp[2] = inIt2.Get();
        outIt2.Set(temp);
        ++outIt2;
        ++inIt2;
        }

      constantProjectionsSource->Update();
      vectorConstantProjections->CopyInformation(constantProjectionsSource->GetOutput());
      vectorConstantProjections->SetRegions(constantProjectionsSource->GetOutput()->GetLargestPossibleRegion());
      vectorConstantProjections->Allocate();
      itk::ImageRegionIterator<VectorImageType> outIt3(vectorConstantProjections, vectorConstantProjections->GetLargestPossibleRegion());
      itk::ImageRegionIterator<OutputImageType> inIt3(constantProjectionsSource->GetOutput(), constantProjectionsSource->GetOutput()->GetLargestPossibleRegion());
      while(!outIt3.IsAtEnd())
        {
        temp[0] = inIt3.Get();
        temp[1] = inIt3.Get();
        temp[2] = inIt3.Get();
        outIt3.Set(temp);
        ++outIt3;
        ++inIt3;
        }

      vectorRandomVolume->CopyInformation(randomVolumeSource->GetOutput());
      vectorRandomVolume->SetRegions(randomVolumeSource->GetOutput()->GetLargestPossibleRegion());
      vectorRandomVolume->Allocate();
      itk::ImageRegionIterator<VectorImageType> outIt4(vectorRandomVolume, vectorRandomVolume->GetLargestPossibleRegion());
      itk::ImageRegionIterator<OutputImageType> inIt4(randomVolumeSource->GetOutput(), randomVolumeSource->GetOutput()->GetLargestPossibleRegion());
      while(!outIt4.IsAtEnd())
        {
        temp[0] = inIt4.Get();
        temp[1] = inIt4.Get();
        temp[2] = inIt4.Get();
        outIt4.Set(temp);
        ++outIt4;
        ++inIt4;
        }

      std::cout << "\n\n****** Joseph Vector Forward projector ******" << std::endl;

      typedef rtk::JosephForwardProjectionImageFilter
      <VectorImageType,
       VectorImageType>
      VectorJosephForwardProjectorType;

      VectorJosephForwardProjectorType::Pointer vfw = VectorJosephForwardProjectorType::New();
      vfw->SetInput(0, vectorConstantProjections);
      vfw->SetInput(1, vectorRandomVolume);
      vfw->SetGeometry( geometry );
      TRY_AND_EXIT_ON_ITK_EXCEPTION( vfw->Update() );

      std::cout << "\n\n****** Joseph Vector Back projector ******" << std::endl;
      typedef rtk::JosephBackProjectionImageFilter<VectorImageType,
                                                   VectorImageType> VectorJosephBackProjectorType;
      VectorJosephBackProjectorType::Pointer vbp = VectorJosephBackProjectorType::New();
      vbp->SetInput(0, vectorConstantVolume);
      vbp->SetInput(1, vectorRandomProjections);
      vbp->SetGeometry( geometry.GetPointer() );

      TRY_AND_EXIT_ON_ITK_EXCEPTION( vbp->Update() );

      CheckVectorScalarProducts<VectorImageType, VectorImageType>(vectorRandomVolume, vbp->GetOutput(), vectorRandomProjections, vfw->GetOutput());

      std::cout << "\n\n****** Attenuated Joseph Forward projector ******" << std::endl;

      typedef rtk::JosephForwardAttenuatedProjectionImageFilter<OutputImageType, OutputImageType> JosephForwardAttenuatedProjectorType;
      JosephForwardAttenuatedProjectorType::Pointer attfw = JosephForwardAttenuatedProjectorType::New();
      attfw->SetInput(0, constantProjectionsSource->GetOutput());
      attfw->SetInput(1, randomVolumeSource->GetOutput());
      attfw->SetInput(2, constantAttenuationSource->GetOutput());
      attfw->SetGeometry( geometry );
      TRY_AND_EXIT_ON_ITK_EXCEPTION( attfw->Update() );

      std::cout << "\n\n****** Attenuated Joseph Back projector ******" << std::endl;

      typedef rtk::JosephBackAttenuatedProjectionImageFilter<OutputImageType, OutputImageType> JosephBackAttenuatedProjectorType;
      JosephBackAttenuatedProjectorType::Pointer attbp = JosephBackAttenuatedProjectorType::New();
      attbp->SetInput(0, constantVolumeSource->GetOutput());
      attbp->SetInput(1, randomProjectionsSource->GetOutput());
      attbp->SetInput(2, constantAttenuationSource->GetOutput());
      attbp->SetGeometry( geometry.GetPointer() );

      TRY_AND_EXIT_ON_ITK_EXCEPTION( attbp->Update() );

      CheckScalarProducts<OutputImageType, OutputImageType>(randomVolumeSource->GetOutput(), attbp->GetOutput(), randomProjectionsSource->GetOutput(), attfw->GetOutput());

      std::cout << "\n\nTest PASSED! " << std::endl;

    #ifdef USE_CUDA
      std::cout << "\n\n****** Cuda Ray Cast Forward projector ******" << std::endl;

      typedef rtk::CudaForwardProjectionImageFilter<OutputImageType, OutputImageType> CudaForwardProjectorType;
      CudaForwardProjectorType::Pointer cfw = CudaForwardProjectorType::New();
      cfw->SetInput(0, constantProjectionsSource->GetOutput());
      cfw->SetInput(1, randomVolumeSource->GetOutput());
      cfw->SetGeometry( geometry );
      TRY_AND_EXIT_ON_ITK_EXCEPTION( cfw->Update() );

      std::cout << "\n\n****** Cuda Ray Cast Back projector ******" << std::endl;

      typedef rtk::CudaRayCastBackProjectionImageFilter CudaRayCastBackProjectorType;
      CudaRayCastBackProjectorType::Pointer cbp = CudaRayCastBackProjectorType::New();
      cbp->SetInput(0, constantVolumeSource->GetOutput());
      cbp->SetInput(1, randomProjectionsSource->GetOutput());
      cbp->SetGeometry( geometry.GetPointer() );
      cbp->SetNormalize(false);

      TRY_AND_EXIT_ON_ITK_EXCEPTION( cbp->Update() );

      CheckScalarProducts<OutputImageType, OutputImageType>(randomVolumeSource->GetOutput(), cbp->GetOutput(), randomProjectionsSource->GetOutput(), cfw->GetOutput());
      std::cout << "\n\nTest PASSED! " << std::endl;
    #endif
    }

  return EXIT_SUCCESS;
}
