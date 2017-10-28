#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itksys/SystemTools.hxx>

#include "rtkTest.h"
#include "rtkConstantImageSource.h"

#ifdef RTK_USE_CUDA
  #include "rtkCudaCyclicDeformationImageFilter.h"
#else
  #include "rtkCyclicDeformationImageFilter.h"
#endif

/**
 * \file rtkcyclicdeformationtest.cxx
 *
 * \brief Functional test for classes performing 4D conjugate gradient-based
 * reconstruction.
 *
 * This test generates the projections of a phantom, which consists of two
 * ellipsoids (one of them moving). The resulting moving phantom is
 * reconstructed using 4D conjugate gradient and the generated
 * result is compared to the expected results (analytical computation).
 *
 * \author Cyril Mory
 */

int main(int, char** )
{
  typedef itk::CovariantVector<float, 3>        OutputPixelType;

#ifdef RTK_USE_CUDA
  typedef itk::CudaImage< OutputPixelType, 4 >  DVFSequenceImageType;
  typedef itk::CudaImage< OutputPixelType, 3 >  DVFImageType;
#else
  typedef itk::Image< OutputPixelType, 4 >  DVFSequenceImageType;
  typedef itk::Image< OutputPixelType, 3 >  DVFImageType;
#endif

  DVFSequenceImageType::PointType fourDOrigin;
  DVFSequenceImageType::SizeType fourDSize;
  DVFSequenceImageType::SpacingType fourDSpacing;

  fourDOrigin[0] = -63.;
  fourDOrigin[1] = -31.;
  fourDOrigin[2] = -63.;
  fourDOrigin[3] = 0;
#if FAST_TESTS_NO_CHECKS
  fourDSize[0] = 8;
  fourDSize[1] = 8;
  fourDSize[2] = 8;
  fourDSize[3] = 2;
  fourDSpacing[0] = 16.;
  fourDSpacing[1] = 8.;
  fourDSpacing[2] = 16.;
  fourDSpacing[3] = 1.;
#else
  fourDSize[0] = 32;
  fourDSize[1] = 16;
  fourDSize[2] = 32;
  fourDSize[3] = 8;
  fourDSpacing[0] = 4.;
  fourDSpacing[1] = 4.;
  fourDSpacing[2] = 4.;
  fourDSpacing[3] = 1.;
#endif

  // Create a vector field and its (very rough) inverse
  typedef itk::ImageRegionIteratorWithIndex< DVFSequenceImageType > IteratorType;

  DVFSequenceImageType::Pointer deformationField = DVFSequenceImageType::New();

  DVFSequenceImageType::IndexType startMotion;
  startMotion[0] = 0; // first index on X
  startMotion[1] = 0; // first index on Y
  startMotion[2] = 0; // first index on Z
  startMotion[3] = 0; // first index on t
  DVFSequenceImageType::SizeType sizeMotion;
  sizeMotion[0] = fourDSize[0];
  sizeMotion[1] = fourDSize[1];
  sizeMotion[2] = fourDSize[2];
  sizeMotion[3] = 2;
  DVFSequenceImageType::PointType originMotion;
  originMotion[0] = -63.;
  originMotion[1] = -31.;
  originMotion[2] = -63.;
  originMotion[3] = 0.;
  DVFSequenceImageType::RegionType regionMotion;
  regionMotion.SetSize( sizeMotion );
  regionMotion.SetIndex( startMotion );

  DVFSequenceImageType::SpacingType spacingMotion;
  spacingMotion[0] = fourDSpacing[0];
  spacingMotion[1] = fourDSpacing[1];
  spacingMotion[2] = fourDSpacing[2];
  spacingMotion[3] = fourDSpacing[3];

  deformationField->SetRegions( regionMotion );
  deformationField->SetOrigin(originMotion);
  deformationField->SetSpacing(spacingMotion);
  deformationField->Allocate();

  // Vector Field initilization
  DVFImageType::PixelType vec;
  IteratorType dvfIt( deformationField, deformationField->GetLargestPossibleRegion() );

  DVFSequenceImageType::OffsetType DVFCenter;
  DVFSequenceImageType::IndexType toCenter;
  DVFCenter.Fill(0);
  DVFCenter[0] = sizeMotion[0]/2;
  DVFCenter[1] = sizeMotion[1]/2;
  DVFCenter[2] = sizeMotion[2]/2;
  while (!dvfIt.IsAtEnd())
    {
    vec.Fill(0.);
    toCenter = dvfIt.GetIndex() - DVFCenter;

    if (0.3 * toCenter[0] * toCenter[0] + toCenter[1] * toCenter[1] + toCenter[2] * toCenter[2] < 40)
      {
      if(dvfIt.GetIndex()[3]==0)
        vec[0] = -8.;
      else
        vec[0] = 8.;
      }
    dvfIt.Set(vec);

    ++dvfIt;
    }

  // Signal
  std::ofstream signalFile("signal.txt");
  signalFile << "0.3" << std::endl;

  // Set the forward and back projection filters to be used
  typedef rtk::CyclicDeformationImageFilter<DVFImageType> CyclicDeformationType;

  std::cout << "\n\n****** Case 1: CPU cyclic deformation field ******" << std::endl;

  CyclicDeformationType::Pointer cyclic = CyclicDeformationType::New();
  cyclic->SetInput(deformationField);
  cyclic->SetSignalFilename("signal.txt");
  TRY_AND_EXIT_ON_ITK_EXCEPTION( cyclic->Update() );

  CheckVectorImageQuality<DVFImageType>(cyclic->GetOutput(), cyclic->GetOutput(), 0.4, 12, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;

#ifdef USE_CUDA
  std::cout << "\n\n****** Case 2: GPU cyclic deformation field ******" << std::endl;

  cyclic = rtk::CudaCyclicDeformationImageFilter::New();
  cyclic->SetInput(deformationField);
  cyclic->SetSignalFilename("signal.txt");
  TRY_AND_EXIT_ON_ITK_EXCEPTION( cyclic->Update() );

  CheckVectorImageQuality<DVFImageType>(cyclic->GetOutput(), cyclic->GetOutput(), 0.4, 12, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;

#endif

  itksys::SystemTools::RemoveFile("signal.txt");

  return EXIT_SUCCESS;
}
