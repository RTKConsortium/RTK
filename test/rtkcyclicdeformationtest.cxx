#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itksys/SystemTools.hxx>

#include "rtkTest.h"
#include "rtkConstantImageSource.h"

#ifdef USE_CUDA
#  include "rtkCudaCyclicDeformationImageFilter.h"
#else
#  include "rtkCyclicDeformationImageFilter.h"
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

int
main(int, char **)
{
  using OutputPixelType = itk::CovariantVector<float, 3>;

#ifdef USE_CUDA
  using DVFSequenceImageType = itk::CudaImage<OutputPixelType, 4>;
  using DVFImageType = itk::CudaImage<OutputPixelType, 3>;
#else
  using DVFSequenceImageType = itk::Image<OutputPixelType, 4>;
  using DVFImageType = itk::Image<OutputPixelType, 3>;
#endif

  auto origin = itk::MakePoint(-63., -31., -63., 0.);
#if FAST_TESTS_NO_CHECKS
  auto size = itk::MakeSize(8, 8, 8, 2);
  auto spacing = itk::MakeVector(16., 8., 16., 1.);
#else
  auto size = itk::MakeSize(32, 16, 32, 8);
  auto spacing = itk::MakeVector(4., 4., 4., 1.);
#endif

  DVFSequenceImageType::RegionType regionMotion;
  regionMotion.SetSize(size);

  DVFSequenceImageType::Pointer deformationField = DVFSequenceImageType::New();
  deformationField->SetRegions(regionMotion);
  deformationField->SetOrigin(origin);
  deformationField->SetSpacing(spacing);
  deformationField->Allocate();

  using IteratorType = itk::ImageRegionIteratorWithIndex<DVFSequenceImageType>;
  // Vector Field initilization
  DVFImageType::PixelType vec;
  IteratorType            dvfIt(deformationField, deformationField->GetLargestPossibleRegion());

  DVFSequenceImageType::OffsetType DVFCenter;
  DVFSequenceImageType::IndexType  toCenter;
  DVFCenter.Fill(0);
  DVFCenter[0] = size[0] / 2;
  DVFCenter[1] = size[1] / 2;
  DVFCenter[2] = size[2] / 2;
  while (!dvfIt.IsAtEnd())
  {
    vec.Fill(0.);
    toCenter = dvfIt.GetIndex() - DVFCenter;

    if (0.3 * toCenter[0] * toCenter[0] + toCenter[1] * toCenter[1] + toCenter[2] * toCenter[2] < 40)
    {
      if (dvfIt.GetIndex()[3] == 0)
        vec[0] = -8.;
      else
        vec[0] = 8.;
    }
    dvfIt.Set(vec);

    ++dvfIt;
  }

  // Signal
#ifdef USE_CUDA
  std::string signalFileName = "signal_CyclicDeformation_cuda.txt";
#else
  std::string signalFileName = "signal_CyclicDeformation.txt";
#endif
  std::ofstream signalFile(signalFileName.c_str());
  signalFile << "0.3" << std::endl;
  signalFile.close();

  // Set the forward and back projection filters to be used
  using CyclicDeformationType = rtk::CyclicDeformationImageFilter<DVFSequenceImageType, DVFImageType>;

  std::cout << "\n\n****** Case 1: CPU cyclic deformation field ******" << std::endl;

  CyclicDeformationType::Pointer cyclic = CyclicDeformationType::New();
  cyclic->SetInput(deformationField);
  cyclic->SetSignalFilename(signalFileName);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(cyclic->Update());

  CheckVectorImageQuality<DVFImageType>(cyclic->GetOutput(), cyclic->GetOutput(), 0.4, 12, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;

#ifdef USE_CUDA
  std::cout << "\n\n****** Case 2: GPU cyclic deformation field ******" << std::endl;

  cyclic = rtk::CudaCyclicDeformationImageFilter::New();
  cyclic->SetInput(deformationField);
  cyclic->SetSignalFilename(signalFileName);
  TRY_AND_EXIT_ON_ITK_EXCEPTION(cyclic->Update());

  CheckVectorImageQuality<DVFImageType>(cyclic->GetOutput(), cyclic->GetOutput(), 0.4, 12, 2.0);
  std::cout << "\n\nTest PASSED! " << std::endl;

#endif

  itksys::SystemTools::RemoveFile(signalFileName.c_str());

  return EXIT_SUCCESS;
}
