#include "rtkMatlabSparseMatrix.h"
#include "rtkThreeDCircularProjectionGeometry.h"
#include "rtkConstantImageSource.h"
#include "rtkJosephBackProjectionImageFilter.h"
#include "rtkStoreSparseMatrixSplatWeightMultiplication.h"
#include <vnl/vnl_sparse_matrix.h>
#include <fstream>
#include <itkImage.h>
#include <iostream>

int
main()
{
  using OutputPixelType = float;
  constexpr unsigned int Dimension = 3;
  using OutputImageType = itk::Image<OutputPixelType, Dimension>;

  constexpr unsigned int numberOfProjections = 10;
  constexpr unsigned int sid = 600;  // source to isocenter distance
  constexpr unsigned int sdd = 1200; // source to detector distance

  // Set up the geometry
  auto geometry = rtk::ThreeDCircularProjectionGeometry::New();
  for (unsigned int i = 0; i < numberOfProjections; ++i)
  {
    double angle = (360.0 * i) / numberOfProjections;
    geometry->AddProjectionInRadians(0, 0, angle * itk::Math::pi / 180.0, sid, sdd, 0, 0);
  }

  // Create projection images
  auto projectionsSource = rtk::ConstantImageSource<OutputImageType>::New();
  projectionsSource->SetSize(itk::MakeSize(512, 1, numberOfProjections));
  projectionsSource->SetSpacing(itk::MakeVector(1.0, 1.0, 1.0));
  projectionsSource->SetOrigin(itk::MakePoint(-256.0, 0.0, 0.0));
  projectionsSource->SetConstant(1.0);
  projectionsSource->Update();

  // Create volume
  auto volumeSource = rtk::ConstantImageSource<OutputImageType>::New();
  volumeSource->SetSize(itk::MakeSize(32, 32, 32));
  volumeSource->SetSpacing(itk::MakeVector(2.0, 2.0, 2.0));
  volumeSource->SetOrigin(itk::MakePoint(-32.0, -32.0, -32.0));
  volumeSource->SetConstant(0.0);
  volumeSource->Update();

  // Back-project with matrix capture
  using BackProjectionType = rtk::JosephBackProjectionImageFilter<
    OutputImageType,
    OutputImageType,
    rtk::Functor::InterpolationWeightMultiplicationBackProjection<OutputPixelType, OutputPixelType>,
    rtk::Functor::StoreSparseMatrixSplatWeightMultiplication<OutputPixelType, double, OutputPixelType>>;

  auto backProjection = BackProjectionType::New();
  backProjection->SetInput(volumeSource->GetOutput());
  backProjection->SetInput(1, projectionsSource->GetOutput());
  backProjection->SetGeometry(geometry);

  // Initialize and configure matrix capture
  backProjection->GetSplatWeightMultiplication().GetVnlSparseMatrix().resize(
    projectionsSource->GetOutput()->GetLargestPossibleRegion().GetNumberOfPixels(),
    volumeSource->GetOutput()->GetLargestPossibleRegion().GetNumberOfPixels());
  backProjection->GetSplatWeightMultiplication().SetProjectionsBuffer(
    projectionsSource->GetOutput()->GetBufferPointer());
  backProjection->GetSplatWeightMultiplication().SetVolumeBuffer(volumeSource->GetOutput()->GetBufferPointer());

  backProjection->Update();

  // Export to Matlab format
  vnl_sparse_matrix<double> & systemMatrix = backProjection->GetSplatWeightMultiplication().GetVnlSparseMatrix();
  auto matlabMatrix = rtk::MatlabSparseMatrix<OutputImageType>::New();
  matlabMatrix->SetMatrix(systemMatrix);
  matlabMatrix->SetOutput(volumeSource->GetOutput());

  std::ofstream outputFile("backprojection_matrix.mat", std::ios::binary);
  matlabMatrix->Save(outputFile);

  // Print matrix information
  matlabMatrix->Print();

  return 0;
}
