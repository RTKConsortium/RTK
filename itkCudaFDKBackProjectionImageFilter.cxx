#include "itkCudaFDKBackProjectionImageFilter.h"

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkLinearInterpolateImageFunction.h>

namespace itk
{

void
CudaFDKBackProjectionImageFilter
::GenerateData()
{
  this->AllocateOutputs();
  this->UpdateAngularWeights();
  
  std::vector<double> angWeights = this->GetAngularWeights();
  
  OutputImageRegionType region = this->GetOutput()->GetRequestedRegion();

  const unsigned int Dimension = ImageType::ImageDimension;
  const unsigned int nProj = this->GetInput(1)->GetLargestPossibleRegion().GetSize(Dimension-1);

  // Ramp factor is the correction for ramp filter which did not account for the divergence of the beam
  const GeometryPointer geometry = dynamic_cast<GeometryType *>(this->GetGeometry().GetPointer());
  double rampFactor = geometry->GetSourceToDetectorDistance() / geometry->GetSourceToIsocenterDistance();
  rampFactor *= 0.5; // Factor 1/2 in eq 176, page 106, Kak & Slaney

  // Rotation center (assumed to be at 0 yet)
  ImageType::PointType rotCenterPoint;
  rotCenterPoint.Fill(0.0);
  ContinuousIndex<double, Dimension> rotCenterIndex;
  this->GetInput(0)->TransformPhysicalPointToContinuousIndex(rotCenterPoint, rotCenterIndex);
  
  // Load dimensions arguments
  int3 vol_dim;
  vol_dim.x = this->GetOutput()->GetLargestPossibleRegion().GetSize()[0];
  vol_dim.y = this->GetOutput()->GetLargestPossibleRegion().GetSize()[1];
  vol_dim.z = this->GetOutput()->GetLargestPossibleRegion().GetSize()[2];

  int2 img_dim;
  img_dim.x = this->GetInput(1)->GetLargestPossibleRegion().GetSize()[0];
  img_dim.y = this->GetInput(1)->GetLargestPossibleRegion().GetSize()[1];

  // Cuda init
  float *dev_vol;
  cudaArray *dev_img;
  float *dev_matrix;
  CUDA_reconstruct_conebeam_init (img_dim, vol_dim, dev_vol, dev_img, dev_matrix);

  // Go over each projection
  for(unsigned int iProj=0; iProj<nProj; iProj++)
    {
    // Extract the current slice
    ProjectionImagePointer projection = this->GetProjection(iProj, angWeights[iProj] * rampFactor);

    // Index to index matrix normalized to have a correct backprojection weight (1 at the isocenter)
    ProjectionMatrixType matrix = GetIndexToIndexProjectionMatrix(iProj, projection);
    double perspFactor = matrix[Dimension-1][Dimension];
    for(unsigned int j=0; j<Dimension; j++)
      perspFactor += matrix[Dimension-1][j] * rotCenterIndex[j];
    matrix /= perspFactor;

    float fMatrix[12];
    for (int j = 0; j < 12; j++)
      fMatrix[j] = matrix[j/4][j%4];

    CUDA_reconstruct_conebeam(img_dim, vol_dim, projection->GetBufferPointer(), fMatrix, dev_vol, dev_img, dev_matrix);
    }

  CUDA_reconstruct_conebeam_cleanup (vol_dim, this->GetOutput()->GetBufferPointer(), dev_vol, dev_img, dev_matrix);
}

} // end namespace itk
