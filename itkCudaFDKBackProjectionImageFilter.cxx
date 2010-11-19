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

  // Create interpolator, could be any interpolation
  typedef itk::LinearInterpolateImageFunction< ProjectionImageType, double > InterpolatorType;
  typename InterpolatorType::Pointer interpolator = InterpolatorType::New();

  // Ramp factor is the correction for ramp filter which did not account for the divergence of the beam
  const GeometryPointer geometry = dynamic_cast<GeometryType *>(this->GetGeometry().GetPointer());
  double rampFactor = geometry->GetSourceToDetectorDistance() / geometry->GetSourceToIsocenterDistance();
  rampFactor *= 0.5; // Factor 1/2 in eq 176, page 106, Kak & Slaney

  // Rotation center (assumed to be at 0 yet)
  ImageType::PointType rotCenterPoint;
  rotCenterPoint.Fill(0.0);
  ContinuousIndex<double, Dimension> rotCenterIndex;
  this->GetInput(0)->TransformPhysicalPointToContinuousIndex(rotCenterPoint, rotCenterIndex);
  
  // Load kernel arguments
  m_kargs.vol_dim.x = this->GetOutput()->GetLargestPossibleRegion().GetSize()[0];
  m_kargs.vol_dim.y = this->GetOutput()->GetLargestPossibleRegion().GetSize()[1];
  m_kargs.vol_dim.z = this->GetOutput()->GetLargestPossibleRegion().GetSize()[2];
  m_kargs.img_dim.x = this->GetInput(1)->GetLargestPossibleRegion().GetSize()[0];
  m_kargs.img_dim.y = this->GetInput(1)->GetLargestPossibleRegion().GetSize()[1];

  // Cuda init
  kernel_args_fdk *dev_kargs;
  float *dev_vol;
  float *dev_img;
  float *dev_matrix;
  CUDA_reconstruct_conebeam_init (&m_kargs, dev_kargs, dev_vol, dev_img, dev_matrix);

  // Go over each projection
  for(unsigned int iProj=0; iProj<nProj; iProj++)
    {
    // Extract the current slice
    ProjectionImagePointer projection = this->GetProjection(iProj, angWeights[iProj] * rampFactor);
    interpolator->SetInputImage(projection);

    // Index to index matrix normalized to have a correct backprojection weight (1 at the isocenter)
    ProjectionMatrixType matrix = GetIndexToIndexProjectionMatrix(iProj, projection);
    double perspFactor = matrix[Dimension-1][Dimension];
    for(unsigned int j=0; j<Dimension; j++)
      perspFactor += matrix[Dimension-1][j] * rotCenterIndex[j];
    matrix /= perspFactor;

    
    for (int j = 0; j < 12; j++) {
        m_kargs.matrix[j] = matrix[j/4][j%4];
    }

    CUDA_reconstruct_conebeam(this->GetOutput()->GetBufferPointer(),
                              projection->GetBufferPointer(),
                              &m_kargs, dev_kargs, dev_vol, dev_img, dev_matrix);
    }

  CUDA_reconstruct_conebeam_cleanup (dev_kargs, dev_vol, dev_img, dev_matrix);
}

} // end namespace itk
