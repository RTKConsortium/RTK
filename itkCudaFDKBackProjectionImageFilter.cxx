#include "itkCudaFDKBackProjectionImageFilter.h"
#include "itkCudaUtilities.hcu"

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkMacro.h>

namespace itk
{

void
CudaFDKBackProjectionImageFilter
::GenerateData()
{
  this->AllocateOutputs();

  OutputImageRegionType region = this->GetOutput()->GetRequestedRegion();

  if(region != this->GetOutput()->GetBufferedRegion() )
    itkExceptionMacro(<< "Can't handle different requested and buffered regions "
                      << region
                      << this->GetOutput()->GetBufferedRegion() );

  const unsigned int Dimension = ImageType::ImageDimension;
  const unsigned int nProj = this->GetInput(1)->GetLargestPossibleRegion().GetSize(Dimension-1);

  // Ramp factor is the correction for ramp filter which did not account for the
  // divergence of the beam
  const GeometryPointer geometry = dynamic_cast<GeometryType *>(this->GetGeometry().GetPointer() );

  // Rotation center (assumed to be at 0 yet)
  ImageType::PointType rotCenterPoint;
  rotCenterPoint.Fill(0.0);
  ContinuousIndex<double, Dimension> rotCenterIndex;
  this->GetInput(0)->TransformPhysicalPointToContinuousIndex(rotCenterPoint, rotCenterIndex);

  // Include non-zero index in matrix
  itk::Matrix<double, 4, 4> matrixIdxVol;
  matrixIdxVol.SetIdentity();
  for(unsigned int i=0; i<3; i++)
    {
    matrixIdxVol[i][3] = region.GetIndex()[i];
    rotCenterIndex[i] -= region.GetIndex()[i];
    }

  // Load dimensions arguments
  int3 vol_dim;
  vol_dim.x = region.GetSize()[0];
  vol_dim.y = region.GetSize()[1];
  vol_dim.z = region.GetSize()[2];

  int2 img_dim;
  img_dim.x = this->GetInput(1)->GetBufferedRegion().GetSize()[0];
  img_dim.y = this->GetInput(1)->GetBufferedRegion().GetSize()[1];

  // Cuda init
  std::vector<int> devices = GetListOfCudaDevices();
  if(devices.size()>1)
    {
    cudaThreadExit();
    cudaSetDevice(devices[0]);
    }

  float *    dev_vol;
  cudaArray *dev_img;
  float *    dev_matrix;
  CUDA_reconstruct_conebeam_init (img_dim, vol_dim, dev_vol, dev_img, dev_matrix);

  // Go over each projection
  for(unsigned int iProj=0; iProj<nProj; iProj++)
    {
    // Extract the current slice
    ProjectionImagePointer projection = this->GetProjection(iProj);

    // Index to index matrix normalized to have a correct backprojection weight
    // (1 at the isocenter)
    ProjectionMatrixType matrix = GetIndexToIndexProjectionMatrix(iProj, projection);

    // We correct the matrix for non zero indexes
    itk::Matrix<double, 3, 3> matrixIdxProj;
    matrixIdxProj.SetIdentity();
    for(unsigned int i=0; i<2; i++)
       //SR: 0.5 for 2D texture
      matrixIdxProj[i][2] = -1*(projection->GetBufferedRegion().GetIndex()[i])+0.5;

    matrix = matrixIdxProj.GetVnlMatrix() * matrix.GetVnlMatrix() * matrixIdxVol.GetVnlMatrix();

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
