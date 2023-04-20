/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#ifndef rtkCudaBackProjectionImageFilter_hxx
#define rtkCudaBackProjectionImageFilter_hxx

#include "rtkConfiguration.h"
// Conditional definition of the class to pass ITKHeaderTest
#ifdef RTK_USE_CUDA

#  include "rtkCudaBackProjectionImageFilter.h"
#  include "rtkCudaUtilities.hcu"
#  include "rtkCudaBackProjectionImageFilter.hcu"

#  include <itkImageRegionConstIterator.h>
#  include <itkImageRegionIteratorWithIndex.h>
#  include <itkLinearInterpolateImageFunction.h>
#  include <itkMacro.h>

namespace rtk
{

template <class ImageType>
CudaBackProjectionImageFilter<ImageType>::CudaBackProjectionImageFilter()
{}

template <class ImageType>
void
CudaBackProjectionImageFilter<ImageType>::GPUGenerateData()
{
  const unsigned int Dimension = ImageType::ImageDimension;
  const unsigned int nProj = this->GetInput(1)->GetLargestPossibleRegion().GetSize(Dimension - 1);
  const unsigned int iFirstProj = this->GetInput(1)->GetLargestPossibleRegion().GetIndex(Dimension - 1);
  if (SLAB_SIZE > 1024)
    itkGenericExceptionMacro(
      "The CUDA voxel based back projection image filter can only handle slabs of at most 1024 projections");

  // Rotation center (assumed to be at 0 yet)
  typename ImageType::PointType rotCenterPoint;
  rotCenterPoint.Fill(0.0);
  using ValueType = typename ImageType::PointType::ValueType;
  itk::ContinuousIndex<double, Dimension> rotCenterIndex =
    this->GetInput(0)->template TransformPhysicalPointToContinuousIndex<ValueType, double>(rotCenterPoint);

  // Include non-zero index in matrix
  itk::Matrix<double, 4, 4> matrixIdxVol;
  matrixIdxVol.SetIdentity();
  for (unsigned int i = 0; i < 3; i++)
  {
    matrixIdxVol[i][3] = this->GetOutput()->GetRequestedRegion().GetIndex()[i];
    rotCenterIndex[i] -= this->GetOutput()->GetRequestedRegion().GetIndex()[i];
  }

  // Cuda convenient format for dimensions
  int projectionSize[3];
  projectionSize[0] = this->GetInput(1)->GetBufferedRegion().GetSize()[0];
  projectionSize[1] = this->GetInput(1)->GetBufferedRegion().GetSize()[1];

  int volumeSize[3];
  volumeSize[0] = this->GetOutput()->GetBufferedRegion().GetSize()[0];
  volumeSize[1] = this->GetOutput()->GetBufferedRegion().GetSize()[1];
  volumeSize[2] = this->GetOutput()->GetBufferedRegion().GetSize()[2];

  float * pin = *(float **)(this->GetInput()->GetCudaDataManager()->GetGPUBufferPointer());
  float * pout = *(float **)(this->GetOutput()->GetCudaDataManager()->GetGPUBufferPointer());

  float *   stackGPUPointer = *(float **)(this->GetInput(1)->GetCudaDataManager()->GetGPUBufferPointer());
  ptrdiff_t projSize = this->GetInput(1)->GetBufferedRegion().GetSize()[0] *
                       this->GetInput(1)->GetBufferedRegion().GetSize()[1] *
                       itk::NumericTraits<typename ImageType::PixelType>::GetLength();
  stackGPUPointer += projSize * (iFirstProj - this->GetInput(1)->GetBufferedRegion().GetIndex()[2]);

  // Allocate a large matrix to hold the matrix of all projections
  // fMatrix is for flat detector, the other two are for cylindrical
  float * fMatrix = new float[12 * nProj];
  float * fvolIndexToProjPP = new float[12 * nProj];
  float * fprojPPToProjIndex = new float[9];

  // Correction for non-zero indices in the projections
  itk::Matrix<double, 3, 3> matrixIdxProj;
  matrixIdxProj.SetIdentity();
  for (unsigned int i = 0; i < 2; i++)
    // SR: 0.5 for 2D texture
    matrixIdxProj[i][2] = -1 * (this->GetInput(1)->GetBufferedRegion().GetIndex()[i]) + 0.5;

  // Go over each projection
  for (unsigned int iProj = iFirstProj; iProj < iFirstProj + nProj; iProj++)
  {
    // Projection physical point to projection index matrix
    itk::Matrix<double, 3, 3> projPPToProjIndex = this->GetProjectionPhysicalPointToProjectionIndexMatrix(iProj);
    projPPToProjIndex = matrixIdxProj.GetVnlMatrix() * projPPToProjIndex.GetVnlMatrix();
    for (int j = 0; j < 9; j++)
      fprojPPToProjIndex[j] = projPPToProjIndex[j / 3][j % 3];

    // Volume index to projection physical point matrix
    // normalized to have a correct backprojection weight
    // (1 at the isocenter)
    typename BackProjectionImageFilterType::ProjectionMatrixType volIndexToProjPP =
      this->GetVolumeIndexToProjectionPhysicalPointMatrix(iProj);

    // Correction for non-zero indices in the volume
    volIndexToProjPP = volIndexToProjPP.GetVnlMatrix() * matrixIdxVol.GetVnlMatrix();

    double perspFactor = volIndexToProjPP[Dimension - 1][Dimension];
    for (unsigned int j = 0; j < Dimension; j++)
      perspFactor += volIndexToProjPP[Dimension - 1][j] * rotCenterIndex[j];
    volIndexToProjPP /= perspFactor;

    typename BackProjectionImageFilterType::ProjectionMatrixType matrix =
      typename BackProjectionImageFilterType::ProjectionMatrixType(projPPToProjIndex.GetVnlMatrix() *
                                                                   volIndexToProjPP.GetVnlMatrix());

    // Fill float arrays with matrices coefficients, to be passed to GPU
    for (int j = 0; j < 12; j++)
    {
      fvolIndexToProjPP[j + (iProj - iFirstProj) * 12] = volIndexToProjPP[j / 4][j % 4];
      fMatrix[j + (iProj - iFirstProj) * 12] = matrix[j / 4][j % 4];
    }
  }

  const unsigned int vectorLength = itk::PixelTraits<typename ImageType::PixelType>::Dimension;

  for (unsigned int i = 0; i < nProj; i += SLAB_SIZE)
  {
    // If nProj is not a multiple of SLAB_SIZE, the last slab will contain less than SLAB_SIZE projections
    projectionSize[2] = std::min(nProj - i, (unsigned int)SLAB_SIZE);

    // Run the back projection with a slab of SLAB_SIZE or less projections
    CUDA_back_project(projectionSize,
                      volumeSize,
                      fMatrix + 12 * i,
                      fvolIndexToProjPP + 12 * i,
                      fprojPPToProjIndex,
                      pin,
                      pout,
                      stackGPUPointer + projSize * i,
                      this->m_Geometry->GetRadiusCylindricalDetector(),
                      vectorLength);

    // Re-use the output as input
    pin = pout;
  }

  delete[] fMatrix;
  delete[] fvolIndexToProjPP;
  delete[] fprojPPToProjIndex;
}

} // end namespace rtk

#endif // end conditional definition of the class

#endif
