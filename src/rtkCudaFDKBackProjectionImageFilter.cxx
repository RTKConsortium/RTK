/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#include "rtkCudaFDKBackProjectionImageFilter.h"
#include "rtkCudaUtilities.hcu"
#include "rtkCudaFDKBackProjectionImageFilter.hcu"

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkMacro.h>

namespace rtk
{

CudaFDKBackProjectionImageFilter
::CudaFDKBackProjectionImageFilter()
{
}

void
CudaFDKBackProjectionImageFilter
::GPUGenerateData()
{
  const unsigned int Dimension = ImageType::ImageDimension;
  const unsigned int nProj = this->GetInput(1)->GetLargestPossibleRegion().GetSize(Dimension-1);
  const unsigned int iFirstProj = this->GetInput(1)->GetLargestPossibleRegion().GetIndex(Dimension-1);
  if (nProj>1024)
    itkGenericExceptionMacro("The CUDA voxel based back projection image filter can only handle stacks of at most 1024 projections")

  // Rotation center (assumed to be at 0 yet)
  ImageType::PointType rotCenterPoint;
  rotCenterPoint.Fill(0.0);
  itk::ContinuousIndex<double, Dimension> rotCenterIndex;
  this->GetInput(0)->TransformPhysicalPointToContinuousIndex(rotCenterPoint, rotCenterIndex);

  // Include non-zero index in matrix
  itk::Matrix<double, 4, 4> matrixIdxVol;
  matrixIdxVol.SetIdentity();
  for(unsigned int i=0; i<3; i++)
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

  float *pin  = *(float**)( this->GetInput()->GetCudaDataManager()->GetGPUBufferPointer() );
  float *pout = *(float**)( this->GetOutput()->GetCudaDataManager()->GetGPUBufferPointer() );

  float *stackGPUPointer = *(float**)( this->GetInput(1)->GetCudaDataManager()->GetGPUBufferPointer() );
  ptrdiff_t projSize = this->GetInput(1)->GetBufferedRegion().GetSize()[0] *
                       this->GetInput(1)->GetBufferedRegion().GetSize()[1];
  stackGPUPointer += projSize * (iFirstProj-this->GetInput(1)->GetBufferedRegion().GetIndex()[2]);

  // Allocate a large matrix to hold the matrix of all projections
  float *fMatrix = new float[12 * nProj];

  // Go over each projection
  for(unsigned int iProj=iFirstProj; iProj<iFirstProj+nProj; iProj++)
    {
    // Index to index matrix normalized to have a correct backprojection weight
    // (1 at the isocenter)
    ProjectionMatrixType matrix = GetIndexToIndexProjectionMatrix(iProj);

    // We correct the matrix for non zero indexes
    itk::Matrix<double, 3, 3> matrixIdxProj;
    matrixIdxProj.SetIdentity();
    for(unsigned int i=0; i<2; i++)
      //SR: 0.5 for 2D texture
      matrixIdxProj[i][2] = -1*(this->GetInput(1)->GetBufferedRegion().GetIndex()[i])+0.5;

    matrix = matrixIdxProj.GetVnlMatrix() * matrix.GetVnlMatrix() * matrixIdxVol.GetVnlMatrix();

    double perspFactor = matrix[Dimension-1][Dimension];
    for(unsigned int j=0; j<Dimension; j++)
      perspFactor += matrix[Dimension-1][j] * rotCenterIndex[j];
    matrix /= perspFactor;

    for (int j = 0; j < 12; j++)
      fMatrix[j + (iProj-iFirstProj) * 12] = matrix[j/4][j%4];
    }

  for (unsigned int i=0; i<nProj; i+=SLAB_SIZE)
    {
    // If nProj is not a multiple of SLAB_SIZE, the last slab will contain less than SLAB_SIZE projections
    projectionSize[2] = std::min(nProj-i, (unsigned int)SLAB_SIZE);

    // Run the back projection with a slab of SLAB_SIZE or less projections
    CUDA_reconstruct_conebeam(projectionSize,
                              volumeSize,
                              fMatrix + 12 * i,
                              pin,
                              pout,
                              stackGPUPointer + projSize * i
                              );

    // Re-use the output as input
    pin = pout;
    }

  delete[] fMatrix;
}

} // end namespace rtk
