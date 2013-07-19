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
  this->AllocateOutputs();

  const unsigned int Dimension = ImageType::ImageDimension;
  const unsigned int nProj = this->GetInput(1)->GetLargestPossibleRegion().GetSize(Dimension-1);
  const unsigned int iFirstProj = this->GetInput(1)->GetLargestPossibleRegion().GetIndex(Dimension-1);

  // Ramp factor is the correction for ramp filter which did not account for the
  // divergence of the beam
  const GeometryPointer geometry = dynamic_cast<GeometryType *>(this->GetGeometry().GetPointer() );

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
  int projectionSize[2];
  projectionSize[0] = this->GetInput(1)->GetBufferedRegion().GetSize()[0];
  projectionSize[1] = this->GetInput(1)->GetBufferedRegion().GetSize()[1];

  int volumeSize[3];
  volumeSize[0] = this->GetOutput()->GetRequestedRegion().GetSize()[0];
  volumeSize[1] = this->GetOutput()->GetRequestedRegion().GetSize()[1];
  volumeSize[2] = this->GetOutput()->GetRequestedRegion().GetSize()[2];

  float *pin  = *(float**)( this->GetInput()->GetCudaDataManager()->GetGPUBufferPointer() );
  float *pout = *(float**)( this->GetOutput()->GetCudaDataManager()->GetGPUBufferPointer() );

  // Go over each projection
  for(unsigned int iProj=iFirstProj; iProj<iFirstProj+nProj; iProj++)
    {
    // Extract the current slice
    ProjectionImagePointer projection = this->GetProjection<ProjectionImageType>(iProj);

    // Index to index matrix normalized to have a correct backprojection weight
    // (1 at the isocenter)
    ProjectionMatrixType matrix = GetIndexToIndexProjectionMatrix(iProj);

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

    CUDA_reconstruct_conebeam(projectionSize,
                              volumeSize,
                              fMatrix,
                              pin,
                              pout,
                              *(float**)( projection->GetCudaDataManager()->GetGPUBufferPointer() )
                              );
    pin = pout;
    }
}

} // end namespace rtk
