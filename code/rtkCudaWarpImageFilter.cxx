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

#include "rtkCudaWarpImageFilter.h"
#include "rtkCudaUtilities.hcu"
#include "rtkCudaWarpImageFilter.hcu"
#include "rtkHomogeneousMatrix.h"

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkMacro.h>

namespace rtk
{

CudaWarpImageFilter
::CudaWarpImageFilter()
{
}

void
CudaWarpImageFilter
::GPUGenerateData()
{
  // Include non-zero index in matrix
  itk::Matrix<double, 4, 4> matrixIdxOutputVol;
  itk::Matrix<double, 4, 4> matrixIdxInputVol;
  itk::Matrix<double, 4, 4> indexOutputToPPOutputMatrix;
  itk::Matrix<double, 4, 4> indexOutputToIndexDVFMatrix;
  itk::Matrix<double, 4, 4> PPInputToIndexInputMatrix;
  matrixIdxOutputVol.SetIdentity();
  matrixIdxInputVol.SetIdentity();
  for(unsigned int i=0; i<3; i++)
    {
    matrixIdxOutputVol[i][3] = this->GetOutput()->GetRequestedRegion().GetIndex()[i];
    matrixIdxInputVol[i][3] = this->GetInput()->GetBufferedRegion().GetIndex()[i]; // Should 0.5 be added here ?
    }

  // Cuda convenient format for dimensions
  int inputVolumeSize[3];
  inputVolumeSize[0] = this->GetInput(0)->GetBufferedRegion().GetSize()[0];
  inputVolumeSize[1] = this->GetInput(0)->GetBufferedRegion().GetSize()[1];
  inputVolumeSize[2] = this->GetInput(0)->GetBufferedRegion().GetSize()[2];

  int inputDVFSize[3];
  inputDVFSize[0] = this->GetDisplacementField()->GetBufferedRegion().GetSize()[0];
  inputDVFSize[1] = this->GetDisplacementField()->GetBufferedRegion().GetSize()[1];
  inputDVFSize[2] = this->GetDisplacementField()->GetBufferedRegion().GetSize()[2];

  int outputVolumeSize[3];
  outputVolumeSize[0] = this->GetOutput()->GetBufferedRegion().GetSize()[0];
  outputVolumeSize[1] = this->GetOutput()->GetBufferedRegion().GetSize()[1];
  outputVolumeSize[2] = this->GetOutput()->GetBufferedRegion().GetSize()[2];

  float *pinVol  = *(float**)( this->GetInput(0)->GetCudaDataManager()->GetGPUBufferPointer() );
  float *poutVol = *(float**)( this->GetOutput()->GetCudaDataManager()->GetGPUBufferPointer() );
  float *pDVF = *(float**)( this->GetDisplacementField()->GetCudaDataManager()->GetGPUBufferPointer() );

  // Transform matrices that we will need during the warping process
  indexOutputToPPOutputMatrix = rtk::GetIndexToPhysicalPointMatrix( this->GetOutput() ).GetVnlMatrix()
                              * matrixIdxOutputVol.GetVnlMatrix();

  indexOutputToIndexDVFMatrix = rtk::GetPhysicalPointToIndexMatrix( this->GetDisplacementField() ).GetVnlMatrix()
                              * rtk::GetIndexToPhysicalPointMatrix( this->GetOutput() ).GetVnlMatrix()
                              * matrixIdxOutputVol.GetVnlMatrix();

  PPInputToIndexInputMatrix = rtk::GetPhysicalPointToIndexMatrix( this->GetInput(0) ).GetVnlMatrix();

  // Convert the matrices to arrays of floats (skipping the last line, as we don't care)
  float fIndexOutputToPPOutputMatrix[12];
  float fIndexOutputToIndexDVFMatrix[12];
  float fPPInputToIndexInputMatrix[12];
  for (int j = 0; j < 12; j++)
    {
    fIndexOutputToIndexDVFMatrix[j] = (float) indexOutputToIndexDVFMatrix[j/4][j%4];
    fPPInputToIndexInputMatrix[j] = (float) PPInputToIndexInputMatrix[j/4][j%4];
    fIndexOutputToPPOutputMatrix[j] = (float) indexOutputToPPOutputMatrix[j/4][j%4];
    }

  bool isLinear;
  if (std::string("LinearInterpolateImageFunction").compare(this->GetInterpolator()->GetNameOfClass()) == 0)
    isLinear = true;
  else if (std::string("NearestNeighborInterpolateImageFunction").compare(this->GetInterpolator()->GetNameOfClass()) == 0)
    isLinear = false;
  else
    itkGenericExceptionMacro(<< "In rtkCudaWarpImageFilter: unknown interpolator");

  // Run on GPU
  CUDA_warp(
    inputVolumeSize,
    inputDVFSize,
    outputVolumeSize,
    fIndexOutputToPPOutputMatrix,
    fIndexOutputToIndexDVFMatrix,
    fPPInputToIndexInputMatrix,
    pinVol,
    poutVol,
    pDVF,
    isLinear
    );

  // The filter is inPlace
  pinVol = poutVol;
}

} // end namespace rtk
