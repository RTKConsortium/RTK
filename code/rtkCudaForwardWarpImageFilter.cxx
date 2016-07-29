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

#include "rtkCudaForwardWarpImageFilter.h"
#include "rtkCudaUtilities.hcu"
#include "rtkCudaForwardWarpImageFilter.hcu"
#include "rtkHomogeneousMatrix.h"

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkMacro.h>

namespace rtk
{

CudaForwardWarpImageFilter
::CudaForwardWarpImageFilter()
{
}

void
CudaForwardWarpImageFilter
::GPUGenerateData()
{
  // Include non-zero index in matrix
  itk::Matrix<double, 4, 4> matrixIdxOutputVol;
  itk::Matrix<double, 4, 4> matrixIdxInputVol;
  itk::Matrix<double, 4, 4> indexInputToPPInputMatrix;
  itk::Matrix<double, 4, 4> indexInputToIndexDVFMatrix;
  itk::Matrix<double, 4, 4> PPOutputToIndexOutputMatrix;
  matrixIdxOutputVol.SetIdentity();
  matrixIdxInputVol.SetIdentity();
  for(unsigned int i=0; i<3; i++)
    {
    matrixIdxOutputVol[i][3] = this->GetOutput()->GetRequestedRegion().GetIndex()[i];
    matrixIdxInputVol[i][3] = this->GetInput(0)->GetBufferedRegion().GetIndex()[i];
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

  // Split the DVF into three images (one per component)
  ImageType::Pointer xCompDVF = ImageType::New();
  ImageType::Pointer yCompDVF = ImageType::New();
  ImageType::Pointer zCompDVF = ImageType::New();
  ImageType::RegionType largest = this->GetDisplacementField()->GetLargestPossibleRegion();
  xCompDVF->SetRegions(largest);
  yCompDVF->SetRegions(largest);
  zCompDVF->SetRegions(largest);
  xCompDVF->Allocate();
  yCompDVF->Allocate();
  zCompDVF->Allocate();
  itk::ImageRegionIterator<ImageType>      itxComp(xCompDVF, largest);
  itk::ImageRegionIterator<ImageType>      ityComp(yCompDVF, largest);
  itk::ImageRegionIterator<ImageType>      itzComp(zCompDVF, largest);
  itk::ImageRegionConstIterator<DVFType>   itDVF(this->GetDisplacementField(), largest);
  while(!itDVF.IsAtEnd())
    {
      itxComp.Set(itDVF.Get()[0]);
      ityComp.Set(itDVF.Get()[1]);
      itzComp.Set(itDVF.Get()[2]);
      ++itxComp;
      ++ityComp;
      ++itzComp;
      ++itDVF;
    }

  float *pinVol  = *(float**)( this->GetInput(0)->GetCudaDataManager()->GetGPUBufferPointer() );
  float *poutVol = *(float**)( this->GetOutput()->GetCudaDataManager()->GetGPUBufferPointer() );
  float *pinxDVF = *(float**)( xCompDVF->GetCudaDataManager()->GetGPUBufferPointer() );
  float *pinyDVF = *(float**)( yCompDVF->GetCudaDataManager()->GetGPUBufferPointer() );
  float *pinzDVF = *(float**)( zCompDVF->GetCudaDataManager()->GetGPUBufferPointer() );

  // Transform matrices that we will need during the ForwardWarping process
  indexInputToPPInputMatrix = rtk::GetIndexToPhysicalPointMatrix( this->GetInput(0) ).GetVnlMatrix()
                              * matrixIdxInputVol.GetVnlMatrix();

  indexInputToIndexDVFMatrix = rtk::GetPhysicalPointToIndexMatrix( this->GetDisplacementField() ).GetVnlMatrix()
                              * rtk::GetIndexToPhysicalPointMatrix( this->GetInput(0) ).GetVnlMatrix()
                              * matrixIdxInputVol.GetVnlMatrix();

  PPOutputToIndexOutputMatrix = rtk::GetPhysicalPointToIndexMatrix( this->GetOutput() ).GetVnlMatrix();

  // Convert the matrices to arrays of floats (skipping the last line, as we don't care)
  float fIndexInputToPPInputMatrix[12];
  float fIndexInputToIndexDVFMatrix[12];
  float fPPOutputToIndexOutputMatrix[12];
  for (int j = 0; j < 12; j++)
    {
    fIndexInputToIndexDVFMatrix[j] = (float) indexInputToIndexDVFMatrix[j/4][j%4];
    fPPOutputToIndexOutputMatrix[j] = (float) PPOutputToIndexOutputMatrix[j/4][j%4];
    fIndexInputToPPInputMatrix[j] = (float) indexInputToPPInputMatrix[j/4][j%4];
    }

  bool isLinear;
  if (std::string("LinearInterpolateImageFunction").compare(this->GetInterpolator()->GetNameOfClass()) == 0)
    isLinear = true;
  else if (std::string("NearestNeighborInterpolateImageFunction").compare(this->GetInterpolator()->GetNameOfClass()) == 0)
    isLinear = false;
  else
    itkGenericExceptionMacro(<< "In rtkCudaForwardWarpImageFilter: unknown interpolator");

  // Run on GPU
  CUDA_ForwardWarp(
    inputVolumeSize,
    inputDVFSize,
    outputVolumeSize,
    fIndexInputToPPInputMatrix,
    fIndexInputToIndexDVFMatrix,
    fPPOutputToIndexOutputMatrix,
    pinVol,
    pinxDVF,
    pinyDVF,
    pinzDVF,
    poutVol,
    isLinear
    );

  // Get rid of the intermediate images used to split the DVF into three components
  xCompDVF = ITK_NULLPTR;
  yCompDVF = ITK_NULLPTR;
  zCompDVF = ITK_NULLPTR;

  // The filter is inPlace
//  pinVol = poutVol;
}

} // end namespace rtk
