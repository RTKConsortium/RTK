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

#ifndef __rtkCudaWarpForwardProjectionImageFilter_hxx
#define __rtkCudaWarpForwardProjectionImageFilter_hxx

#include "rtkCudaWarpForwardProjectionImageFilter.h"
#include "rtkCudaUtilities.hcu"
#include "rtkCudaWarpForwardProjectionImageFilter.hcu"
#include "rtkHomogeneousMatrix.h"
#include "rtkMacro.h"

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkMacro.h>
#include <itkImageRegionIterator.h>
#include <itkImageAlgorithm.h>
#include "itkCudaUtil.h"

namespace rtk
{

CudaWarpForwardProjectionImageFilter
::CudaWarpForwardProjectionImageFilter():
  m_StepSize(1)
{
  this->SetNumberOfRequiredInputs(2);
}

void
CudaWarpForwardProjectionImageFilter
::SetInputProjectionStack(const InputImageType* ProjectionStack)
{
  this->SetInput(0, const_cast<InputImageType*>(ProjectionStack));
}

void
CudaWarpForwardProjectionImageFilter
::SetInputVolume(const InputImageType* Volume)
{
  this->SetInput(1, const_cast<InputImageType*>(Volume));
}

void
CudaWarpForwardProjectionImageFilter
::SetDisplacementField(const DVFType* DVF)
{
  this->SetInput("DisplacementField", const_cast<DVFType*>(DVF));
}

CudaWarpForwardProjectionImageFilter::InputImageType::Pointer
CudaWarpForwardProjectionImageFilter
::GetInputProjectionStack()
{
  return static_cast< InputImageType * >
          ( this->itk::ProcessObject::GetInput(0) );
}

CudaWarpForwardProjectionImageFilter::InputImageType::Pointer
CudaWarpForwardProjectionImageFilter
::GetInputVolume()
{
  return static_cast< InputImageType * >
          ( this->itk::ProcessObject::GetInput(1) );
}

CudaWarpForwardProjectionImageFilter::DVFType::Pointer
CudaWarpForwardProjectionImageFilter
::GetDisplacementField()
{
  return static_cast< DVFType * >
          ( this->itk::ProcessObject::GetInput("DisplacementField") );
}

void
CudaWarpForwardProjectionImageFilter
::GenerateInputRequestedRegion()
{
  Superclass::GenerateInputRequestedRegion();
  
  // Since we do not know where the DVF points, the 
  // whole input volume is required. 
  this->GetInputVolume()->SetRequestedRegionToLargestPossibleRegion();

  // The requested region on the projection stack input is the same as the output requested region
  this->GetInputProjectionStack()->SetRequestedRegion(this->GetOutput()->GetRequestedRegion());

#if ITK_VERSION_MAJOR < 4 || (ITK_VERSION_MAJOR == 4 && ITK_VERSION_MINOR < 8)
  this->GetDisplacementField()->SetRequestedRegionToLargestPossibleRegion();
#else  
  // Determine the smallest region of the deformation field that fully
  // contains the physical space covered by the input volume's requested
  // region
  DVFType::Pointer    fieldPtr = this->GetDisplacementField();
  InputImageType::Pointer  inputPtr = this->GetInputVolume();
  if ( fieldPtr.IsNotNull() )
    {
    // tolerance for origin and spacing depends on the size of pixel
    // tolerance for direction is a fraction of the unit cube.
    const itk::SpacePrecisionType coordinateTol = this->GetCoordinateTolerance() * inputPtr->GetSpacing()[0]; // use first dimension spacing

    bool DefFieldSameInformation =
       (inputPtr->GetOrigin().GetVnlVector().is_equal(fieldPtr->GetOrigin().GetVnlVector(), coordinateTol))
    && (inputPtr->GetSpacing().GetVnlVector().is_equal(fieldPtr->GetSpacing().GetVnlVector(), coordinateTol))
    && (inputPtr->GetDirection().GetVnlMatrix().as_ref().is_equal(fieldPtr->GetDirection().GetVnlMatrix(), this->GetDirectionTolerance()));

    if (DefFieldSameInformation)
      {
      fieldPtr->SetRequestedRegion( inputPtr->GetRequestedRegion() );
      }
    else
      {
      typedef DVFType::RegionType DisplacementRegionType;

      DisplacementRegionType fieldRequestedRegion = itk::ImageAlgorithm::EnlargeRegionOverBox(inputPtr->GetRequestedRegion(),
                                                                                         inputPtr.GetPointer(),
                                                                                         fieldPtr.GetPointer());
      fieldPtr->SetRequestedRegion( fieldRequestedRegion );
      }
    if ( !fieldPtr->VerifyRequestedRegion() )
      {
      fieldPtr->SetRequestedRegion( fieldPtr->GetLargestPossibleRegion() );
      }
    }
#endif
}


void
CudaWarpForwardProjectionImageFilter
::GPUGenerateData()
{
  itk::Matrix<double, 4, 4> matrixIdxInputVol;
  itk::Matrix<double, 4, 4> indexInputToPPInputMatrix;
  itk::Matrix<double, 4, 4> indexInputToIndexDVFMatrix;
  itk::Matrix<double, 4, 4> PPInputToIndexInputMatrix;  
  matrixIdxInputVol.SetIdentity();
  for(unsigned int i=0; i<3; i++)
    {
    matrixIdxInputVol[i][3] = this->GetInputVolume()->GetBufferedRegion().GetIndex()[i]; // Should 0.5 be added here ?
    }
  
  const Superclass::GeometryType::Pointer geometry = this->GetGeometry();
  const unsigned int Dimension = InputImageType::ImageDimension;
  const unsigned int iFirstProj = this->GetInputProjectionStack()->GetRequestedRegion().GetIndex(Dimension-1);
  const unsigned int nProj = this->GetInputProjectionStack()->GetRequestedRegion().GetSize(Dimension-1);
  const unsigned int nPixelsPerProj = this->GetOutput()->GetBufferedRegion().GetSize(0) *
    this->GetOutput()->GetBufferedRegion().GetSize(1);

  itk::Vector<double, 4> source_position;

  // Setting BoxMin and BoxMax
  // SR: we are using cuda textures where the pixel definition is not center but corner.
  // Therefore, we set the box limits from index to index+size instead of, for ITK,
  // index-0.5 to index+size-0.5.
  float boxMin[3];
  float boxMax[3];
  for(unsigned int i=0; i<3; i++)
    {
    boxMin[i] = this->GetInputVolume()->GetBufferedRegion().GetIndex()[i]+0.5;
    boxMax[i] = boxMin[i] + this->GetInputVolume()->GetBufferedRegion().GetSize()[i]-1.0;
    }

  // Getting Spacing
  float spacing[3];
  for(unsigned int i=0; i<3; i++)
    {
    spacing[i] = this->GetInputVolume()->GetSpacing()[i];
    }

  // Cuda convenient format for dimensions
  int projectionSize[3];
  projectionSize[0] = this->GetOutput()->GetBufferedRegion().GetSize()[0];
  projectionSize[1] = this->GetOutput()->GetBufferedRegion().GetSize()[1];

  int volumeSize[3];
  volumeSize[0] = this->GetInputVolume()->GetBufferedRegion().GetSize()[0];
  volumeSize[1] = this->GetInputVolume()->GetBufferedRegion().GetSize()[1];
  volumeSize[2] = this->GetInputVolume()->GetBufferedRegion().GetSize()[2];
  
  int inputDVFSize[3];
  inputDVFSize[0] = this->GetDisplacementField()->GetBufferedRegion().GetSize()[0];
  inputDVFSize[1] = this->GetDisplacementField()->GetBufferedRegion().GetSize()[1];
  inputDVFSize[2] = this->GetDisplacementField()->GetBufferedRegion().GetSize()[2];
  
  float *pin = *(float**)( this->GetInputProjectionStack()->GetCudaDataManager()->GetGPUBufferPointer() );
  float *pout = *(float**)( this->GetOutput()->GetCudaDataManager()->GetGPUBufferPointer() );
  float *pvol = *(float**)( this->GetInputVolume()->GetCudaDataManager()->GetGPUBufferPointer() );
  float *pDVF = *(float**)( this->GetDisplacementField()->GetCudaDataManager()->GetGPUBufferPointer() );

  // Transform matrices that we will need during the warping process
  indexInputToPPInputMatrix = rtk::GetIndexToPhysicalPointMatrix( this->GetInputVolume().GetPointer() ).GetVnlMatrix()
                              * matrixIdxInputVol.GetVnlMatrix();

  indexInputToIndexDVFMatrix = rtk::GetPhysicalPointToIndexMatrix( this->GetDisplacementField().GetPointer() ).GetVnlMatrix()
                              * rtk::GetIndexToPhysicalPointMatrix( this->GetInputVolume().GetPointer() ).GetVnlMatrix()
                              * matrixIdxInputVol.GetVnlMatrix();

  PPInputToIndexInputMatrix = rtk::GetPhysicalPointToIndexMatrix( this->GetInputVolume().GetPointer() ).GetVnlMatrix();

  // Convert the matrices to arrays of floats (skipping the last line, as we don't care)
  float fIndexInputToIndexDVFMatrix[12];
  float fPPInputToIndexInputMatrix[12];
  float fIndexInputToPPInputMatrix[12];
  for (int j = 0; j < 12; j++)
    {
    fIndexInputToIndexDVFMatrix[j] = (float) indexInputToIndexDVFMatrix[j/4][j%4];
    fPPInputToIndexInputMatrix[j] = (float) PPInputToIndexInputMatrix[j/4][j%4];
    fIndexInputToPPInputMatrix[j] = (float) indexInputToPPInputMatrix[j/4][j%4];
    }

  // Account for system rotations
  Superclass::GeometryType::ThreeDHomogeneousMatrixType volPPToIndex;
  volPPToIndex = GetPhysicalPointToIndexMatrix( this->GetInputVolume().GetPointer() );

  // Compute matrix to translate the pixel indices on the volume and the detector
  // if the Requested region has non-zero index
  Superclass::GeometryType::ThreeDHomogeneousMatrixType projIndexTranslation, volIndexTranslation;
  projIndexTranslation.SetIdentity();
  volIndexTranslation.SetIdentity();
  for(unsigned int i=0; i<3; i++)
    {
    projIndexTranslation[i][3] = this->GetOutput()->GetRequestedRegion().GetIndex(i);
    volIndexTranslation[i][3] = -this->GetInputVolume()->GetBufferedRegion().GetIndex(i);

    // Adding 0.5 offset to change from the centered pixel convention (ITK)
    // to the corner pixel convention (CUDA).
    volPPToIndex[i][3] += 0.5;
    }

  // Compute matrices to transform projection index to volume index, one per projection
  float* matrices = new float[12 * nProj];
  float* source_positions = new float[4 * nProj];

  // Go over each projection
  for(unsigned int iProj = iFirstProj; iProj < iFirstProj + nProj; iProj++)
    {
    Superclass::GeometryType::ThreeDHomogeneousMatrixType d_matrix;
    d_matrix =
      volIndexTranslation.GetVnlMatrix() *
      volPPToIndex.GetVnlMatrix() *
      geometry->GetProjectionCoordinatesToFixedSystemMatrix(iProj).GetVnlMatrix() *
      rtk::GetIndexToPhysicalPointMatrix( this->GetInput() ).GetVnlMatrix() *
      projIndexTranslation.GetVnlMatrix();
    for (int j=0; j<3; j++) // Ignore the 4th row
      for (int k=0; k<4; k++)
        matrices[(j + 3 * (iProj-iFirstProj))*4+k] = (float)d_matrix[j][k];

    // Compute source position in volume indices
    source_position= volPPToIndex * geometry->GetSourcePosition(iProj);

    // Copy it into a single large array
    for (unsigned int d=0; d<3; d++)
      source_positions[(iProj-iFirstProj)*3 + d] = source_position[d]; // Ignore the 4th component
    }

  int projectionOffset = 0;
  for (unsigned int i=0; i<nProj; i+=SLAB_SIZE)
    {
    // If nProj is not a multiple of SLAB_SIZE, the last slab will contain less than SLAB_SIZE projections
    projectionSize[2] = std::min(nProj-i, (unsigned int)SLAB_SIZE);
    projectionOffset = iFirstProj + i - this->GetOutput()->GetBufferedRegion().GetIndex(2);

    CUDA_warp_forward_project(projectionSize,
                        volumeSize,
                        inputDVFSize,
                        (float*)&(matrices[12 * i]),
                        pin + nPixelsPerProj * projectionOffset,
                        pout + nPixelsPerProj * projectionOffset,
                        pvol,
                        m_StepSize,
                        (float*)&(source_positions[3 * i]),
                        boxMin,
                        boxMax,
                        spacing,
                        pDVF,
                        fIndexInputToIndexDVFMatrix,
                        fPPInputToIndexInputMatrix,
                        fIndexInputToPPInputMatrix
                        );
    }

  delete[] matrices;
  delete[] source_positions;
}

} // end namespace rtk

#endif
