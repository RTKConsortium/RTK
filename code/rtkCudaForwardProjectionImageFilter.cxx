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

#include "rtkCudaForwardProjectionImageFilter.h"
#include "rtkCudaUtilities.hcu"
#include "rtkCudaForwardProjectionImageFilter.hcu"

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkMacro.h>
#include "rtkMacro.h"

namespace rtk
{

CudaForwardProjectionImageFilter
::CudaForwardProjectionImageFilter()
{
  m_DeviceVolume     = NULL;
  m_DeviceProjection = NULL;
  m_DeviceMatrix     = NULL;
  m_ExplicitGPUMemoryManagementFlag = false;
}

void
CudaForwardProjectionImageFilter
::InitDevice()
{
  // Dimension arguments in CUDA format for volume
  m_VolumeDimension[0] = this->GetInput(1)->GetRequestedRegion().GetSize()[0];
  m_VolumeDimension[1] = this->GetInput(1)->GetRequestedRegion().GetSize()[1];
  m_VolumeDimension[2] = this->GetInput(1)->GetRequestedRegion().GetSize()[2];
  // Dimension arguments in CUDA format for projections
  m_ProjectionDimension[0] = this->GetInput(0)->GetRequestedRegion().GetSize()[0];
  m_ProjectionDimension[1] = this->GetInput(0)->GetRequestedRegion().GetSize()[1];
  // Cuda initialization
  std::vector<int> devices = GetListOfCudaDevices();
  if(devices.size()>1)
    {
    cudaThreadExit();
    cudaSetDevice(devices[0]);
    }
  const float *host_volume = this->GetInput(1)->GetBufferPointer();

  CUDA_forward_project_init (m_ProjectionDimension, m_VolumeDimension,
                             m_DeviceVolume, m_DeviceProjection, m_DeviceMatrix, host_volume);
}

void
CudaForwardProjectionImageFilter
::CleanUpDevice()
{
  if(this->GetOutput()->GetRequestedRegion() != this->GetOutput()->GetBufferedRegion() )
    itkExceptionMacro(<< "Can't handle different requested and buffered regions "
                      << this->GetOutput()->GetRequestedRegion()
                      << this->GetOutput()->GetBufferedRegion() );
  CUDA_forward_project_cleanup (m_ProjectionDimension,
                                m_DeviceVolume,
                                m_DeviceProjection,
                                m_DeviceMatrix);
  m_DeviceVolume     = NULL;
  m_DeviceProjection = NULL;
  m_DeviceMatrix     = NULL;
}

void
CudaForwardProjectionImageFilter
::GenerateData()
{
  this->AllocateOutputs();
  if(!m_ExplicitGPUMemoryManagementFlag)
    this->InitDevice();
  const Superclass::GeometryType::Pointer geometry = this->GetGeometry();
  const unsigned int Dimension = ImageType::ImageDimension;
  const unsigned int iFirstProj = this->GetInput(0)->GetRequestedRegion().GetIndex(Dimension-1);
  const unsigned int nProj = this->GetInput(0)->GetRequestedRegion().GetSize(Dimension-1);
  const unsigned int nPixelsPerProj = this->GetOutput()->GetBufferedRegion().GetSize(0) *
                                      this->GetOutput()->GetBufferedRegion().GetSize(1);

  float t_step = 1; // Step in mm
  int blockSize[3] = {16,16,1};
  itk::Vector<double, 4> source_position;

  // Setting BoxMin and BoxMax
  // SR: we are using cuda textures where the pixel definition is not center but corner.
  // Therefore, we set the box limits from index to index+size instead of, for ITK,
  // index-0.5 to index+size-0.5.
  float boxMin[3];
  float boxMax[3];
  for(unsigned int i=0; i<3; i++)
    {
      boxMin[i] = this->GetInput(1)->GetBufferedRegion().GetIndex()[i];
      boxMax[i] = boxMin[i] + this->GetInput(1)->GetBufferedRegion().GetSize()[i];
    }

  // Getting Spacing
  float spacing[3];
  for(unsigned int i=0; i<3; i++)
    spacing[i] = this->GetInput(1)->GetSpacing()[i];

  // Go over each projection
  for(unsigned int iProj=iFirstProj; iProj<iFirstProj+nProj; iProj++)
    {
      // Account for system rotations
      Superclass::GeometryType::ThreeDHomogeneousMatrixType volPPToIndex;
      volPPToIndex = GetPhysicalPointToIndexMatrix( this->GetInput(1) );

      // Adding 0.5 offset to change from the centered pixel convention (ITK)
      // to the corner pixel convention (CUDA).
      for(unsigned int i=0; i<3; i++)
        volPPToIndex[i][3] += 0.5;

      // Compute matrix to transform projection index to volume index
      Superclass::GeometryType::ThreeDHomogeneousMatrixType d_matrix;
      d_matrix = volPPToIndex.GetVnlMatrix() *
                 geometry->GetProjectionCoordinatesToFixedSystemMatrix(iProj).GetVnlMatrix() *
                 GetIndexToPhysicalPointMatrix( this->GetInput() ).GetVnlMatrix();
      float matrix[4][4];
      for (int j=0; j<4; j++)
        for (int k=0; k<4; k++)
          matrix[j][k] = (float)d_matrix[j][k];

      // Set source position in volume indices
      source_position = volPPToIndex * geometry->GetSourcePosition(iProj);
      CUDA_forward_project(blockSize,
                           this->GetOutput()->GetBufferPointer() + (iProj -
                           this->GetOutput()->GetBufferedRegion().GetIndex()[2]) *
                           nPixelsPerProj,
                           m_DeviceProjection,
                           (double*)&(source_position[0]),
                           m_ProjectionDimension,
                           t_step,
                           m_DeviceMatrix,
                           (float*)&(matrix[0][0]),
                           boxMin,
                           boxMax,
                           spacing);
    }
  if(!m_ExplicitGPUMemoryManagementFlag)
    this->CleanUpDevice();
}

} // end namespace rtk
