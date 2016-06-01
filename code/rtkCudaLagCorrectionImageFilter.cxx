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

#include "rtkCudaLagCorrectionImageFilter.h"
#include "rtkCudaLagCorrectionImageFilter.hcu"

namespace rtk
{

CudaLagCorrectionImageFilter
::CudaLagCorrectionImageFilter()
{
}

CudaLagCorrectionImageFilter
::~CudaLagCorrectionImageFilter()
{
}

void
CudaLagCorrectionImageFilter
::GPUGenerateData()
{
  // compute overlap region by cropping output region with input buffer
  OutputImageRegionType overlapRegion = this->GetOutput()->GetRequestedRegion();
  //overlapRegion.Crop(this->GetInput()->GetBufferedRegion());

  // Put the two data pointers at the same location
  unsigned short *inBuffer = *static_cast<unsigned short **>(this->GetInput()->GetCudaDataManager()->GetGPUBufferPointer());
  inBuffer += this->GetInput()->ComputeOffset(overlapRegion.GetIndex());
  unsigned short *outBuffer = *static_cast<unsigned short **>(this->GetOutput()->GetCudaDataManager()->GetGPUBufferPointer());
  outBuffer += this->GetOutput()->ComputeOffset( this->GetOutput()->GetRequestedRegion().GetIndex() );

  int proj_idx_in[3];
  proj_idx_in[0] = overlapRegion.GetIndex()[0];
  proj_idx_in[1] = overlapRegion.GetIndex()[1];
  proj_idx_in[2] = overlapRegion.GetIndex()[2];

  int proj_size_in_buf[2];
  proj_size_in_buf[0] = this->GetInput()->GetBufferedRegion().GetSize()[0];
  proj_size_in_buf[1] = this->GetInput()->GetBufferedRegion().GetSize()[1];

  int proj_size_in[3];
  proj_size_in[0] = overlapRegion.GetSize()[0];
  proj_size_in[1] = overlapRegion.GetSize()[1];
  proj_size_in[2] = overlapRegion.GetSize()[2];

  int proj_idx_out[3];
  proj_idx_out[0] = this->GetOutput()->GetRequestedRegion().GetIndex()[0];
  proj_idx_out[1] = this->GetOutput()->GetRequestedRegion().GetIndex()[1];
  proj_idx_out[2] = this->GetOutput()->GetRequestedRegion().GetIndex()[2];

  int proj_size_out_buf[2];
  proj_size_out_buf[0] = this->GetOutput()->GetBufferedRegion().GetSize()[0];
  proj_size_out_buf[1] = this->GetOutput()->GetBufferedRegion().GetSize()[1];

  int proj_size_out[3];
  proj_size_out[0] = this->GetOutput()->GetRequestedRegion().GetSize()[0];
  proj_size_out[1] = this->GetOutput()->GetRequestedRegion().GetSize()[1];
  proj_size_out[2] = this->GetOutput()->GetRequestedRegion().GetSize()[2];

  float coefficients[9] = { m_B[0], m_B[1], m_B[2], m_B[3], m_ExpmA[0], m_ExpmA[1], m_ExpmA[2], m_ExpmA[3] , m_SumB};

  int S_size = sizeof(float)*m_S.size();
  CUDA_lag_correction(
      proj_idx_in, proj_size_in, proj_size_in_buf, proj_idx_out, proj_size_out, proj_size_out_buf,
      inBuffer, outBuffer, &m_S[0], S_size, coefficients);
}

}
