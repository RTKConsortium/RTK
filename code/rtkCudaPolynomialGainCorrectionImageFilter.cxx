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

#include "rtkCudaPolynomialGainCorrectionImageFilter.h"
#include "rtkCudaPolynomialGainCorrectionImageFilter.hcu"

namespace rtk
{

CudaPolynomialGainCorrectionImageFilter
::CudaPolynomialGainCorrectionImageFilter()
{
}

CudaPolynomialGainCorrectionImageFilter
::~CudaPolynomialGainCorrectionImageFilter()
{
}

void
CudaPolynomialGainCorrectionImageFilter
::GPUGenerateData()
{
  // compute overlap region by cropping output region with input buffer
  OutputImageRegionType overlapRegion = this->GetOutput()->GetRequestedRegion();
  
  // Put the two data pointers at the same location
  unsigned short *inBuffer = *static_cast<unsigned short **>(this->GetInput()->GetCudaDataManager()->GetGPUBufferPointer());
  inBuffer += this->GetInput()->ComputeOffset(overlapRegion.GetIndex());
  float *outBuffer = *static_cast<float **>(this->GetOutput()->GetCudaDataManager()->GetGPUBufferPointer());
  outBuffer += this->GetOutput()->ComputeOffset( this->GetOutput()->GetRequestedRegion().GetIndex() );

  unsigned short *darkBuffer = *static_cast<unsigned short **>(m_DarkImage->GetCudaDataManager()->GetGPUBufferPointer());
  
  float *gainBuffer = *static_cast<float **>(m_GainImage->GetCudaDataManager()->GetGPUBufferPointer());
  
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

  float coefficients[2] = { static_cast<float>(m_ModelOrder), m_K };

  int LUT_size = sizeof(float)*m_PowerLut.size();
  CUDA_gain_correction(
      proj_idx_in, proj_size_in, proj_size_in_buf, proj_idx_out, proj_size_out, proj_size_out_buf,
      inBuffer, outBuffer, darkBuffer, gainBuffer, &m_PowerLut[0], LUT_size, coefficients);
}

}
