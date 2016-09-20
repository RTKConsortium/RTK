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

#include "rtkCudaFDKWeightProjectionFilter.h"
#include "rtkCudaFDKWeightProjectionFilter.hcu"

namespace rtk
{

CudaFDKWeightProjectionFilter
::CudaFDKWeightProjectionFilter()
{
}

CudaFDKWeightProjectionFilter
::~CudaFDKWeightProjectionFilter()
{
}

void
CudaFDKWeightProjectionFilter
::GPUGenerateData()
{
  // Get angular weights from geometry
  std::vector<double> constantProjectionFactor =
      this->GetGeometry()->GetAngularGaps(this->GetGeometry()->GetSourceAngles());
  std::vector<double> tiltAngles =
      this->GetGeometry()->GetTiltAngles();

  for(unsigned int g = 0; g < constantProjectionFactor.size(); g++)
  {
    // Add correction factor for ramp filter
    const double sdd  = this->GetGeometry()->GetSourceToDetectorDistances()[g];
    if(sdd == 0.) // Parallel
      constantProjectionFactor[g] *= 0.5;
    else        // Divergent
    {
      // See [Rit and Clackdoyle, CT meeting, 2014]
      ThreeDCircularProjectionGeometry::HomogeneousVectorType sp;
      sp = this->GetGeometry()->GetSourcePosition(g);
      sp[3] = 0.;
      const double sid  = this->GetGeometry()->GetSourceToIsocenterDistances()[g];
      constantProjectionFactor[g] *= std::abs(sdd) / (2. * sid *sid);
      constantProjectionFactor[g] *= sp.GetNorm();
    }
  }

  float proj_orig[2];
  proj_orig[0] = this->GetInput()->GetOrigin()[0];
  proj_orig[1] = this->GetInput()->GetOrigin()[1];

  float proj_row[2];
  proj_row[0] = this->GetInput()->GetDirection()[0][0] * this->GetInput()->GetSpacing()[0];
  proj_row[1] = this->GetInput()->GetDirection()[1][0] * this->GetInput()->GetSpacing()[0];

  float proj_col[2];
  proj_col[0] = this->GetInput()->GetDirection()[0][1] * this->GetInput()->GetSpacing()[1];
  proj_col[1] = this->GetInput()->GetDirection()[1][1] * this->GetInput()->GetSpacing()[1];

  int proj_idx[2];
  proj_idx[0] = this->GetInput()->GetRequestedRegion().GetIndex()[0];
  proj_idx[1] = this->GetInput()->GetRequestedRegion().GetIndex()[1];

  int proj_size[3];
  proj_size[0] = this->GetInput()->GetRequestedRegion().GetSize()[0];
  proj_size[1] = this->GetInput()->GetRequestedRegion().GetSize()[1];
  proj_size[2] = this->GetInput()->GetRequestedRegion().GetSize()[2];

  int proj_size_buf_in[2];
  proj_size_buf_in[0] = this->GetInput()->GetBufferedRegion().GetSize()[0];
  proj_size_buf_in[1] = this->GetInput()->GetBufferedRegion().GetSize()[1];

  int proj_size_buf_out[2];
  proj_size_buf_out[0] = this->GetOutput()->GetBufferedRegion().GetSize()[0];
  proj_size_buf_out[1] = this->GetOutput()->GetBufferedRegion().GetSize()[1];

  // 2D matrix (numgeom * 7 values) in one block for memcpy!
  // for each geometry, the following structure is used:
  // 0: sdd
  // 1: sid
  // 2: projection offset x
  // 3: projection offset y
  // 4: source offset x
  // 5: source offset y
  // 6: weight factor
  int geomIdx = this->GetInput()->GetRequestedRegion().GetIndex()[2];
  float *geomMatrix = new float[proj_size[2] * 7];
  if(geomMatrix == ITK_NULLPTR)
     itkExceptionMacro(<< "Couldn't allocate geomMatrix");
  for (int g = 0; g < proj_size[2]; ++g)
  {
    geomMatrix[g * 7 + 0] = this->GetGeometry()->GetSourceToDetectorDistances()[g + geomIdx];
    geomMatrix[g * 7 + 1] = this->GetGeometry()->GetSourceToIsocenterDistances()[g + geomIdx];
    geomMatrix[g * 7 + 2] = this->GetGeometry()->GetProjectionOffsetsX()[g + geomIdx];
    geomMatrix[g * 7 + 3] = this->GetGeometry()->GetProjectionOffsetsY()[g + geomIdx];
    geomMatrix[g * 7 + 4] = this->GetGeometry()->GetSourceOffsetsY()[g + geomIdx];
    geomMatrix[g * 7 + 5] = constantProjectionFactor[g + geomIdx];
    geomMatrix[g * 7 + 6] = tiltAngles[g + geomIdx];
  }

  // Put the two data pointers at the same location
  float *inBuffer = *static_cast<float **>(this->GetInput()->GetCudaDataManager()->GetGPUBufferPointer());
  inBuffer += this->GetInput()->ComputeOffset( this->GetInput()->GetRequestedRegion().GetIndex() );
  float *outBuffer = *static_cast<float **>(this->GetOutput()->GetCudaDataManager()->GetGPUBufferPointer());
  outBuffer += this->GetOutput()->ComputeOffset( this->GetOutput()->GetRequestedRegion().GetIndex() );

  CUDA_weight_projection(
      proj_idx,
      proj_size,
      proj_size_buf_in,
      proj_size_buf_out,
      inBuffer, outBuffer,
      geomMatrix,
      proj_orig, proj_row, proj_col
      );

  delete[] geomMatrix;
}

}
