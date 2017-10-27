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

#include "rtkCudaDisplacedDetectorImageFilter.h"
#include "rtkCudaDisplacedDetectorImageFilter.hcu"

namespace rtk
{

CudaDisplacedDetectorImageFilter
::CudaDisplacedDetectorImageFilter()
{
}

CudaDisplacedDetectorImageFilter
::~CudaDisplacedDetectorImageFilter()
{
}

void
CudaDisplacedDetectorImageFilter
::GPUGenerateData()
{
  // compute overlap region by cropping output region with input buffer
  OutputImageRegionType overlapRegion = this->GetOutput()->GetRequestedRegion();
  overlapRegion.Crop(this->GetInput()->GetBufferedRegion());

  // Put the two data pointers at the same location
  float *inBuffer = *static_cast<float **>(this->GetInput()->GetCudaDataManager()->GetGPUBufferPointer());
  inBuffer += this->GetInput()->ComputeOffset( overlapRegion.GetIndex() );
  float *outBuffer = *static_cast<float **>(this->GetOutput()->GetCudaDataManager()->GetGPUBufferPointer());
  outBuffer += this->GetOutput()->ComputeOffset( this->GetOutput()->GetRequestedRegion().GetIndex() );

  // nothing to do
  if ((fabs(this->GetInferiorCorner() + this->GetSuperiorCorner())
       < 0.1 * fabs(this->GetSuperiorCorner() - this->GetInferiorCorner())) || this->GetDisable())
    {
    if ( outBuffer != inBuffer )
      {
      size_t count = this->GetOutput()->GetRequestedRegion().GetSize(0);
      count *= sizeof(ImageType::PixelType);
      for(unsigned int k=0; k<this->GetOutput()->GetRequestedRegion().GetSize(2); k++)
        {
        for(unsigned int j=0; j<this->GetOutput()->GetRequestedRegion().GetSize(1); j++)
          {
          cudaMemcpy(outBuffer, inBuffer, count, cudaMemcpyDeviceToDevice);
          inBuffer += this->GetInput()->GetBufferedRegion().GetSize(0);
          outBuffer += this->GetOutput()->GetBufferedRegion().GetSize(0);
          }
        inBuffer += (this->GetInput()->GetBufferedRegion().GetSize(1)-this->GetInput()->GetRequestedRegion().GetSize(1)) * this->GetInput()->GetBufferedRegion().GetSize(0);
        outBuffer += (this->GetOutput()->GetBufferedRegion().GetSize(1)-this->GetOutput()->GetRequestedRegion().GetSize(1)) * this->GetOutput()->GetBufferedRegion().GetSize(0);
        }
      }
    return;
    }

  float proj_orig = this->GetOutput()->GetOrigin()[0];

  float proj_row = this->GetOutput()->GetDirection()[0][0] * this->GetOutput()->GetSpacing()[0];

  float proj_col = this->GetOutput()->GetDirection()[0][1] * this->GetOutput()->GetSpacing()[1];

  int proj_idx_in[3];
  proj_idx_in[0] = overlapRegion.GetIndex()[0];
  proj_idx_in[1] = overlapRegion.GetIndex()[1];
  proj_idx_in[2] = overlapRegion.GetIndex()[2];

  int proj_size_in[3];
  proj_size_in[0] = overlapRegion.GetSize()[0];
  proj_size_in[1] = overlapRegion.GetSize()[1];
  proj_size_in[2] = overlapRegion.GetSize()[2];

  int proj_size_in_buf[2];
  proj_size_in_buf[0] = this->GetInput()->GetBufferedRegion().GetSize()[0];
  proj_size_in_buf[1] = this->GetInput()->GetBufferedRegion().GetSize()[1];

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

  double theta = vnl_math_min(-1. * this->GetInferiorCorner(), this->GetSuperiorCorner());
  bool isPositiveCase = (this->GetSuperiorCorner() + this->GetInferiorCorner() > 0.) ? true : false;

  // 2D matrix (numgeom * 4 values) in one block for memcpy!
  // for each geometry, the following structure is used:
  // 0: sdd
  // 1: source offset x
  // 2: projection offset x
  // 3: sid
  int geomIdx = proj_idx_out[2];
  float *geomMatrix = new float[proj_size_out[2] * 4];
  if(geomMatrix == ITK_NULLPTR)
     itkExceptionMacro(<< "Couldn't allocate geomMatrix");
  for (int g = 0; g < proj_size_out[2]; ++g)
  {
    geomMatrix[g * 4 + 0] = this->GetGeometry()->GetSourceToDetectorDistances()[g + geomIdx];
    geomMatrix[g * 4 + 1] = this->GetGeometry()->GetSourceOffsetsX()[g + geomIdx];
    geomMatrix[g * 4 + 2] = this->GetGeometry()->GetProjectionOffsetsX()[g + geomIdx];
    geomMatrix[g * 4 + 3] = this->GetGeometry()->GetSourceToIsocenterDistances()[g + geomIdx];
  }


  CUDA_displaced_weight(
      proj_idx_in, proj_size_in, proj_size_in_buf, proj_idx_out, proj_size_out, proj_size_out_buf,
      inBuffer, outBuffer,
      geomMatrix,
      theta, isPositiveCase,
      proj_orig, proj_row, proj_col
      );

  delete[] geomMatrix;
}

}
