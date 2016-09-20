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

#include "rtkCudaParkerShortScanImageFilter.h"
#include "rtkCudaParkerShortScanImageFilter.hcu"

namespace rtk
{

CudaParkerShortScanImageFilter
::CudaParkerShortScanImageFilter()
{
}

CudaParkerShortScanImageFilter
::~CudaParkerShortScanImageFilter()
{
}

void
CudaParkerShortScanImageFilter
::GPUGenerateData()
{
  // Put the two data pointers at the same location
  float *inBuffer = *static_cast<float **>(this->GetInput()->GetCudaDataManager()->GetGPUBufferPointer());
  inBuffer += this->GetInput()->ComputeOffset( this->GetInput()->GetRequestedRegion().GetIndex() );
  float *outBuffer = *static_cast<float **>(this->GetOutput()->GetCudaDataManager()->GetGPUBufferPointer());
  outBuffer += this->GetOutput()->ComputeOffset( this->GetOutput()->GetRequestedRegion().GetIndex() );

  const std::vector<double> rotationAngles = this->GetGeometry()->GetGantryAngles();
  std::vector<double> angularGaps = this->GetGeometry()->GetAngularGapsWithNext(rotationAngles);

  // compute max. angular gap
  int maxAngularGapPos = 0;
  for (unsigned int i = 1; i < angularGaps.size(); i++)
    if(angularGaps[i] > angularGaps[maxAngularGapPos])
      maxAngularGapPos = i;

  // check if it is a short scan
  // NOTE: not a short scan if less than 20 degrees max gap
  if (this->GetGeometry()->GetSourceToDetectorDistances()[0] == 0. ||
      angularGaps[maxAngularGapPos] < itk::Math::pi / 9.)
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

  const std::map<double,unsigned int> sortedAngles = this->GetGeometry()->GetUniqueSortedAngles( rotationAngles );

  // Compute delta between first and last angle where there is weighting required
  std::map<double,unsigned int>::const_iterator itLastAngle;
  itLastAngle = sortedAngles.find(rotationAngles[maxAngularGapPos]);
  std::map<double,unsigned int>::const_iterator itFirstAngle = itLastAngle;
  itFirstAngle = (++itFirstAngle==sortedAngles.end())?sortedAngles.begin():itFirstAngle;
  const double firstAngle = itFirstAngle->first;
  double lastAngle = itLastAngle->first;
  if (lastAngle < firstAngle)
    lastAngle += 2*vnl_math::pi;
  //Delta
  double delta = 0.5 * (lastAngle - firstAngle - vnl_math::pi);
  delta = delta - 2*vnl_math::pi* floor(delta / (2*vnl_math::pi)); // between -2*PI and 2*PI

  // check for enough data
  const int geomStart = this->GetInput()->GetBufferedRegion().GetIndex()[2];
  double sox = this->GetGeometry()->GetSourceOffsetsX()[geomStart];
  double sid = this->GetGeometry()->GetSourceToIsocenterDistances()[geomStart];
  const double invsid = 1./sqrt(sid*sid+sox*sox);
  const double detectorWidth = this->GetInput()->GetSpacing()[0] *
                               this->GetInput()->GetLargestPossibleRegion().GetSize()[0];
  if (delta < atan(0.5 * detectorWidth * invsid))
    itkWarningMacro(<< "You do not have enough data for proper Parker weighting (short scan)"
                    << "Delta is " << delta * 180. / itk::Math::pi
                    << " degrees and should be more than half the beam angle, i.e. "
                    << atan(0.5 * detectorWidth * invsid) * 180. / itk::Math::pi << " degrees.");

  float proj_orig = this->GetInput()->GetOrigin()[0];

  float proj_row = this->GetInput()->GetDirection()[0][0] * this->GetInput()->GetSpacing()[0];

  float proj_col = this->GetInput()->GetDirection()[0][1] * this->GetInput()->GetSpacing()[1];

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

  // 2D matrix (numgeom * 3values) in one block for memcpy!
  // for each geometry, the following structure is used:
  // 0: sdd
  // 1: projection offset x
  // 2: gantry angle
  int geomIdx = this->GetInput()->GetRequestedRegion().GetIndex()[2];
  float *geomMatrix = new float[proj_size[2] * 5];
  if(geomMatrix == ITK_NULLPTR)
     itkExceptionMacro(<< "Couldn't allocate geomMatrix");
  for (int g = 0; g < proj_size[2]; ++g)
    {
    geomMatrix[g * 5 + 0] = this->GetGeometry()->GetSourceToDetectorDistances()[g + geomIdx];
    geomMatrix[g * 5 + 1] = this->GetGeometry()->GetSourceOffsetsX()[g + geomIdx];
    geomMatrix[g * 5 + 2] = this->GetGeometry()->GetProjectionOffsetsX()[g + geomIdx];
    geomMatrix[g * 5 + 3] = this->GetGeometry()->GetSourceToIsocenterDistances()[g + geomIdx];
    geomMatrix[g * 5 + 4] = this->GetGeometry()->GetGantryAngles()[g + geomIdx];
    }

  CUDA_parker_weight(
      proj_idx,
      proj_size,
      proj_size_buf_in,
      proj_size_buf_out,
      inBuffer, outBuffer,
      geomMatrix,
      delta, firstAngle,
      proj_orig, proj_row, proj_col
      );

  delete[] geomMatrix;
}

}
