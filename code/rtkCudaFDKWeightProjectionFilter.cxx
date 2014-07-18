//
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
      constantProjectionFactor[g] *= sdd / (2. * sid *sid);
      constantProjectionFactor[g] *= sp.GetNorm();
    }
  }

  float proj_orig[3];
  proj_orig[0] = this->GetInput()->GetOrigin()[0];
  proj_orig[1] = this->GetInput()->GetOrigin()[1];
  proj_orig[2] = this->GetInput()->GetOrigin()[2];

  float proj_row[3];
  proj_row[0] = this->GetInput()->GetDirection()[0][0] * this->GetInput()->GetSpacing()[0];
  proj_row[1] = this->GetInput()->GetDirection()[1][0] * this->GetInput()->GetSpacing()[0];
  proj_row[2] = this->GetInput()->GetDirection()[2][0] * this->GetInput()->GetSpacing()[0];

  float proj_col[3];
  proj_col[0] = this->GetInput()->GetDirection()[0][1] * this->GetInput()->GetSpacing()[1];
  proj_col[1] = this->GetInput()->GetDirection()[1][1] * this->GetInput()->GetSpacing()[1];
  proj_col[2] = this->GetInput()->GetDirection()[2][1] * this->GetInput()->GetSpacing()[1];

  int proj_size[3];
  proj_size[0] = this->GetInput()->GetBufferedRegion().GetSize()[0];
  proj_size[1] = this->GetInput()->GetBufferedRegion().GetSize()[1];
  proj_size[2] = this->GetInput()->GetBufferedRegion().GetSize()[2];

  // 2D matrix (numgeom * 6values) in one block for memcpy!
  // for each geometry, the following structure is used:
  // 0: sdd
  // 1: projection offset x
  // 2: projection offset y
  // 3: source offset x
  // 4: source offset y
  // 5: weight factor
  int geomIdx = this->GetInput()->GetBufferedRegion().GetIndex()[2];
  float *geomMatrix = new float[proj_size[2] * 7];
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

  float *inBuffer = *static_cast<float **>(this->GetInput()->GetCudaDataManager()->GetGPUBufferPointer());
  float *outBuffer = *static_cast<float **>(this->GetOutput()->GetCudaDataManager()->GetGPUBufferPointer());

  CUDA_weight_projection(
      proj_size,
      inBuffer, outBuffer,
      geomMatrix,
      proj_orig, proj_row, proj_col
      );
}

}
