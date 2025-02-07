/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#ifndef rtkCudaWeidingerForwardModelImageFilter_hxx
#define rtkCudaWeidingerForwardModelImageFilter_hxx

#include "rtkConfiguration.h"
// Conditional definition of the class to pass ITKHeaderTest
#ifdef RTK_USE_CUDA

#  include "rtkCudaWeidingerForwardModelImageFilter.h"
#  include "rtkCudaUtilities.hcu"
#  include "rtkCudaWeidingerForwardModelImageFilter.hcu"

#  include <itkMacro.h>
#  include "rtkMacro.h"
#  include "itkCudaUtil.h"

namespace rtk
{

template <class TDecomposedProjections, class TMeasuredProjections, class TIncidentSpectrum, class TProjections>
CudaWeidingerForwardModelImageFilter<TDecomposedProjections, TMeasuredProjections, TIncidentSpectrum, TProjections>::
  CudaWeidingerForwardModelImageFilter()
{}

template <class TDecomposedProjections, class TMeasuredProjections, class TIncidentSpectrum, class TProjections>
void
CudaWeidingerForwardModelImageFilter<TDecomposedProjections, TMeasuredProjections, TIncidentSpectrum, TProjections>::GPUGenerateData()
{
  this->AllocateOutputs();

  const unsigned int Dimension = TDecomposedProjections::ImageDimension;
  unsigned int       nEnergies = this->m_MaterialAttenuations.rows();

  // Get the size of the input projections in a Cuda-convenient format
  int projectionSize[Dimension];
  for (unsigned int d = 0; d < Dimension; d++)
    projectionSize[d] = this->GetInputDecomposedProjections()->GetBufferedRegion().GetSize()[d];

  // Pointers to inputs and outputs
#  ifdef CUDACOMMON_VERSION_MAJOR
  float * pMatProj = (float *)(this->GetInputDecomposedProjections()->GetCudaDataManager()->GetGPUBufferPointer());
  float * pPhoCount = (float *)(this->GetInputMeasuredProjections()->GetCudaDataManager()->GetGPUBufferPointer());
  float * pSpectrum = (float *)(this->GetInputIncidentSpectrum()->GetCudaDataManager()->GetGPUBufferPointer());
  float * pProjOnes = (float *)(this->GetInputProjectionsOfOnes()->GetCudaDataManager()->GetGPUBufferPointer());
  float * pOut1 = (float *)(this->GetOutput1()->GetCudaDataManager()->GetGPUBufferPointer());
  float * pOut2 = (float *)(this->GetOutput2()->GetCudaDataManager()->GetGPUBufferPointer());
#  else
  float * pMatProj = *(float **)(this->GetInputDecomposedProjections()->GetCudaDataManager()->GetGPUBufferPointer());
  float * pPhoCount = *(float **)(this->GetInputMeasuredProjections()->GetCudaDataManager()->GetGPUBufferPointer());
  float * pSpectrum = *(float **)(this->GetInputIncidentSpectrum()->GetCudaDataManager()->GetGPUBufferPointer());
  float * pProjOnes = *(float **)(this->GetInputProjectionsOfOnes()->GetCudaDataManager()->GetGPUBufferPointer());
  float * pOut1 = *(float **)(this->GetOutput1()->GetCudaDataManager()->GetGPUBufferPointer());
  float * pOut2 = *(float **)(this->GetOutput2()->GetCudaDataManager()->GetGPUBufferPointer());
#  endif

  // Run the forward projection with a slab of SLAB_SIZE or less projections
  CUDA_WeidingerForwardModel(
    projectionSize,
    this->m_MaterialAttenuations.data_block(),
    this->m_BinnedDetectorResponse.data_block(),
    pMatProj,
    pPhoCount,
    pSpectrum,
    pProjOnes,
    pOut1,
    pOut2,
    CudaWeidingerForwardModelImageFilter<TDecomposedProjections, TMeasuredProjections, TIncidentSpectrum, TProjections>::nBins,
    nEnergies,
    CudaWeidingerForwardModelImageFilter<TDecomposedProjections, TMeasuredProjections, TIncidentSpectrum, TProjections>::nMaterials,
    this->m_NumberOfProjectionsInIncidentSpectrum,
    this->GetInputMeasuredProjections()->GetBufferedRegion().GetIndex(Dimension - 1));
}

} // end namespace rtk

#endif // end conditional definition of the class

#endif
