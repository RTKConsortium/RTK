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

#ifndef rtkCudaWeidingerForwardModelImageFilter_hxx
#define rtkCudaWeidingerForwardModelImageFilter_hxx

#include "rtkConfiguration.h"
//Conditional definition of the class to pass ITKHeaderTest
#ifdef RTK_USE_CUDA

#include "rtkCudaWeidingerForwardModelImageFilter.h"
#include "rtkCudaUtilities.hcu"
#include "rtkCudaWeidingerForwardModelImageFilter.hcu"

#include <itkMacro.h>
#include "rtkMacro.h"
#include "itkCudaUtil.h"

namespace rtk
{

template <class TMaterialProjections,
          class TPhotonCounts,
          class TSpectrum,
          class TProjections>
CudaWeidingerForwardModelImageFilter< TMaterialProjections, TPhotonCounts, TSpectrum, TProjections >
::CudaWeidingerForwardModelImageFilter()
{
}

template <class TMaterialProjections,
          class TPhotonCounts,
          class TSpectrum,
          class TProjections>
void
CudaWeidingerForwardModelImageFilter< TMaterialProjections, TPhotonCounts, TSpectrum, TProjections >
::GPUGenerateData()
{
  this->AllocateOutputs();

  const unsigned int Dimension = TMaterialProjections::ImageDimension;

  // Get the size of the input projections in a Cuda-convenient format
  int projectionSize[Dimension];
  for (unsigned int d=0; d<Dimension; d++)
    projectionSize[d] = this->GetInputMaterialProjections()->GetBufferedRegion().GetSize()[d];

  // Pointers to inputs and outputs
  float *pMatProj = *(float**)( this->GetInputMaterialProjections()->GetCudaDataManager()->GetGPUBufferPointer() );
  float *pPhoCount = *(float**)( this->GetInputPhotonCounts()->GetCudaDataManager()->GetGPUBufferPointer() );
  float *pSpectrum = *(float**)( this->GetInputSpectrum()->GetCudaDataManager()->GetGPUBufferPointer() );
  float *pProjOnes = *(float**)( this->GetInputProjectionsOfOnes()->GetCudaDataManager()->GetGPUBufferPointer() );
  float *pOut1 = *(float**)( this->GetOutput1()->GetCudaDataManager()->GetGPUBufferPointer() );
  float *pOut2 = *(float**)( this->GetOutput2()->GetCudaDataManager()->GetGPUBufferPointer() );

  // Run the forward projection with a slab of SLAB_SIZE or less projections
  CUDA_WeidingerForwardModel( projectionSize,
                              this->m_MaterialAttenuations.GetVnlMatrix().data_block(),
                              this->m_BinnedDetectorResponse.GetVnlMatrix().data_block(),
                              pMatProj,
                              pPhoCount,
                              pSpectrum,
                              pProjOnes,
                              pOut1,
                              pOut2,
                              CudaWeidingerForwardModelImageFilter< TMaterialProjections, TPhotonCounts, TSpectrum, TProjections >::nBins,
                              CudaWeidingerForwardModelImageFilter< TMaterialProjections, TPhotonCounts, TSpectrum, TProjections >::nEnergies,
                              CudaWeidingerForwardModelImageFilter< TMaterialProjections, TPhotonCounts, TSpectrum, TProjections >::nMaterials);

}

} // end namespace rtk

#endif //end conditional definition of the class

#endif
