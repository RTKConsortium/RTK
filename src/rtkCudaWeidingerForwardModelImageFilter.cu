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

/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
/*****************
*  rtk #includes *
*****************/
#include "rtkCudaUtilities.hcu"
#include "rtkConfiguration.h"
#include "rtkCudaWeidingerForwardModelImageFilter.hcu"

/*****************
*  C   #includes *
*****************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*****************
* CUDA #includes *
*****************/
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define IDX2D(r,c,cols) = ((r)*(cols)+(c))

// CONSTANTS //////////////////////////////////////////////////////////////
__constant__ int3 c_projSize;
__constant__ float c_materialAttenuations[3 * 150];
__constant__ float c_binnedDetectorResponse[5 * 150];
////////////////////////////////////////////////////////////////////////////

//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_( S T A R T )_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

template<unsigned int nBins, unsigned int nEnergies, unsigned int nMaterials>
__global__
void kernel_forward_model(float* pMatProj, float* pPhoCount, float* pSpectrum, float* pProjOnes, float* pOut1, float* pOut2)
{
  unsigned int i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  unsigned int j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
  unsigned int k = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

  if (i >= c_projSize.x || j >= c_projSize.y || k >= c_projSize.z)
    {
    return;
    }

  // Index row major in the projection
  long int first_proj_idx = i + j * c_projSize.x; // To determine the efficient spectrum
  long int proj_idx = i + (j + k*c_projSize.y)*(c_projSize.x); // For all the rest

  // Compute the efficient spectrum at the current pixel
  float efficientSpectrum[nBins * nEnergies];
  for (unsigned int r=0; r<nBins; r++)
    for (unsigned int c=0; c<nEnergies; c++)
      efficientSpectrum[c + nEnergies * r] = pSpectrum[c + nEnergies * first_proj_idx] * c_binnedDetectorResponse[c + nEnergies * r];

  // Get attenuation factors at each energy from material projections
  float attenuationFactors[nEnergies];
  matrix_matrix_multiply(c_materialAttenuations,
                         (float*)&pMatProj[proj_idx * nMaterials],
                         attenuationFactors,
                         nEnergies,
                         1,
                         nMaterials);

  for (unsigned int r=0; r<nEnergies; r++)
    attenuationFactors[r] = std::exp(-attenuationFactors[r]);

  // Get the expected photon counts through these attenuations
  float expectedCounts[nBins];
  matrix_matrix_multiply(efficientSpectrum,
                         attenuationFactors,
                         expectedCounts,
                         nBins,
                         1,
                         nEnergies);

  // Get intermediate variables used in the computation of the first output
  float oneMinusRatios[nBins];
  for (unsigned int r=0; r<nBins; r++)
    oneMinusRatios[r] = 1 - (pPhoCount[proj_idx * nBins + r] / expectedCounts[r]);

  // Form an intermediate variable used for the gradient of the cost function,
  // (the derivation of the exponential implies that a m_MaterialAttenuations
  // gets out), by equivalent of element-wise product with implicit extension
  float intermForGradient[nEnergies * nMaterials];
  for (unsigned int r=0; r<nEnergies; r++)
    for (unsigned int c=0; c<nMaterials; c++)
      intermForGradient[c + nMaterials * r] = c_materialAttenuations[c + nMaterials * r] * attenuationFactors[r];

  // Multiply by the spectrum
  float interm2ForGradient[nBins * nMaterials];
  matrix_matrix_multiply(efficientSpectrum,
                         intermForGradient,
                         interm2ForGradient,
                         nBins,
                         nMaterials,
                         nEnergies);

  // Take the opposite
  for (unsigned int r=0; r<nBins; r++)
    for (unsigned int c=0; c<nMaterials; c++)
      interm2ForGradient[c + nMaterials * r] *= -1;

  // Compute the product with oneMinusRatios, with implicit extension
  for (unsigned int r=0; r<nBins; r++)
    for (unsigned int c=0; c<nMaterials; c++)
      interm2ForGradient[c + nMaterials * r] *= oneMinusRatios[r];

  // Finally, compute the vector to be written in first output
  // by summing on the bins
  for (unsigned int r=0; r<nBins; r++)
    for (unsigned int c=0; c<nMaterials; c++)
      pOut1[proj_idx * nMaterials + c] += interm2ForGradient[c + nMaterials * r];

  // Now compute output2

  // Form an intermediate variable used for the hessian of the cost function,
  // (the double derivation of the exponential implies that a m_MaterialAttenuations^2
  // gets out), by equivalent of element-wise product with implicit extension
  float intermForHessian[nEnergies * nMaterials * nMaterials];
  for (unsigned int r=0; r<nEnergies; r++)
    for (unsigned int c=0; c<nMaterials; c++)
      for (unsigned int c2=0; c2<nMaterials; c2++)
        intermForHessian[(r * nMaterials + c) * nMaterials + c2] = c_materialAttenuations[c + nMaterials * r] * c_materialAttenuations[c2 + nMaterials * r] * attenuationFactors[r];

  // Multiply by the spectrum
  float interm2ForHessian[nBins * nMaterials * nMaterials];
  matrix_matrix_multiply(efficientSpectrum,
                         intermForHessian,
                         interm2ForHessian,
                         nBins,
                         nMaterials * nMaterials,
                         nEnergies);

  // Sum on the bins
  for (unsigned int r=0; r<nBins; r++)
    for (unsigned int c=0; c<nMaterials * nMaterials; c++)
      pOut2[proj_idx * nMaterials * nMaterials + c] += interm2ForHessian[c + nMaterials * nMaterials * r];

  // Multiply by the projection of ones
  for (unsigned int c=0; c<nMaterials * nMaterials; c++)
       pOut2[proj_idx * nMaterials * nMaterials + c] *= pProjOnes[proj_idx];
}

//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-( E N D )-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

///////////////////////////////////////////////////////////////////////////
// FUNCTION: CUDA_WeidingerForwardModel /////////////////////////////
void
CUDA_WeidingerForwardModel( int projectionSize[3],
                            float* materialAttenuations,
                            float* binnedDetectorResponse,
                            float* pMatProj,
                            float* pPhoCount,
                            float* pSpectrum,
                            float* pProjOnes,
                            float* pOut1,
                            float* pOut2,
                            unsigned int nBins,
                            unsigned int nEnergies,
                            unsigned int nMaterials)
{
cudaMemcpyToSymbol(c_projSize, projectionSize, sizeof(int3));
cudaMemcpyToSymbol(c_binnedDetectorResponse, &(binnedDetectorResponse[0]), nBins * nEnergies * sizeof(float));
cudaMemcpyToSymbol(c_materialAttenuations, &(materialAttenuations[0]), nMaterials * nEnergies * sizeof(float));

// Set both outputs to zeros
cudaMemset((void *)pOut1, 0, projectionSize[0] * projectionSize[1] * projectionSize[2] * nMaterials * sizeof(float) );
cudaMemset((void *)pOut2, 0, projectionSize[0] * projectionSize[1] * projectionSize[2] * nMaterials * nMaterials * sizeof(float) );

dim3 dimBlock  = dim3(4, 4, 4);
dim3 dimGrid = dim3(iDivUp(projectionSize[0], dimBlock.x), iDivUp(projectionSize[1], dimBlock.y), iDivUp(projectionSize[2], dimBlock.z));

if (nBins==5 || nEnergies == 150 || nMaterials == 3)
  kernel_forward_model<5, 150, 3> <<< dimGrid, dimBlock >>> (pMatProj, pPhoCount, pSpectrum, pProjOnes, pOut1, pOut2);
else
  itkGenericExceptionMacro(<< "The CUDA version of WeidingerForwardModel works with hard-coded parameters, currently set to nBins=5, nEnergies=150 and nMaterials=3")
CUDA_CHECK_ERROR;
}
