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

// rtk includes
#include "rtkCudaCyclicDeformationImageFilter.hcu"
#include "rtkCudaUtilities.hcu"

#include <itkMacro.h>

// cuda includes
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// TEXTURES AND CONSTANTS //

__constant__ int4 c_inputSize;

//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_( S T A R T )_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

void
CUDA_linear_interpolate_along_fourth_dimension(unsigned int inputSize[4],
                                              float* input,
                                              float* output,
                                              unsigned int frameInf,
                                              unsigned int frameSup,
                                              double weightInf,
                                              double weightSup)
{
  cublasHandle_t  handle;
  cublasCreate(&handle);

  float wInf = (float) weightInf;
  float wSup = (float) weightSup;

  int numel = inputSize[0] * inputSize[1] * inputSize[2];

  cudaMemset((void *)output, 0, numel * sizeof(float) * 3);

  for (unsigned int component = 0; component < 3; component++)
    {
    // Create a pointer to the component-th component of frameInf in the input,
    // and add it weightInf times to the component-th component of the output
    float * pinf = input + frameInf * numel * 3 + component;
    cublasSaxpy(handle, numel, &wInf, pinf, 3, output + component, 3);

    // Create a pointer to the component-th component of frameSup in the input,
    // and add it weightSup times to the component-th component of the output
    float * psup = input + frameSup * numel * 3 + component;
    cublasSaxpy(handle, numel, &wSup, psup, 3, output + component, 3);
    }

  // Destroy Cublas context
  cublasDestroy(handle);

  CUDA_CHECK_ERROR;
}
