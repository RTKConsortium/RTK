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
#include "rtkCudaInterpolateImageFilter.hcu"
#include "rtkCudaUtilities.hcu"

#include <itkMacro.h>

// cuda includes
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

void
CUDA_splat(const int4 &outputSize,
                   float* input,
                   float* output,
                   int projectionNumber,
                   float **weights)
{
  cublasHandle_t  handle;
  cublasCreate(&handle);

  int numel = outputSize.x * outputSize.y * outputSize.z;

  for (int phase=0; phase<outputSize.w; phase++)
    {
    float weight = weights[phase][projectionNumber];
    if(weight!=0)
      {
      // Create a pointer to the "phase"-th volume in the output
      float * p = output + phase * numel;

      // Add "weight" times the input to the "phase"-th volume in the output
      cublasSaxpy(handle, numel, &weight, input, 1, p, 1);
      }
    }

  // Destroy Cublas context
  cublasDestroy(handle);

  CUDA_CHECK_ERROR;
}
