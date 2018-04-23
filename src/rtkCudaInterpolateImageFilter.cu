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
CUDA_interpolation(const int4 &inputSize,
                   float* input,
                   float* output,
                   int projectionNumber,
                   float **weights)
{
  cublasHandle_t  handle;
  cublasCreate(&handle);

  // CUDA device pointers
  int    nVoxelsOutput = inputSize.x * inputSize.y * inputSize.z;
  int    memorySizeOutput = nVoxelsOutput*sizeof(float);

  // Reset output volume
  cudaMemset((void *)output, 0, memorySizeOutput );

  for (int phase=0; phase<inputSize.w; phase++)
    {
    float weight = weights[phase][projectionNumber];
    if(weight!=0)
      {
      // Create a pointer to the "phase"-th volume in the input
      float * p = input + phase * nVoxelsOutput;

      // Add "weight" times the "phase"-th volume in the input to the output
      cublasSaxpy(handle, nVoxelsOutput, &weight, p, 1, output, 1);
      }
    }

  // Destroy Cublas context
  cublasDestroy(handle);

  CUDA_CHECK_ERROR;
}
