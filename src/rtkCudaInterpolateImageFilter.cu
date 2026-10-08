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

// rtk includes
#include "rtkCudaInterpolateImageFilter.hcu"
#include "rtkCudaUtilities.hcu"

#include <itkMacro.h>

// cuda includes
#include <cuda.h>
#include <cublas_v2.h>
#include <limits>
#include <cuda_runtime.h>

void
CUDA_interpolation(const itk::SizeValueType inputSize[4],
                   float *                  input,
                   float *                  output,
                   int                      projectionNumber,
                   float **                 weights)
{
  // CUDA device pointers
  itk::SizeValueType nVoxelsOutput = inputSize[0] * inputSize[1] * inputSize[2];

#if CUBLAS_VER_MAJOR < 12
  if (nVoxelsOutput > static_cast<itk::SizeValueType>(std::numeric_limits<int>::max()))
  {
    itkGenericExceptionMacro(<< "Operations with more than INT_MAX elements require cuBLAS 12 or newer.");
  }
#endif

  cublasHandle_t handle;
  cublasCreate(&handle);
  size_t memorySizeOutput = nVoxelsOutput * sizeof(float);

  // Reset output volume
  cudaMemset((void *)output, 0, memorySizeOutput);

  for (itk::SizeValueType phase = 0; phase < inputSize[3]; phase++)
  {
    float weight = weights[phase][projectionNumber];
    if (weight != 0)
    {
      // Create a pointer to the "phase"-th volume in the input
      float * p = input + phase * nVoxelsOutput;

      // Add "weight" times the "phase"-th volume in the input to the output
#if CUBLAS_VER_MAJOR < 12
      cublasSaxpy(handle, (int)nVoxelsOutput, &weight, p, 1, output, 1);
#else
      cublasSaxpy_64(handle, nVoxelsOutput, &weight, p, 1, output, 1);
#endif
    }
  }

  // Destroy Cublas context
  cublasDestroy(handle);

  CUDA_CHECK_ERROR;
}
