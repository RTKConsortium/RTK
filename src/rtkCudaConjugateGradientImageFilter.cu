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
#include "rtkCudaUtilities.hcu"
#include "rtkCudaConjugateGradientImageFilter.hcu"

// cuda includes
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// TEXTURES AND CONSTANTS //

void
CUDA_copy(long int numberOfElements,
          float* in,
          float* out)
{
  // Copy input volume to output
  long int memorySizeOutput = numberOfElements * sizeof(float);
  cudaMemcpy(out, in, memorySizeOutput, cudaMemcpyDeviceToDevice);
}

void
CUDA_copy(long int numberOfElements,
          double* in,
          double* out)
{
  // Copy input volume to output
  long int memorySizeOutput = numberOfElements * sizeof(double);
  cudaMemcpy(out, in, memorySizeOutput, cudaMemcpyDeviceToDevice);
}

void
CUDA_subtract( long int numberOfElements,
               float* out,
               float* toBeSubtracted)
{
  cublasHandle_t  handle;
  cublasCreate(&handle);

  const float alpha = -1.0;
  cublasSaxpy(handle, numberOfElements, &alpha, toBeSubtracted, 1, out, 1);

  // Destroy Cublas context
  cublasDestroy(handle);

  CUDA_CHECK_ERROR;
}

void
CUDA_subtract( long int numberOfElements,
               double* out,
               double* toBeSubtracted)
{
  cublasHandle_t  handle;
  cublasCreate(&handle);

  const double alpha = -1.0;
  cublasDaxpy(handle, numberOfElements, &alpha, toBeSubtracted, 1, out, 1);

  // Destroy Cublas context
  cublasDestroy(handle);

  CUDA_CHECK_ERROR;
}

void
CUDA_conjugate_gradient( long int numberOfElements,
                         float* Xk,
                         float* Rk,
                         float* Pk,
                         float* APk)
{
  cublasHandle_t  handle;
  cublasCreate(&handle);

  float eps = 1e-30;

  // Compute Rk_square = sum(Rk(:).^2) by cublas
  float Rk_square = 0;
  cublasSdot (handle, numberOfElements, Rk, 1, Rk, 1, &Rk_square);

  // Compute alpha_k = Rk_square / sum(Pk(:) .* APk(:))
  float Pk_APk = 0;
  cublasSdot (handle, numberOfElements, Pk, 1, APk, 1, &Pk_APk);

  const float alpha_k = Rk_square / (Pk_APk + eps);
  const float minus_alpha_k = -alpha_k;

  // Compute Xk+1 = Xk + alpha_k * Pk
  cublasSaxpy(handle, numberOfElements, &alpha_k, Pk, 1, Xk, 1);

   // Compute Rk+1 = Rk - alpha_k * APk
  cublasSaxpy(handle, numberOfElements, &minus_alpha_k, APk, 1, Rk, 1);

  // Compute beta_k = sum(Rk+1(:).^2) / Rk_square
  float Rkplusone_square = 0;
  cublasSdot (handle, numberOfElements, Rk, 1, Rk, 1, &Rkplusone_square);

  float beta_k = Rkplusone_square / (Rk_square + eps);
  float one = 1.0;

  // Compute Pk+1 = Rk+1 + beta_k * Pk
  // This requires two cublas functions,
  // since axpy would store the result in the wrong array
  cublasSscal(handle, numberOfElements, &beta_k, Pk, 1);
  cublasSaxpy(handle, numberOfElements, &one, Rk, 1, Pk, 1);

  // Destroy Cublas context
  cublasDestroy(handle);

  CUDA_CHECK_ERROR;
}

void
CUDA_conjugate_gradient( long int numberOfElements,
                         double* Xk,
                         double* Rk,
                         double* Pk,
                         double* APk)
{
  cublasHandle_t  handle;
  cublasCreate(&handle);

  double eps = 1e-30;

  // Compute Rk_square = sum(Rk(:).^2) by cublas
  double Rk_square = 0;
  cublasDdot (handle, numberOfElements, Rk, 1, Rk, 1, &Rk_square);

  // Compute alpha_k = Rk_square / sum(Pk(:) .* APk(:))
  double Pk_APk = 0;
  cublasDdot (handle, numberOfElements, Pk, 1, APk, 1, &Pk_APk);

  const double alpha_k = Rk_square / (Pk_APk + eps);
  const double minus_alpha_k = -alpha_k;

  // Compute Xk+1 = Xk + alpha_k * Pk
  cublasDaxpy(handle, numberOfElements, &alpha_k, Pk, 1, Xk, 1);

   // Compute Rk+1 = Rk - alpha_k * APk
  cublasDaxpy(handle, numberOfElements, &minus_alpha_k, APk, 1, Rk, 1);

  // Compute beta_k = sum(Rk+1(:).^2) / Rk_square
  double Rkplusone_square = 0;
  cublasDdot (handle, numberOfElements, Rk, 1, Rk, 1, &Rkplusone_square);

  double beta_k = Rkplusone_square / (Rk_square + eps);
  double one = 1.0;

  // Compute Pk+1 = Rk+1 + beta_k * Pk
  // This requires two cublas functions,
  // since axpy would store the result in the wrong array
  cublasDscal(handle, numberOfElements, &beta_k, Pk, 1);
  cublasDaxpy(handle, numberOfElements, &one, Rk, 1, Pk, 1);

  // Destroy Cublas context
  cublasDestroy(handle);

  CUDA_CHECK_ERROR;
}
