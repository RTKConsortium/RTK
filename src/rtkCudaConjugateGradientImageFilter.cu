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
#include "rtkCudaConjugateGradientImageFilter.hcu"
#include "rtkCudaUtilities.hcu"

#include <itkMacro.h>

// cuda includes
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// TEXTURES AND CONSTANTS //

void
CUDA_copy(int size[3],
          float* in,
          float* out,
          unsigned int nComp)
{
  // Copy input volume to output
  long int memorySizeOutput = size[0] * size[1] * size[2] * sizeof(float) * nComp;
  cudaMemcpy(out, in, memorySizeOutput, cudaMemcpyDeviceToDevice);
}

void
CUDA_copy(int size[3],
          double* in,
          double* out,
          unsigned int nComp)
{
  // Copy input volume to output
  long int memorySizeOutput = size[0] * size[1] * size[2] * sizeof(double) * nComp;
  cudaMemcpy(out, in, memorySizeOutput, cudaMemcpyDeviceToDevice);
}

void
CUDA_subtract( int size[3],
               float* out,
               float* toBeSubtracted,
               unsigned int nComp)
{
  cublasHandle_t  handle;
  cublasCreate(&handle);

  int numel = size[0] * size[1] * size[2] * nComp;

  const float alpha = -1.0;
  cublasSaxpy(handle, numel, &alpha, toBeSubtracted, 1, out, 1);

  // Destroy Cublas context
  cublasDestroy(handle);

  CUDA_CHECK_ERROR;
}

void
CUDA_subtract( int size[3],
               double* out,
               double* toBeSubtracted,
               unsigned int nComp)
{
  cublasHandle_t  handle;
  cublasCreate(&handle);

  int numel = size[0] * size[1] * size[2] * nComp;

  const double alpha = -1.0;
  cublasDaxpy(handle, numel, &alpha, toBeSubtracted, 1, out, 1);

  // Destroy Cublas context
  cublasDestroy(handle);

  CUDA_CHECK_ERROR;
}

void
CUDA_conjugate_gradient( int size[3],
                         float* Xk,
                         float* Rk,
                         float* Pk,
                         float* APk,
                         unsigned int nComp)
{
  cublasHandle_t  handle;
  cublasCreate(&handle);

  float eps = 1e-30;

  int numel = size[0] * size[1] * size[2] * nComp;

  // Compute Rk_square = sum(Rk(:).^2) by cublas
  float Rk_square = 0;
  cublasSdot (handle, numel, Rk, 1, Rk, 1, &Rk_square);

  // Compute alpha_k = Rk_square / sum(Pk(:) .* APk(:))
  float Pk_APk = 0;
  cublasSdot (handle, numel, Pk, 1, APk, 1, &Pk_APk);

  const float alpha_k = Rk_square / (Pk_APk + eps);
  const float minus_alpha_k = -alpha_k;

  // Compute Xk+1 = Xk + alpha_k * Pk
  cublasSaxpy(handle, numel, &alpha_k, Pk, 1, Xk, 1);

   // Compute Rk+1 = Rk - alpha_k * APk
  cublasSaxpy(handle, numel, &minus_alpha_k, APk, 1, Rk, 1);

  // Compute beta_k = sum(Rk+1(:).^2) / Rk_square
  float Rkplusone_square = 0;
  cublasSdot (handle, numel, Rk, 1, Rk, 1, &Rkplusone_square);

  float beta_k = Rkplusone_square / (Rk_square + eps);
  float one = 1.0;

  // Compute Pk+1 = Rk+1 + beta_k * Pk
  // This requires two cublas functions,
  // since axpy would store the result in the wrong array
  cublasSscal(handle, numel, &beta_k, Pk, 1);
  cublasSaxpy(handle, numel, &one, Rk, 1, Pk, 1);

  // Destroy Cublas context
  cublasDestroy(handle);

  CUDA_CHECK_ERROR;
}

void
CUDA_conjugate_gradient( int size[3],
                         double* Xk,
                         double* Rk,
                         double* Pk,
                         double* APk,
                         unsigned int nComp)
{
  cublasHandle_t  handle;
  cublasCreate(&handle);

  double eps = 1e-30;

  int numel = size[0] * size[1] * size[2] * nComp;

  // Compute Rk_square = sum(Rk(:).^2) by cublas
  double Rk_square = 0;
  cublasDdot (handle, numel, Rk, 1, Rk, 1, &Rk_square);

  // Compute alpha_k = Rk_square / sum(Pk(:) .* APk(:))
  double Pk_APk = 0;
  cublasDdot (handle, numel, Pk, 1, APk, 1, &Pk_APk);

  const double alpha_k = Rk_square / (Pk_APk + eps);
  const double minus_alpha_k = -alpha_k;

  // Compute Xk+1 = Xk + alpha_k * Pk
  cublasDaxpy(handle, numel, &alpha_k, Pk, 1, Xk, 1);

   // Compute Rk+1 = Rk - alpha_k * APk
  cublasDaxpy(handle, numel, &minus_alpha_k, APk, 1, Rk, 1);

  // Compute beta_k = sum(Rk+1(:).^2) / Rk_square
  double Rkplusone_square = 0;
  cublasDdot (handle, numel, Rk, 1, Rk, 1, &Rkplusone_square);

  double beta_k = Rkplusone_square / (Rk_square + eps);
  double one = 1.0;

  // Compute Pk+1 = Rk+1 + beta_k * Pk
  // This requires two cublas functions,
  // since axpy would store the result in the wrong array
  cublasDscal(handle, numel, &beta_k, Pk, 1);
  cublasDaxpy(handle, numel, &one, Rk, 1, Pk, 1);

  // Destroy Cublas context
  cublasDestroy(handle);

  CUDA_CHECK_ERROR;
}
