// rtk includes
#include "rtkCudaFFTRampImageFilter.hcu"
#include "rtkCudaUtilities.hcu"

#include <itkMacro.h>

// cuda includes
#include <cuda.h>
#include <cufft.h>

__global__
void
multiply_kernel(cufftComplex *projFFT, int3 fftDimension, cufftComplex *kernelFFT, unsigned int Blocks_Y,
                float invBlocks_Y)
{
  unsigned int blockIdx_z = __float2uint_rd(blockIdx.y * invBlocks_Y);
  unsigned int blockIdx_y = blockIdx.y - __umul24(blockIdx_z, Blocks_Y);
  unsigned int i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  unsigned int j = __umul24(blockIdx_y, blockDim.y) + threadIdx.y;
  unsigned int k = __umul24(blockIdx_z, blockDim.z) + threadIdx.z;

  if (i >= fftDimension.x || j >= fftDimension.y || k >= fftDimension.z)
    return;

  long int proj_idx = i + (j + k * fftDimension.y ) * fftDimension.x;

  cufftComplex result;
  result.x = projFFT[proj_idx].x * kernelFFT[i].x - projFFT[proj_idx].y * kernelFFT[i].y;
  result.y = projFFT[proj_idx].y * kernelFFT[i].x + projFFT[proj_idx].x * kernelFFT[i].y;
  projFFT[proj_idx] = result;
}

void
CUDA_fft_convolution(const int3 &inputDimension, float *projection, cufftComplex *kernelFFT)
{
  // CUDA device pointers
  float *deviceProjection;
  int    nPixelsProjection = inputDimension.x*inputDimension.y*inputDimension.z;
  int    memorySizeProjection = nPixelsProjection*sizeof(float);

  cudaMalloc( (void**)&deviceProjection, memorySizeProjection );
  CUDA_CHECK_ERROR;
  cudaMemcpy (deviceProjection, projection, memorySizeProjection, cudaMemcpyHostToDevice);

  cufftComplex *deviceProjectionFFT;
  int3          fftDimension = inputDimension;
  fftDimension.x = inputDimension.x / 2 + 1;

  int memorySizeProjectionFFT = fftDimension.x*fftDimension.y*fftDimension.z*sizeof(cufftComplex);
  cudaMalloc( (void**)&deviceProjectionFFT, memorySizeProjectionFFT);
  CUDA_CHECK_ERROR;

  cufftComplex *deviceKernelFFT;
  int           memorySizeKernelFFT = fftDimension.x*sizeof(cufftComplex);
  cudaMalloc( (void**)&deviceKernelFFT, memorySizeKernelFFT);
  CUDA_CHECK_ERROR;
  cudaMemcpy (deviceKernelFFT, kernelFFT, memorySizeKernelFFT, cudaMemcpyHostToDevice);

  // 3D FFT
  cufftHandle fftFwd;
  cufftResult result;
  if(fftDimension.z==1)
    result = cufftPlan2d(&fftFwd, inputDimension.y, inputDimension.x, CUFFT_R2C);
  else
    result = cufftPlan3d(&fftFwd, inputDimension.z, inputDimension.y, inputDimension.x, CUFFT_R2C);
  CUFFT_CHECK_ERROR(result);
  result = cufftSetCompatibilityMode(fftFwd, CUFFT_COMPATIBILITY_FFTW_ALL);
  CUFFT_CHECK_ERROR(result);
  result = cufftExecR2C(fftFwd, deviceProjection, deviceProjectionFFT);
  CUFFT_CHECK_ERROR(result);

  // Thread Block Dimensions
  int tBlock_x = 16;
  int tBlock_y = 4;
  int tBlock_z = 4;

  int  blocksInX = (fftDimension.x - 1) / tBlock_x + 1;
  int  blocksInY = (fftDimension.y - 1) / tBlock_y + 1;
  int  blocksInZ = (fftDimension.z - 1) / tBlock_z + 1;
  dim3 dimGrid  = dim3(blocksInX, blocksInY*blocksInZ);
  dim3 dimBlock = dim3(tBlock_x, tBlock_y, tBlock_z);
  multiply_kernel <<< dimGrid, dimBlock >>> ( deviceProjectionFFT, fftDimension, deviceKernelFFT, blocksInY, 1.0f/(float)blocksInY );
  CUDA_CHECK_ERROR;

  // 3D inverse FFT
  cufftHandle fftInv;
  if(fftDimension.z==1)
    result = cufftPlan2d(&fftInv, inputDimension.y, inputDimension.x, CUFFT_C2R);
  else
    result = cufftPlan3d(&fftInv, inputDimension.z, inputDimension.y, inputDimension.x, CUFFT_C2R);
  CUFFT_CHECK_ERROR(result);
  result = cufftSetCompatibilityMode(fftInv, CUFFT_COMPATIBILITY_FFTW_ALL);
  CUFFT_CHECK_ERROR(result);
  result = cufftExecC2R(fftInv, deviceProjectionFFT, deviceProjection);
  CUFFT_CHECK_ERROR(result);

  cudaMemcpy (projection, deviceProjection, memorySizeProjection, cudaMemcpyDeviceToHost);

  // Release memory
  cufftDestroy(fftFwd);
  cufftDestroy(fftInv);
  cudaFree(deviceProjection);
  cudaFree(deviceProjectionFFT);
  cudaFree(deviceKernelFFT);
}
