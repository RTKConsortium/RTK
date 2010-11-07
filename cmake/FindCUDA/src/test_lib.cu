
#include <external_dependency.h>
#include <iostream>

__global__ void times2_kernel(int *in, int *out) {

  for (unsigned int i=0;i<blockDim.x;++i) {
    // /*const*/ unsigned int thread = threadIdx.x;
    out[threadIdx.x] = in[threadIdx.x] * MULTIPLIER;
  }
};

void times2(int* in, int* out, int dim) {
  // Setup kernel problem size
  dim3 blocksize(dim,1,1);
  dim3 gridsize(1,1,1);

  // Call kernel
  times2_kernel<<<gridsize, blocksize>>>(in, out);
}

#if test_lib_EXPORTS
#  if defined( _WIN32 ) || defined( _WIN64 )
#    define TEST_LIB_API __declspec(dllexport)
#  else
#    define TEST_LIB_API
#  endif
#else
#  define TEST_LIB_API
#endif

TEST_LIB_API int doit()
{
  cudaFree(0);
  CHECK_CUDA_ERROR();

  int h_val[DIM];
  int h_result[DIM];

  for(int i = 0; i < DIM; ++i)
    h_val[i] = i;

  // Allocate device memory
  unsigned int size = sizeof(int) * DIM;
  int* d_val;
  cudaMalloc((void**)&d_val, size);
  CHECK_CUDA_ERROR();

  int* d_result;
  cudaMalloc((void**)&d_result, size);
  CHECK_CUDA_ERROR();

  // Send input to device
  cudaMemcpy(d_val, h_val, size, cudaMemcpyHostToDevice);
  CHECK_CUDA_ERROR();

  // Call the kernel wrapper
  times2(d_val, d_result, DIM);
  CHECK_CUDA_ERROR();

  // Get back results
  cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost);
  CHECK_CUDA_ERROR();

  for(int i = 0; i < DIM; ++i)
    std::cout << h_val[i] << " * " << MULTIPLIER << " = " << h_result[i] << "\n";

  // Free memory
  cudaFree((void*)d_val);
  CHECK_CUDA_ERROR();

  cudaFree((void*)d_result);
  CHECK_CUDA_ERROR();

  return 0;
}


