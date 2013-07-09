/*=========================================================================
 *
 *  Copyright Insight Software Consortium
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

#include <stdio.h>

/** This is for parallel reduction and is modified from NVIDIA's example code */

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
    __device__ inline operator T *()
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }

    __device__ inline operator const T *() const
    {
        extern __shared__ int __smem[];
        return (T *)__smem;
    }
};

// specialize for double to avoid unaligned memory
// access compile errors
template<>
struct SharedMemory<double>
{
    __device__ inline operator double *()
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }

    __device__ inline operator const double *() const
    {
        extern __shared__ double __smem_d[];
        return (double *)__smem_d;
    }
};

// The following defines are set during runtime compilation, see reduction.cpp
// #define T float
// #define blockSize 128
// #define nIsPow2 1
/*
    This version is completely unrolled.  It uses a template parameter to achieve
    optimal code for any (power of 2) number of threads.  This requires a switch
    statement in the host code to handle all the different thread block sizes at
    compile time.

    Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
    If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <class T>
__device__ void reduce5(T *g_idata, T *g_odata, unsigned int n, unsigned int blockSize)
{
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + threadIdx.x;

    T mySum = (i < n) ? g_idata[i] : 0;

    if (i + blockSize < n)
        mySum += g_idata[i+blockSize];

    sdata[tid] = mySum;
    __syncthreads();

    // do reduction in shared mem
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] = mySum = mySum + sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] = mySum = mySum + sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { sdata[tid] = mySum = mySum + sdata[tid +  64]; } __syncthreads(); }

    if (tid < 32)
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile T *smem = sdata;
        if (blockSize >=  64) { smem[tid] = mySum = mySum + smem[tid + 32]; }
        if (blockSize >=  32) { smem[tid] = mySum = mySum + smem[tid + 16]; }
        if (blockSize >=  16) { smem[tid] = mySum = mySum + smem[tid +  8]; }
        if (blockSize >=   8) { smem[tid] = mySum = mySum + smem[tid +  4]; }
        if (blockSize >=   4) { smem[tid] = mySum = mySum + smem[tid +  2]; }
        if (blockSize >=   2) { smem[tid] = mySum = mySum + smem[tid +  1]; }
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

/*
    This version adds multiple elements per thread sequentially.  This reduces the overall
    cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
    (Brent's Theorem optimization)

    Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
    If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <class T>
__device__ void reduce6(T *g_idata, T *g_odata, unsigned int n, unsigned int blockSize, bool nIsPow2)
{
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;

    T mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        mySum += g_idata[i];

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n)
            mySum += g_idata[i+blockSize];

        i += gridSize;
    }

    // each thread puts its local sum into shared memory
    sdata[tid] = mySum;
    __syncthreads();

    // do reduction in shared mem
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] = mySum = mySum + sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] = mySum = mySum + sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { sdata[tid] = mySum = mySum + sdata[tid +  64]; } __syncthreads(); }

    if (tid < 32)
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile T *smem = sdata;

        if (blockSize >=  64) { smem[tid] = mySum = mySum + smem[tid + 32]; }
        if (blockSize >=  32) { smem[tid] = mySum = mySum + smem[tid + 16]; }
        if (blockSize >=  16) { smem[tid] = mySum = mySum + smem[tid +  8]; }
        if (blockSize >=   8) { smem[tid] = mySum = mySum + smem[tid +  4]; }
        if (blockSize >=   4) { smem[tid] = mySum = mySum + smem[tid +  2]; }
        if (blockSize >=   2) { smem[tid] = mySum = mySum + smem[tid +  1]; }
    }
    
    // write result for this block to global mem
    if (tid == 0)        
        g_odata[blockIdx.x] = sdata[0];
}

#define MAKE_reduce5_TEMPLATE_IMPL(_exportname_, T) \
  extern "C" __global__ void _exportname_(T* g_idata, T *g_odata, unsigned int n, unsigned int blockSize)\
  {\
    reduce5<T>(g_idata, g_odata, n, blockSize);\
  }

#define MAKE_reduce6_TEMPLATE_IMPL(_exportname_, T) \
  extern "C" __global__ void _exportname_(T* g_idata, T *g_odata, unsigned int n, unsigned int blockSize, bool nIsPow2)\
  {\
    reduce6<T>(g_idata, g_odata, n, blockSize, nIsPow2);\
  }

#define MAKE_IMPL(_func_)\
  MAKE_##_func_##_TEMPLATE_IMPL(_func_##_c, char);\
  MAKE_##_func_##_TEMPLATE_IMPL(_func_##_uc, unsigned char);\
  MAKE_##_func_##_TEMPLATE_IMPL(_func_##_s, short);\
  MAKE_##_func_##_TEMPLATE_IMPL(_func_##_i, int);\
  MAKE_##_func_##_TEMPLATE_IMPL(_func_##_ui, unsigned int);\
  MAKE_##_func_##_TEMPLATE_IMPL(_func_##_f, float);\
  MAKE_##_func_##_TEMPLATE_IMPL(_func_##_d, double);

// Export the template implementations
MAKE_IMPL(reduce5);
MAKE_IMPL(reduce6);
