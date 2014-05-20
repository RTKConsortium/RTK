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

__global__
void
splat_kernel(float *input, int4 outputSize, float* output, int phase, float weight)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= outputSize.x || j >= outputSize.y || k >= outputSize.z || phase >= outputSize.w)
        return;

    long int output_idx = ((phase * outputSize.z + k) * outputSize.y + j) * outputSize.x + i;
    long int input_idx = (k * outputSize.y + j) * outputSize.x + i;

    output[output_idx] += input[input_idx] * weight;
}

void
CUDA_splat(const int4 &outputSize,
                   float* input,
                   float* output,
                   int projectionNumber,
                   float **weights)
{
//        printf("In Cuda_splat\n");

    // CUDA device pointers
    float *deviceInput;
    unsigned long long int      nVoxelsInput = outputSize.x * outputSize.y * outputSize.z;
    unsigned long long int      memorySizeInput = nVoxelsInput*sizeof(float);
    float *deviceOutput;
    unsigned long long int      nVoxelsOutput = outputSize.x * outputSize.y * outputSize.z * outputSize.w;
    unsigned long long int      memorySizeOutput = nVoxelsOutput*sizeof(float);
//    printf("In Cuda_splat : memorySizeInput = %u\n", memorySizeInput);
//    printf("In Cuda_splat : memorySizeOutput = %u\n", memorySizeOutput);

//    printf("In Cuda_splat : About to malloc input\n");
    cudaMalloc( (void**)&deviceInput, memorySizeInput );
    CUDA_CHECK_ERROR;
    cudaMemcpy (deviceInput, input, memorySizeInput, cudaMemcpyHostToDevice);
//    printf("In Cuda_splat : malloc input OK\n");

//    printf("In Cuda_splat : About to malloc output\n");
    cudaMalloc( (void**)&deviceOutput, memorySizeOutput);
    CUDA_CHECK_ERROR;
    cudaMemcpy(deviceOutput, output, memorySizeOutput, cudaMemcpyHostToDevice); //The filter is in place, thus the output needs to be copied (and not set to zero)
//    printf("In Cuda_splat : malloc output OK\n");

    // Thread Block Dimensions
    int tBlock_x = 16;
    int tBlock_y = 4;
    int tBlock_z = 4;
    int  blocksInX = (outputSize.x - 1) / tBlock_x + 1;
    int  blocksInY = (outputSize.y - 1) / tBlock_y + 1;
    int  blocksInZ = (outputSize.z - 1) / tBlock_z + 1;
    dim3 dimGrid  = dim3(blocksInX, blocksInY, blocksInZ);
    dim3 dimBlock = dim3(tBlock_x, tBlock_y, tBlock_z);
    for (int phase=0; phase<outputSize.w; phase++){
        float weight = weights[phase][projectionNumber];
        if(weight!=0)
        {
//            printf("%f\n", weight);
            splat_kernel <<< dimGrid, dimBlock >>> ( deviceInput,
                                                       outputSize,
                                                       deviceOutput,
                                                       phase,
                                                       weight);
        }
    }

    cudaMemcpy (output, deviceOutput, memorySizeOutput, cudaMemcpyDeviceToHost);

    // Release memory
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
}
