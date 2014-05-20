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
weighAndAddToOutput_kernel(float *input, int4 inputSize, float* output, int phase, float weight)
{
    unsigned int i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    unsigned int j = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    unsigned int k = __umul24(blockIdx.z, blockDim.z) + threadIdx.z;

    if (i >= inputSize.x || j >= inputSize.y || k >= inputSize.z)
        return;

    long int input_idx = ((phase * inputSize.z + k) * inputSize.y + j) * inputSize.x + i;
    long int output_idx = (k * inputSize.y + j) * inputSize.x + i;

    output[output_idx] += input[input_idx] * weight;
}

void
CUDA_interpolation(const int4 &inputSize,
                   float* input,
                   float* output,
                   int projectionNumber,
                   float **weights)
{
    // CUDA device pointers
    float *deviceInput;
    int    nVoxelsInput = inputSize.x * inputSize.y * inputSize.z * inputSize.w;
    int    memorySizeInput = nVoxelsInput*sizeof(float);
    float *deviceOutput;
    int    nVoxelsOutput = inputSize.x * inputSize.y * inputSize.z;
    int    memorySizeOutput = nVoxelsOutput*sizeof(float);

    cudaMalloc( (void**)&deviceInput, memorySizeInput );
    CUDA_CHECK_ERROR;
    cudaMemcpy (deviceInput, input, memorySizeInput, cudaMemcpyHostToDevice);

    cudaMalloc( (void**)&deviceOutput, memorySizeOutput);
    CUDA_CHECK_ERROR;
    cudaMemset(deviceOutput, 0, memorySizeOutput);

    // Thread Block Dimensions
    int tBlock_x = 16;
    int tBlock_y = 4;
    int tBlock_z = 4;
    int  blocksInX = (inputSize.x - 1) / tBlock_x + 1;
    int  blocksInY = (inputSize.y - 1) / tBlock_y + 1;
    int  blocksInZ = (inputSize.z - 1) / tBlock_z + 1;
    dim3 dimGrid  = dim3(blocksInX, blocksInY, blocksInZ);
    dim3 dimBlock = dim3(tBlock_x, tBlock_y, tBlock_z);
    for (int phase=0; phase<inputSize.w; phase++){
        float weight = weights[phase][projectionNumber];
        if(weight!=0)
        {
//            printf("%f\n", weight);
            weighAndAddToOutput_kernel <<< dimGrid, dimBlock >>> ( deviceInput,
                                                                   inputSize,
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
