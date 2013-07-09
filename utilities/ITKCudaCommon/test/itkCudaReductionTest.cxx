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

/**
 * Test program for itkGPUImage class.
 * This program shows how to use GPU image and GPU program.
 */
//#include "pathToOpenCLSourceCode.h"
#include "itkCudaImage.h"
#include "itkCudaReduction.h"

int itkCudaReductionTest(int argc, char *argv[])
{
  if (argc > 2)
    {
    std::cout << "received " << argc << " arguments, but didn't expect any more than 1."
      << "first ignored argument: " << argv[2] << std::endl;
    }
  int numPixels = 256;
  if (argc > 1)
    {
    numPixels  = atoi(argv[1]);
    }

  // create input
  typedef int ElementType;

  itk::CudaReduction<ElementType>::Pointer summer = itk::CudaReduction<ElementType>::New();
  summer->InitializeKernel(numPixels);
  unsigned int bytes = numPixels * sizeof(ElementType);
  ElementType* h_idata = (ElementType*)malloc(bytes);

  for (int i = 0; i < numPixels; i++)
    {
    h_idata[i] = static_cast<ElementType>(1);
    }

  summer->AllocateGPUInputBuffer(h_idata);
  summer->GPUGenerateData();
  ElementType GPUsum = summer->GetGPUResult();
  summer->ReleaseGPUInputBuffer();
  int status = EXIT_FAILURE;
  if (GPUsum == static_cast<ElementType>(numPixels))
    {
    std::cout << "GPU reduction to sum passed, sum = " << GPUsum << ", numPixels = " << numPixels << std::endl;
    status = EXIT_SUCCESS;
    }
  else
    {
    std::cout << "Expected sum to be " << numPixels << ", GPUReduction computed " << GPUsum << " which is wrong." << std::endl;
    status = EXIT_FAILURE;
    }
  int CPUsum = summer->CPUGenerateData(h_idata, numPixels);
  if (CPUsum == static_cast<ElementType>(numPixels))
    {
    std::cout << "CPU reduction to sum passed, sum = " << CPUsum << ", numPixels = " << numPixels << std::endl;
    }
  else
    {
    std::cout << "Expected sum to be " << numPixels << ", CPUReduction computed " << CPUsum << " which is wrong." << std::endl;
    status = EXIT_FAILURE;
    }

  return status;
}
