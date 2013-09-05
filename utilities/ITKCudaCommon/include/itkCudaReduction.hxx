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
#ifndef __itkCudaReduction_hxx
#define __itkCudaReduction_hxx

#include "itkMacro.h"
#include "itkCudaReduction.h"

//#define CPU_VERIFY

namespace itk
{
/**
 * Default constructor
 */
template< class TElement >
CudaReduction< TElement >
::CudaReduction()
{
  /*** Prepare Cuda GPU program ***/
  m_CudaKernelManager = CudaKernelManager::New();
  m_CudaDataManager = NULL;

}
template< class TElement >
CudaReduction< TElement >
::~CudaReduction()
{
  this->ReleaseGPUInputBuffer();
}

/**
 * Standard "PrintSelf" method.
 */
template< class TElement >
void CudaReduction< TElement >
::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  //GetTypenameInString(typeid(TElement), os);
}

template< class TElement >
unsigned int CudaReduction< TElement >
::NextPow2(unsigned int x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

template< class TElement >
bool CudaReduction< TElement >
::isPow2(unsigned int x)
{
  return ((x&(x-1)) == 0);
}

template< class TElement >
void CudaReduction< TElement >
::GetNumBlocksAndThreads(int whichKernel, int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{
  //if (whichKernel < 3)
    {
    threads = (n < maxThreads) ? this->NextPow2(n) : maxThreads;
    blocks = (n + threads - 1) / threads;
    }
  /*else
    {
    threads = (n < maxThreads*2) ? this->NextPow2((n + 1)/ 2) : maxThreads;
    blocks = (n + (threads * 2 - 1)) / (threads * 2);
    }
    */

  if (whichKernel == 6)
    {
    if (maxBlocks < blocks)
      {
      blocks = maxBlocks;
      }
    }
}

template< class TElement >
unsigned int CudaReduction< TElement >
::GetReductionKernel(int whichKernel, int blockSize, int isPowOf2)
{
  if (whichKernel != 5 && whichKernel != 6)
    {
    itkExceptionMacro(<< "Reduction kernel undefined!");
    return 0;
    }

  std::string CudaSource = CudaReduction::GetCudaPTXSource();
  // load and build program
  if (!this->m_CudaKernelManager->LoadProgramFromString(CudaSource.c_str()))
    {
    itkExceptionMacro(<< "Unable to create the CUDA program!");
    return 0;
    }

  m_blockSize = blockSize;

  std::ostringstream kernelName;
  kernelName << "reduce" << whichKernel;

  unsigned int handle = this->m_CudaKernelManager->CreateKernel(kernelName.str().c_str(), typeid(TElement));

  m_SmallBlock = 0; //(wgSize == 64);

  // NOTE: the program will get deleted when the kernel is also released
  //this->m_CudaKernelManager->ReleaseProgram();

  return handle;
}

template< class TElement >
void CudaReduction< TElement >
::AllocateGPUInputBuffer(TElement *h_idata)
{
  unsigned int bytes = m_Size * sizeof(TElement);

  m_CudaDataManager = CudaDataManager::New();
  m_CudaDataManager->SetBufferSize(bytes);
  m_CudaDataManager->SetCPUBufferPointer(h_idata);
  m_CudaDataManager->Allocate();

  if (h_idata)
    {
    m_CudaDataManager->SetGPUDirtyFlag(true);
    }
}

template< class TElement >
void CudaReduction< TElement >
::ReleaseGPUInputBuffer()
{
  if (m_CudaDataManager == (CudaDataPointer)NULL)
    {
    return;
    }

  m_CudaDataManager->Initialize();
}

template< class TElement >
TElement CudaReduction< TElement >
::RandomTest()
{
  int size = (1<<24) - 1917;    // number of elements to reduce

  this->InitializeKernel(size);

  TElement* h_idata = new TElement[size];

  for (int i = 0; i < size; i++)
    {
    // Keep the numbers small so we don't get truncation error in the sum
    h_idata[i] = (TElement)(rand() & 0xFF);
    }

  this->AllocateGPUInputBuffer(h_idata);

  TElement gpu_result = this->GPUGenerateData();
  std::cout << "Cuda result = " << gpu_result << std::endl << std::flush;

  TElement cpu_result = this->CPUGenerateData(h_idata, size);
  std::cout << "CPU result = " << cpu_result << std::endl;

  this->ReleaseGPUInputBuffer();

  delete [] h_idata;

  return 0;
}

template< class TElement >
void CudaReduction< TElement >
::InitializeKernel(unsigned int size)
{
  m_Size = size;

  // Create a testing kernel to decide block size
//   m_TestCudaKernelHandle = this->GetReductionKernel(6, 64, 1);
  //m_CudaKernelManager->ReleaseKernel(kernelHandle);

  // number of threads per block
  int maxThreads = m_SmallBlock ? 64 : 128;

  int whichKernel = 6;
  int maxBlocks = 64;

  int numBlocks = 0;
  int numThreads = 0;

  this->GetNumBlocksAndThreads(whichKernel, size, maxBlocks, maxThreads, numBlocks, numThreads);

  m_ReduceCudaKernelHandle = this->GetReductionKernel(whichKernel, numThreads, isPow2(size));
}

template< class TElement >
TElement CudaReduction< TElement >
::GPUGenerateData()
{
  unsigned int size = m_Size;

  // number of threads per block
  int maxThreads = m_SmallBlock ? 64 : 128;

  int whichKernel = 6;
  int maxBlocks = 64;
  bool cpuFinalReduction = true;
  int  cpuFinalThreshold = 1;

  int numBlocks = 0;
  int numThreads = 0;

  this->GetNumBlocksAndThreads(whichKernel, size, maxBlocks, maxThreads, numBlocks, numThreads);

  if (numBlocks == 1) cpuFinalThreshold = 1;

  // allocate output data for the result
  TElement* h_odata = (TElement*)malloc(numBlocks * sizeof(TElement));
  memset(h_odata, 0, numBlocks * sizeof(TElement));
  CudaDataPointer odata = CudaDataManager::New();
  odata->SetBufferSize(numBlocks * sizeof(TElement));
  odata->SetCPUBufferPointer(h_odata);
  odata->Allocate();
  odata->UpdateGPUBuffer();

  double dTotalTime = 0.0;

  m_GPUResult = 0;
  m_GPUResult = this->GPUReduce(size, numThreads, numBlocks, maxThreads, maxBlocks,
                                  whichKernel, cpuFinalReduction,
                                  cpuFinalThreshold, &dTotalTime,
                                  m_CudaDataManager, odata);

  // cleanup
  free(h_odata);

  return m_GPUResult;
}

template< class TElement >
TElement CudaReduction< TElement >
::GPUReduce(int  n,
             int  numThreads,
             int  numBlocks,
             int  itkNotUsed(maxThreads),
             int  itkNotUsed(maxBlocks),
             int  whichKernel,
             bool itkNotUsed(cpuFinalReduction),
             int  itkNotUsed(cpuFinalThreshold),
             double* itkNotUsed(dTotalTime),
             CudaDataPointer idata,
             CudaDataPointer odata)
{
  // arguments set up
  int argidx = 0;

  this->m_CudaKernelManager->SetKernelArgWithImage(m_ReduceCudaKernelHandle, argidx++, idata);
  this->m_CudaKernelManager->SetKernelArgWithImage(m_ReduceCudaKernelHandle, argidx++, odata);

  this->m_CudaKernelManager->SetKernelArg(m_ReduceCudaKernelHandle, argidx++, 1, &n);
  this->m_CudaKernelManager->SetKernelArg(m_ReduceCudaKernelHandle, argidx++, 1, &m_blockSize);
  int pow2 = isPow2(m_Size);
  if (whichKernel == 6)
    {
    this->m_CudaKernelManager->SetKernelArg(m_ReduceCudaKernelHandle, argidx++, 1, &pow2);
    }

  size_t globalSize = numBlocks * numThreads;
  size_t localSize = numThreads;

  // when there is only one warp per block, we need to allocate two warps
  // worth of shared memory so that we don't index shared memory out of bounds
  int sharedMemSize = (numThreads <= 32) ?
    2 * numThreads * sizeof(TElement) :
    numThreads * sizeof(TElement);

  // execute the kernel
  this->m_CudaKernelManager->LaunchKernel1D(m_ReduceCudaKernelHandle,
    globalSize, localSize, sharedMemSize);

  odata->SetGPUDirtyFlag(false);
  odata->SetCPUDirtyFlag(true);
  odata->Update();
  TElement* h_odata = (TElement*)odata->GetCPUBufferPointer();

#ifdef CPU_VERIFY
  idata->SetCPUDirtyFlag(true);
  TElement* h_idata = (TElement*)idata->GetCPUBufferPointer(); //debug
  if (!h_idata)
  {
    h_idata = (TElement*)malloc(sizeof(TElement) * n);
    idata->SetCPUBufferPointer(h_idata);
    idata->SetCPUDirtyFlag(true);
    h_idata = (TElement*)idata->GetCPUBufferPointer(); //debug
  }

  TElement CPUSum = this->CPUGenerateData(h_idata, n);
  std::cout << "CPU_VERIFY sum = " << CPUSum << std::endl;
#endif

  TElement gpu_result = 0;
  TElement c = 0;
  for (int i = 0; i < numBlocks; i++)
    {
    // Compensated sum algorithm
    TElement y = h_odata[i] - c;
    TElement t = gpu_result + y;
    c = (t - gpu_result) - y;
    gpu_result = t;
    }

  return gpu_result;
}

template< class TElement >
TElement CudaReduction< TElement >
::CPUGenerateData(TElement *data, int size)
{
    TElement sum = 0;
    TElement c = 0;
    for (int i = 0; i < size; i++)
      {
      // Compensated sum algorithm
      TElement y = data[i] - c;
      TElement t = sum + y;
      c = (t - sum) - y;
      sum = t;
      }
    m_CPUResult = sum;
    return sum;
}

} // end namespace itk

#endif
