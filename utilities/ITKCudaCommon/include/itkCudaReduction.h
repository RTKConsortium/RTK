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
#ifndef itkCudaReduction_h
#define itkCudaReduction_h

#include "itkObject.h"
#include "itkCudaDataManager.h"
#include "itkCudaKernelManager.h"
#include "itkCudaUtil.h"

namespace itk
{
/**
 * \class CudaReduction
 *
 * This class encapsulate the parallel reduction algorithm. An example
 * of this algorithm is to compute the sum of a long array in parallel.
 *
 * \ingroup ITKCudaCommon
 */

/** Create a helper Cuda Kernel class for CudaReduction */
itkCudaKernelClassMacro(CudaReductionKernel);

template <class TElement>
class ITK_EXPORT CudaReduction : public Object
{
public:
  /** Standard class type alias. */
  using Self = CudaReduction;
  using Superclass = Object;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(CudaReduction, Object);

  using CudaDataPointer = CudaDataManager::Pointer;

  itkGetMacro(CudaDataManager, CudaDataPointer);
  itkGetMacro(GPUResult, TElement);
  itkGetMacro(CPUResult, TElement);

  /** Get Cuda Kernel source as a string, creates a GetOpenCLSource method */
  itkGetCudaPTXSourceMacro(CudaReductionKernel);

  unsigned int
  NextPow2(unsigned int x);
  bool
  isPow2(unsigned int x);
  void
  GetNumBlocksAndThreads(int whichKernel, int n, int maxBlocks, int maxThreads, int & blocks, int & threads);
  unsigned int
  GetReductionKernel(int whichKernel, int blockSize, int isPowOf2);

  void
  AllocateGPUInputBuffer(TElement * h_idata = NULL);
  void
  ReleaseGPUInputBuffer();
  void
  InitializeKernel(unsigned int size);

  TElement
  RandomTest();
  TElement
  GPUGenerateData();
  TElement
  CPUGenerateData(TElement * data, int size);

  TElement
  GPUReduce(int             n,
            int             numThreads,
            int             numBlocks,
            int             maxThreads,
            int             maxBlocks,
            int             whichKernel,
            bool            cpuFinalReduction,
            int             cpuFinalThreshold,
            double *        dTotalTime,
            CudaDataPointer idata,
            CudaDataPointer odata);

protected:
  CudaReduction();
  ~CudaReduction();
  void
  PrintSelf(std::ostream & os, Indent indent) const;

  /** Cuda kernel manager for CudaFiniteDifferenceFunction class */
  CudaKernelManager::Pointer m_CudaKernelManager;
  CudaDataPointer            m_CudaDataManager;

  /* Cuda kernel handle for CudaComputeUpdate */
  int m_ReduceCudaKernelHandle;
  int m_TestCudaKernelHandle;

  unsigned int m_Size;
  bool         m_SmallBlock;
  int          m_blockSize;

  TElement m_GPUResult;
  TElement m_CPUResult;

private:
  CudaReduction(const Self &); // purposely not implemented
  void
  operator=(const Self &); // purposely not implemented
};
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkCudaReduction.hxx"
#endif

#endif
