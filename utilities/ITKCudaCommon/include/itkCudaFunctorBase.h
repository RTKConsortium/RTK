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
#ifndef __itkCudaFunctorBase_h
#define __itkCudaFunctorBase_h

#include "itkCudaKernelManager.h"

namespace itk
{
namespace Functor
{

/** \class CudaFunctorBase
 *
 * \brief Base functor class for Cuda functor image filters.
 *
 * \ingroup ITKCudaCommon
 */
class CudaFunctorBase
{
public:

  // constructor
  CudaFunctorBase() {
  }

  // destructor
  ~CudaFunctorBase() {
  }

  /** Setup Cuda kernel arguments for this functor.
   * \return Current argument index to set additional arguments in the Cuda kernel. */
  virtual int SetCudaKernelArguments(CudaKernelManager::Pointer KernelManager, int KernelHandle) = 0;

};

} // end of namespace Functor
} // end of namespace itk

#endif
