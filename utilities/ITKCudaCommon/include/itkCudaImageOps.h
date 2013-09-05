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
#ifndef __itkCudaImageOps_h
#define __itkCudaImageOps_h

#include "itkMacro.h"
#include "itkCudaUtil.h"

namespace itk
{
/** \class CudaImageOps
 *
 * \brief Provides the kernels for some basic Cuda Image Operations
 *
 * \ingroup ITKCudaCommon
 */

/** Create a helper Cuda Kernel class for CudaImageOps */
itkCudaKernelClassMacro(CudaImageOpsKernel);

/** CudaImageOps class definition */
class ITK_EXPORT CudaImageOps
{
public:
  /** Standard class typedefs. */
  typedef CudaImageOps Self;

  /** Get Cuda Kernel PTX sourcefilename, creates a GetCudaPTXFile method */
  itkGetCudaPTXSourceMacro(CudaImageOpsKernel);

private:
  CudaImageOps();
  virtual ~CudaImageOps();

  CudaImageOps(const Self &);                  //purposely not implemented
  void operator=(const Self &);                //purposely not implemented

};

} // end of namespace itk

#endif
