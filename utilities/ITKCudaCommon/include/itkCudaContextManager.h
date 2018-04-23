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
#ifndef __itkCudaContextManager_h
#define __itkCudaContextManager_h

#include "itkCudaUtil.h"
#include <itkLightObject.h>
#include "itkCudaWin32Header.h"

//
// Singleton class for CudaContextManager
//

/** \class CudaContextManager
 *
 * \brief Class to store the Cuda context.
 *
 * \ingroup ITKCudaCommon
 */
namespace itk
{
class ITKCudaCommon_EXPORT CudaContextManager : public LightObject
{
public:

  static CudaContextManager* GetInstance();

  static void DestroyInstance();
  
  CUcontext* GetCurrentContext();

  int GetCurrentDevice();

private:

  CudaContextManager();
  ~CudaContextManager();

  CUcontext m_Context;
  int m_Device;
  int m_DeviceIdx;
  int m_NumberOfDevices;

  static CudaContextManager* m_Instance;
  static bool m_Initialized;
};
} // namespace itk

#endif
