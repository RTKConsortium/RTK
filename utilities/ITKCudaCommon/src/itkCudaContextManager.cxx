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
#include <assert.h>
#include "itkCudaContextManager.h"
#include "cuda.h"
#include "cuda_runtime_api.h"

namespace itk
{
// static variable initialization
CudaContextManager* CudaContextManager::m_Instance = NULL;
bool CudaContextManager::m_Initialized = false;


CudaContextManager* CudaContextManager::GetInstance()
{
  if (m_Instance == NULL)
    {
    m_Instance = new CudaContextManager();
    }
  return m_Instance;
}

void CudaContextManager::DestroyInstance()
{
  delete m_Instance;
  m_Instance = NULL;
}

CudaContextManager::CudaContextManager()
{
  m_Context = 0;
  m_DeviceIdx = -1;
  m_Device = 0;

  if (!m_Initialized)
    {
    cuInit(0);
    m_Initialized = true;
    }

  std::vector<cudaDeviceProp> devices;
  m_NumberOfDevices = itk::CudaGetAvailableDevices(devices);

  CUcontext context;
  CUdevice device;

  m_DeviceIdx = itk::CudaGetMaxFlopsDev();

  CUDA_CHECK(cuDeviceGet(&device, m_DeviceIdx));

  CUDA_CHECK(cuCtxCreate(&context, CU_CTX_SCHED_AUTO, device));

  CUDA_CHECK(cuCtxSetCurrent(context));

  m_Device = device;
  m_Context = context;
}

CudaContextManager::~CudaContextManager()
{
}

int CudaContextManager::GetCurrentContext()
{
  int device = -1;
  CUDA_CHECK(cudaGetDevice(&device));
  return device;
}

} // namespace itk
