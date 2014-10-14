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

#include "itkCudaDataManager.h"
//#define VERBOSE

namespace itk
{
// constructor
CudaDataManager::CudaDataManager()
{
  m_ContextManager = CudaContextManager::GetInstance();
  CUDA_CHECK(cuCtxSetCurrent(*(m_ContextManager->GetCurrentContext())));
  m_CPUBuffer = NULL;
  m_GPUBuffer = GPUMemPointer::New();
  this->Initialize();
}

CudaDataManager::~CudaDataManager()
{
  m_GPUBuffer = NULL;
  CudaContextManager::DestroyInstance();
}

void CudaDataManager::SetBufferSize(unsigned int num)
{
  m_BufferSize = num;
}

void CudaDataManager::SetBufferFlag(int flags)
{
  m_MemFlags = flags;
}

void CudaDataManager::Allocate()
{
  if (m_BufferSize > 0 && m_GPUBuffer->GetBufferSize() != m_BufferSize)
    {
    m_GPUBuffer->Allocate(m_BufferSize);
    #ifdef VERBOSE
      std::cout << this << "::Allocate Create GPU buffer of size " << m_BufferSize << " Bytes" << " : " << m_GPUBuffer->GetPointer() << std::endl;
    #endif
    m_IsGPUBufferDirty = true;
    }
}

void CudaDataManager::ForceReleaseGPUBuffer()
{
  if (m_GPUBuffer)
    {
    #ifdef VERBOSE
      std::cout << this << "::Release GPU buffer of size " << m_BufferSize << " Bytes" << " : " << m_GPUBuffer->GetPointer() << std::endl;
    #endif
    m_GPUBuffer->Release();
    m_IsGPUBufferDirty = true;
    m_IsCPUBufferDirty = false;
    }
}

void CudaDataManager::SetCPUBufferPointer(void* ptr)
{
  m_CPUBuffer = ptr;
}

void CudaDataManager::SetCPUDirtyFlag(bool isDirty)
{
  m_IsCPUBufferDirty = isDirty;
}

void CudaDataManager::SetGPUDirtyFlag(bool isDirty)
{
  m_IsGPUBufferDirty = isDirty;
}

void CudaDataManager::SetGPUBufferDirty()
{
  this->UpdateCPUBuffer();
  m_IsGPUBufferDirty = true;
}

void CudaDataManager::SetCPUBufferDirty()
{
  this->UpdateGPUBuffer();
  m_IsCPUBufferDirty = true;
}

void CudaDataManager::UpdateCPUBuffer()
{
  MutexHolderType holder(m_Mutex);

  if(m_IsGPUBufferDirty)
    {
    m_IsCPUBufferDirty = false;
    }
  else if(m_IsCPUBufferDirty && m_GPUBuffer && m_CPUBuffer)
    {
#ifdef VERBOSE
    std::cout << this << "::UpdateCPUBuffer GPU->CPU data copy " << m_GPUBuffer->GetPointer() << "->" << m_CPUBuffer << " : " << m_BufferSize << std::endl;
#endif

    CUDA_CHECK(cuCtxSetCurrent(*(this->m_ContextManager->GetCurrentContext()))); // This is necessary when running multithread to bind the host CPU thread to the right context
    CUDA_CHECK(cudaMemcpy(m_CPUBuffer, m_GPUBuffer->GetPointer(), m_BufferSize, cudaMemcpyDeviceToHost));
    m_IsCPUBufferDirty = false;
    }
}

void CudaDataManager::UpdateGPUBuffer()
{
  MutexHolderType mutexHolder(m_Mutex);
  if (m_IsGPUBufferDirty && m_GPUBuffer)
    {
    this->Allocate(); // do the allocation

    if(!m_IsCPUBufferDirty && m_CPUBuffer)
      {
#ifdef VERBOSE
      std::cout << this << "::UpdateGPUBuffer CPU->GPU data copy " << m_CPUBuffer << "->" << m_GPUBuffer->GetPointer() << " : " << m_BufferSize << std::endl;
#endif
      CUDA_CHECK(cuCtxSetCurrent(*(this->m_ContextManager->GetCurrentContext()))); // This is necessary when running multithread to bind the host CPU thread to the right context 
      CUDA_CHECK(cudaMemcpy(m_GPUBuffer->GetPointer(), m_CPUBuffer, m_BufferSize, cudaMemcpyHostToDevice));
      }
    m_IsGPUBufferDirty = false;
    }
}

void* CudaDataManager::GetGPUBufferPointer()
{
  SetCPUBufferDirty();
  return m_GPUBuffer->GetPointerPtr();
}

void* CudaDataManager::GetCPUBufferPointer()
{
  SetGPUBufferDirty();
  return m_CPUBuffer;
}

bool CudaDataManager::Update()
{
  if (m_IsGPUBufferDirty && m_IsCPUBufferDirty)
    {
    itkExceptionMacro("Cannot make up-to-date buffer because both CPU and GPU buffers are dirty");
    return false;
    }

  this->UpdateGPUBuffer();
  this->UpdateCPUBuffer();

  m_IsGPUBufferDirty = m_IsCPUBufferDirty = false;

  return true;
}

void CudaDataManager::Graft(const CudaDataManager* data)
{
  if (data)
    {
    m_BufferSize = data->m_BufferSize;
    m_ContextManager = data->m_ContextManager;
    m_GPUBuffer = data->m_GPUBuffer;
    m_CPUBuffer = data->m_CPUBuffer;
    m_IsCPUBufferDirty = data->m_IsCPUBufferDirty;
    m_IsGPUBufferDirty = data->m_IsGPUBufferDirty;
    m_TimeStamp = data->m_TimeStamp;
    }
}

void CudaDataManager::Initialize()
{
  m_BufferSize = 0;
  m_CPUBuffer = NULL;
  m_MemFlags  = 0; // default flag
  m_IsGPUBufferDirty = false;
  m_IsCPUBufferDirty = false;
}

void CudaDataManager::PrintSelf(std::ostream & os, Indent indent) const
{
  os << indent << "CudaDataManager (" << this << ")" << std::endl;
  os << indent << "m_BufferSize: " << m_BufferSize << std::endl;
  os << indent << "m_IsGPUBufferDirty: " << m_IsGPUBufferDirty << std::endl;
  os << indent << "m_GPUBuffer: " << m_GPUBuffer << std::endl;
  os << indent << "m_IsCPUBufferDirty: " << m_IsCPUBufferDirty << std::endl;
  os << indent << "m_CPUBuffer: " << m_CPUBuffer << std::endl;
}

} // namespace itk
