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

#include "itkCudaKernelManager.h"
#include "itkCudaUtil.h"

// Helper to convert OpenCL global worker size to Cuda block size
#define GETBLOCKSIZE(_g_, _l_) ((_g_ + _l_ - 1) / _l_) // ceil(_g_ / (float)_l_)

namespace itk
{
CudaKernelManager::CudaKernelManager()
{
  m_Program = 0;
  m_Manager = CudaContextManager::GetInstance();
}

CudaKernelManager::~CudaKernelManager()
{
  if (m_Program)
    {
    CUDA_CHECK(cuModuleUnload(m_Program));
    m_Program = 0;
    }
  m_KernelContainer.clear();
}

bool CudaKernelManager::LoadProgramFromFile(const char* filename)
{
  //
  // Create CUDA module from cubin or PTX file
  //
  CUmodule module;
  CUresult err = cuModuleLoad(&module, filename);
  if (err != CUDA_SUCCESS) 
    {
    itkWarningMacro(<< "Cannot create Cuda program");
    return false;
    }

  m_Program = module;
  
  return true;
}

bool CudaKernelManager::LoadProgramFromString(const char* str)
{
  //
  // Create CUDA module from cubin or PTX file
  //
  CUmodule module;
  try
    {
    CUDA_CHECK(cuModuleLoadData(&module, str));
    }
  catch(::itk::ExceptionObject e)
    {
    itkWarningMacro(<< e.GetDescription());
    }

  m_Program = module;
  
  return true;
}

int CudaKernelManager::CreateKernel(const char* kernelName)
{
  CUfunction func;
  CUresult errid = cuModuleGetFunction(&func, static_cast<CUmodule>(m_Program), kernelName);
  if (errid != CUDA_SUCCESS) 
    {
    itkWarningMacro("Fail to create Cuda kernel " + std::string(kernelName));
    return false;
    }
  
  m_KernelContainer.push_back(func);

  // argument list
  m_KernelArgumentReady.push_back(std::vector< KernelArgumentList >());
  
  return (int)m_KernelContainer.size()-1;
}

int CudaKernelManager::CreateKernel(const char* kernelName, const std::type_info& type)
{
  std::stringstream s;
  s << kernelName << "_" << GetTypename(type);
  return CreateKernel(s.str().c_str());
}

bool CudaKernelManager::PushKernelArg(int kernelIdx, const void* argVal)
{
  if (kernelIdx < 0 || kernelIdx >= (int)m_KernelContainer.size()) return false;

  size_t argIdx = m_KernelArgumentReady.size();
  m_KernelArgumentReady[kernelIdx].resize(argIdx+1);
  m_KernelArgumentReady[kernelIdx][argIdx].m_Arg = argVal;
  m_KernelArgumentReady[kernelIdx][argIdx].m_IsReady = true;
  m_KernelArgumentReady[kernelIdx][argIdx].m_CudaDataManager = (CudaDataManager::Pointer)NULL;

  return true;
}

bool CudaKernelManager::SetKernelArg(int kernelIdx, int argIdx, size_t itkNotUsed(argSize), const void* argVal)
{
  if (kernelIdx < 0 || kernelIdx >= (int)m_KernelContainer.size()) return false;

  m_KernelArgumentReady[kernelIdx].resize(argIdx+1);
  m_KernelArgumentReady[kernelIdx][argIdx].m_Arg = argVal;
  m_KernelArgumentReady[kernelIdx][argIdx].m_IsReady = true;
  m_KernelArgumentReady[kernelIdx][argIdx].m_CudaDataManager = (CudaDataManager::Pointer)NULL;

  return true;
}

bool CudaKernelManager::SetKernelArgWithImage(int kernelIdx, int argIdx, CudaDataManager::Pointer manager)
{
  if (kernelIdx < 0 || kernelIdx >= (int)m_KernelContainer.size()) return false;

  m_KernelArgumentReady[kernelIdx].resize(argIdx+1);
  m_KernelArgumentReady[kernelIdx][argIdx].m_Arg = manager->GetGPUBufferPointer();
  m_KernelArgumentReady[kernelIdx][argIdx].m_IsReady = true;
  m_KernelArgumentReady[kernelIdx][argIdx].m_CudaDataManager = manager;

  return true;
}

// this function must be called right before Cuda kernel is launched
bool CudaKernelManager::CheckArgumentReady(int kernelIdx)
{
  if (kernelIdx < 0 || kernelIdx >= (int)m_KernelContainer.size()) return false;

  int nArg = m_KernelArgumentReady[kernelIdx].size();

  for (int i = 0; i < nArg; i++)
    {
    if (!(m_KernelArgumentReady[kernelIdx][i].m_IsReady)) return false;

    // automatic synchronization before kernel launch
    if (m_KernelArgumentReady[kernelIdx][i].m_CudaDataManager != (CudaDataManager::Pointer)NULL)
      {
      m_KernelArgumentReady[kernelIdx][i].m_CudaDataManager->SetCPUBufferDirty();
      }
    }
  return true;
}

void CudaKernelManager::ClearKernelArgs(int kernelIdx)
{
  if (kernelIdx < 0 || kernelIdx >= (int)m_KernelContainer.size()) return;
  m_KernelArgumentReady[kernelIdx] = std::vector< KernelArgumentList >();
}

void CudaKernelManager::ResetArguments(int kernelIdx)
{
  if (kernelIdx < 0 || kernelIdx >= (int)m_KernelContainer.size()) return;

  int nArg = m_KernelArgumentReady[kernelIdx].size();
  for (int i = 0; i < nArg; i++)
    {
    m_KernelArgumentReady[kernelIdx][i].m_IsReady = false;
    m_KernelArgumentReady[kernelIdx][i].m_CudaDataManager = (CudaDataManager::Pointer)NULL;
    }
}

bool CudaKernelManager::GetKernelParams(int kernelIdx, std::vector<void*>& params)
{
  if (kernelIdx < 0 || kernelIdx >= (int)m_KernelContainer.size()) return false;

  int nArg = m_KernelArgumentReady[kernelIdx].size();
  params.resize(nArg);
  for (int i = 0; i < nArg; i++)
    {
    params[i] = const_cast<void*>(m_KernelArgumentReady[kernelIdx][i].m_Arg);
    }
  return true;
}

bool CudaKernelManager::LaunchKernel1D(int kernelIdx, size_t globalWorkSize, size_t localWorkSize,
                                       unsigned int sharedMemBytes)
{
  size_t globalWorkSizet[3] = { globalWorkSize, 1, 1 };
  size_t localWorkSizet[3] = { localWorkSize, 1, 1 };
  return LaunchKernel(kernelIdx, 1, globalWorkSizet, localWorkSizet, sharedMemBytes);
}

bool CudaKernelManager::LaunchKernel2D(int kernelIdx,
                                      size_t globalWorkSizeX, size_t globalWorkSizeY,
                                      size_t localWorkSizeX,  size_t localWorkSizeY,
                                      unsigned int sharedMemBytes)
{
  size_t globalWorkSize[3] = { globalWorkSizeX, globalWorkSizeY, 1 };
  size_t localWorkSize[3] = { localWorkSizeX, localWorkSizeY, 1 };
  return LaunchKernel(kernelIdx, 2, globalWorkSize, localWorkSize, sharedMemBytes);
}

bool CudaKernelManager::LaunchKernel3D(int kernelIdx,
                                      size_t globalWorkSizeX, size_t globalWorkSizeY, size_t globalWorkSizeZ,
                                      size_t localWorkSizeX,  size_t localWorkSizeY, size_t localWorkSizeZ,
                                      unsigned int sharedMemBytes)
{
  size_t globalWorkSize[3] = { globalWorkSizeX, globalWorkSizeY, globalWorkSizeZ };
  size_t localWorkSize[3] = { localWorkSizeX, localWorkSizeY, localWorkSizeZ };
  return LaunchKernel(kernelIdx, 3, globalWorkSize, localWorkSize, sharedMemBytes);
}

bool CudaKernelManager::LaunchKernel(int kernelIdx, int itkNotUsed(dim), size_t *globalWorkSize, size_t *localWorkSize,
                                     unsigned int sharedMemBytes)
{
  if (kernelIdx < 0 || kernelIdx >= (int)m_KernelContainer.size()) return false;

  if (!CheckArgumentReady(kernelIdx))
    {
    itkWarningMacro(<< "Cuda kernel arguments are not completely assigned");
    return false;
    }

  std::vector<void*> params;
  GetKernelParams(kernelIdx, params);
  
  std::cout << "Kernel " << m_KernelContainer[kernelIdx] << " has " << params.size() << " params" << std::endl;
  /*for (int i = 0; i < params.size() ;i++)
    {
    std::cout << " param " << i << " : " << (void*)*(unsigned int*)params[i] << std::endl;
    }*/

  std::cout << " globalWorkSize: " << globalWorkSize[0] << ", " << globalWorkSize[1] << ", " << globalWorkSize[2] << std::endl;
  std::cout << " localWorkSize: " << localWorkSize[0] << ", " << localWorkSize[1] << ", " << localWorkSize[2] << std::endl;

  std::cout << "blocks " << GETBLOCKSIZE(globalWorkSize[0], localWorkSize[0]) << ", " <<
    GETBLOCKSIZE(globalWorkSize[1], localWorkSize[1]) << ", " << 
    GETBLOCKSIZE(globalWorkSize[2], localWorkSize[2]) << std::endl;
  
  CUDA_CHECK(cuLaunchKernel(m_KernelContainer[kernelIdx], 
    GETBLOCKSIZE(globalWorkSize[0], localWorkSize[0]), 
    GETBLOCKSIZE(globalWorkSize[1], localWorkSize[1]),
    GETBLOCKSIZE(globalWorkSize[2], localWorkSize[2]),
    localWorkSize[0], localWorkSize[1], localWorkSize[2], 
    sharedMemBytes, 0, &params[0], 0));
  return true;
}

void CudaKernelManager::Synchronize()
{
  CUDA_CHECK(cuCtxSynchronize());
}

}
