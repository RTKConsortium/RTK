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
#ifndef __itkCudaImageDataManager_hxx
#define __itkCudaImageDataManager_hxx

#include "itkCudaImageDataManager.h"
#include "itkCudaUtil.h"
//#define VERBOSE

namespace itk
{

template < class ImageType >
void CudaImageDataManager< ImageType >::SetImagePointer(ImageType* img)
{
  m_Image = img;

  RegionType region = m_Image->GetBufferedRegion();
  IndexType  index  = region.GetIndex();
  SizeType   size   = region.GetSize();

  for (unsigned int d = 0; d < ImageType::ImageDimension; d++)
    {
    m_BufferedRegionIndex[d] = index[d];
    m_BufferedRegionSize[d] = size[d];
    }

  m_GPUBufferedRegionIndex = CudaDataManager::New();
  m_GPUBufferedRegionIndex->SetBufferSize(sizeof(int) * ImageType::ImageDimension);
  m_GPUBufferedRegionIndex->SetCPUBufferPointer(&m_BufferedRegionIndex);
  m_GPUBufferedRegionIndex->SetGPUBufferDirty();

  m_GPUBufferedRegionSize = CudaDataManager::New();
  m_GPUBufferedRegionSize->SetBufferSize(sizeof(int) * ImageType::ImageDimension);
  m_GPUBufferedRegionSize->SetCPUBufferPointer(&m_BufferedRegionSize);
  m_GPUBufferedRegionSize->SetGPUBufferDirty();
}

template < class ImageType >
void CudaImageDataManager< ImageType >::MakeCPUBufferUpToDate()
{
  if (m_Image)
    {
    m_Mutex.Lock();

    TimeStamp gpu_time_stamp = this->GetTimeStamp();
    TimeStamp cpu_time_stamp = m_Image->GetTimeStamp();

    /* Why we check dirty flag and time stamp together?
     * Because existing CPU image filters do not use pixel/buffer
     * access function in CudaImage and therefore dirty flag is not
     * correctly managed. Therefore, we check the time stamp of
     * CPU and Cuda data as well
     */
    if ((m_IsCPUBufferDirty || (gpu_time_stamp > cpu_time_stamp)) && m_GPUBuffer.GetPointer() != NULL && m_CPUBuffer != NULL)
      {
      cudaError_t errid;
#ifdef VERBOSE
      std::cout << this << ": GPU->CPU data copy" << std::endl;
#endif

      CUDA_CHECK(cuCtxSetCurrent(*(this->m_ContextManager->GetCurrentContext()))); // This is necessary when running multithread to bind the host CPU thread to the right context
      errid = cudaMemcpy(m_CPUBuffer, m_GPUBuffer->GetPointer(), m_BufferSize, cudaMemcpyDeviceToHost);
      CudaCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);

      m_Image->Modified();

      m_IsCPUBufferDirty = false;
      m_IsGPUBufferDirty = false;
      }

    m_Mutex.Unlock();
    }
}

template < class ImageType >
void CudaImageDataManager< ImageType >::MakeGPUBufferUpToDate()
{
  if (m_Image)
    {
    m_Mutex.Lock();

    TimeStamp gpu_time_stamp = this->GetTimeStamp();
    TimeStamp cpu_time_stamp = m_Image->GetTimeStamp();
    
    /* Why we check dirty flag and time stamp together?
    * Because existing CPU image filters do not use pixel/buffer
    * access function in CudaImage and therefore dirty flag is not
    * correctly managed. Therefore, we check the time stamp of
    * CPU and GPU data as well
    */
    if ((m_IsGPUBufferDirty || (gpu_time_stamp < cpu_time_stamp)) && m_CPUBuffer != NULL && m_GPUBuffer.GetPointer() != NULL)
      {
      cudaError_t errid;
#ifdef VERBOSE
      std::cout << "CPU->GPU data copy" << std::endl;
#endif

      CUDA_CHECK(cuCtxSetCurrent(*(this->m_ContextManager->GetCurrentContext()))); // This is necessary when running multithread to bind the host CPU thread to the right context
      errid = cudaMemcpy(m_GPUBuffer->GetPointer(), m_CPUBuffer, m_BufferSize, cudaMemcpyHostToDevice);
      CudaCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);

      this->SetTimeStamp(cpu_time_stamp);

      m_IsCPUBufferDirty = false;
      m_IsGPUBufferDirty = false;
      }

    m_Mutex.Unlock();
    }
}

template < class ImageType >
void CudaImageDataManager< ImageType >::Graft(const CudaDataManager* data)
{
  Superclass::Graft(data);
}

template < class ImageType >
void CudaImageDataManager< ImageType >::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os,indent);

  os << indent << "m_GPUBufferedRegionIndex: " << m_GPUBufferedRegionIndex << std::endl;
  os << indent << "m_GPUBufferedRegionSize: " << m_GPUBufferedRegionSize << std::endl;
}

} // namespace itk

#endif
