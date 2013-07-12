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

  typedef typename ImageType::RegionType RegionType;
  typedef typename ImageType::IndexType  IndexType;
  typedef typename ImageType::SizeType   SizeType;

  RegionType region = m_Image->GetBufferedRegion();
  IndexType  index  = region.GetIndex();
  SizeType   size   = region.GetSize();

  for (unsigned int d = 0; d < ImageDimension; d++)
    {
    m_BufferedRegionIndex[d] = index[d];
    m_BufferedRegionSize[d] = size[d];
    }

  m_GPUBufferedRegionIndex = CudaDataManager::New();
  m_GPUBufferedRegionIndex->SetBufferSize(sizeof(int) * ImageDimension);
  m_GPUBufferedRegionIndex->SetCPUBufferPointer(m_BufferedRegionIndex);
  m_GPUBufferedRegionIndex->SetGPUBufferDirty();

  m_GPUBufferedRegionSize = CudaDataManager::New();
  m_GPUBufferedRegionSize->SetBufferSize(sizeof(int) * ImageDimension);
  m_GPUBufferedRegionSize->SetCPUBufferPointer(m_BufferedRegionSize);
  m_GPUBufferedRegionSize->SetGPUBufferDirty();
}

template < class ImageType >
void CudaImageDataManager< ImageType >::MakeCPUBufferUpToDate()
{
  if (m_Image)
    {
    m_Mutex.Lock();

    ModifiedTimeType gpu_time = this->GetMTime();
    TimeStamp cpu_time_stamp = m_Image->GetTimeStamp();
    ModifiedTimeType cpu_time = cpu_time_stamp.GetMTime();

    /* Why we check dirty flag and time stamp together?
     * Because existing CPU image filters do not use pixel/buffer
     * access function in CudaImage and therefore dirty flag is not
     * correctly managed. Therefore, we check the time stamp of
     * CPU and Cuda data as well
     */
    if ((m_IsCPUBufferDirty || (gpu_time > cpu_time)) && m_GPUBuffer.GetPointer() != NULL && m_CPUBuffer != NULL)
      {
      cudaError_t errid;
#ifdef VERBOSE
      std::cout << "GPU->CPU data copy" << std::endl;
#endif
      errid = cudaMemcpy(m_CPUBuffer, m_GPUBuffer->GetPointer(), m_BufferSize, cudaMemcpyDeviceToHost);
      CudaCheckError(errid, __FILE__, __LINE__, ITK_LOCATION);

      m_Image->Modified();
      this->SetTimeStamp(m_Image->GetTimeStamp());

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

    ModifiedTimeType gpu_time = this->GetMTime();
    TimeStamp cpu_time_stamp = m_Image->GetTimeStamp();
    ModifiedTimeType cpu_time = m_Image->GetMTime();

    /* Why we check dirty flag and time stamp together?
    * Because existing CPU image filters do not use pixel/buffer
    * access function in CudaImage and therefore dirty flag is not
    * correctly managed. Therefore, we check the time stamp of
    * CPU and GPU data as well
    */
    if ((m_IsGPUBufferDirty || (gpu_time < cpu_time)) && m_CPUBuffer != NULL && m_GPUBuffer.GetPointer() != NULL)
      {
      cudaError_t errid;
#ifdef VERBOSE
      std::cout << "CPU->GPU data copy" << std::endl;
#endif
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

} // namespace itk

#endif
