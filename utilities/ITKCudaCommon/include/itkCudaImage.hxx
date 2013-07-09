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
#ifndef __itkCudaImage_hxx
#define __itkCudaImage_hxx

#include "itkCudaImage.h"

namespace itk
{
//
// Constructor
//
template <class TPixel, unsigned int VImageDimension>
CudaImage< TPixel, VImageDimension >::CudaImage()
{
  m_DataManager = CudaImageDataManager< CudaImage< TPixel, VImageDimension > >::New();
  m_DataManager->SetTimeStamp(this->GetTimeStamp());
}

template <class TPixel, unsigned int VImageDimension>
CudaImage< TPixel, VImageDimension >::~CudaImage()
{
}

template <class TPixel, unsigned int VImageDimension>
void CudaImage< TPixel, VImageDimension >::Allocate()
{
  // allocate CPU memory - calling Allocate() in superclass
  Superclass::Allocate();

  // allocate Cuda memory
  this->ComputeOffsetTable();
  unsigned long numPixel = this->GetOffsetTable()[VImageDimension];
  m_DataManager->SetBufferSize(sizeof(TPixel)*numPixel);
  m_DataManager->SetImagePointer(this);
  m_DataManager->SetCPUBufferPointer(Superclass::GetBufferPointer());
  m_DataManager->Allocate();

  /* prevent unnecessary copy from CPU to Cuda at the beginning */
  m_DataManager->SetTimeStamp(this->GetTimeStamp());
}

template< class TPixel, unsigned int VImageDimension >
void CudaImage< TPixel, VImageDimension >::Initialize()
{
  // CPU image initialize
  Superclass::Initialize();

  // Cuda image initialize
  m_DataManager->Initialize();
  this->ComputeOffsetTable();
  unsigned long numPixel = this->GetOffsetTable()[VImageDimension];
  m_DataManager->SetBufferSize(sizeof(TPixel)*numPixel);
  m_DataManager->SetImagePointer(this);
  m_DataManager->SetCPUBufferPointer(Superclass::GetBufferPointer());
  m_DataManager->Allocate();

  /* prevent unnecessary copy from CPU to Cuda at the beginning */
  m_DataManager->SetTimeStamp(this->GetTimeStamp());
}

/** If the class is modified we marke the GPUBuffer has dirty */
template <class TPixel, unsigned int VImageDimension>
void CudaImage< TPixel, VImageDimension >::Modified() const
{
  Superclass::Modified();
  m_DataManager->SetCPUDirtyFlag(false); // prevent the GPU to copy to the CPU
  m_DataManager->SetGPUBufferDirty();
}

template <class TPixel, unsigned int VImageDimension>
void CudaImage< TPixel, VImageDimension >::FillBuffer(const TPixel & value)
{
  m_DataManager->SetGPUBufferDirty();
  Superclass::FillBuffer(value);
}

template <class TPixel, unsigned int VImageDimension>
void CudaImage< TPixel, VImageDimension >::SetPixel(const IndexType & index, const TPixel & value)
{
  m_DataManager->SetGPUBufferDirty();
  Superclass::SetPixel(index, value);
}

template <class TPixel, unsigned int VImageDimension>
const TPixel & CudaImage< TPixel, VImageDimension >::GetPixel(const IndexType & index) const
{
  m_DataManager->UpdateCPUBuffer();
  return Superclass::GetPixel(index);
}

template <class TPixel, unsigned int VImageDimension>
TPixel & CudaImage< TPixel, VImageDimension >::GetPixel(const IndexType & index)
{
  m_DataManager->UpdateCPUBuffer();
  return Superclass::GetPixel(index);
}

template <class TPixel, unsigned int VImageDimension>
TPixel & CudaImage< TPixel, VImageDimension >::operator[] (const IndexType &index)
{
  m_DataManager->UpdateCPUBuffer();
  return Superclass::operator[] (index);
}

template <class TPixel, unsigned int VImageDimension>
const TPixel & CudaImage< TPixel, VImageDimension >::operator[] (const IndexType &index) const
{
  m_DataManager->UpdateCPUBuffer();
  return Superclass::operator[] (index);
}

template <class TPixel, unsigned int VImageDimension>
void CudaImage< TPixel, VImageDimension >::SetPixelContainer(PixelContainer *container)
{
  Superclass::SetPixelContainer(container);
  m_DataManager->SetCPUDirtyFlag(false);
  m_DataManager->SetCudaDirtyFlag(true);
}

template <class TPixel, unsigned int VImageDimension>
void CudaImage< TPixel, VImageDimension >::UpdateBuffers()
{
  m_DataManager->UpdateCPUBuffer();
  m_DataManager->UpdateCudaBuffer();
}

template <class TPixel, unsigned int VImageDimension>
TPixel* CudaImage< TPixel, VImageDimension >::GetBufferPointer()
{
  /* less conservative version - if you modify pixel value using
   * this pointer then you must set the image as modified manually!!! */
  m_DataManager->UpdateCPUBuffer();
  return Superclass::GetBufferPointer();
}

template <class TPixel, unsigned int VImageDimension>
const TPixel * CudaImage< TPixel, VImageDimension >::GetBufferPointer() const
{
  // const does not change buffer, but if CPU is dirty then make it up-to-date
  m_DataManager->UpdateCPUBuffer();
  return Superclass::GetBufferPointer();
}

template <class TPixel, unsigned int VImageDimension>
CudaDataManager::Pointer
CudaImage< TPixel, VImageDimension >::GetCudaDataManager() const
{
  typedef typename CudaImageDataManager< CudaImage >::Superclass CudaImageDataSuperclass;
  typedef typename CudaImageDataSuperclass::Pointer             CudaImageDataSuperclassPointer;

  return static_cast< CudaImageDataSuperclassPointer >(m_DataManager.GetPointer());
}

template <class TPixel, unsigned int VImageDimension>
void
CudaImage< TPixel, VImageDimension >::Graft(const DataObject *data)
{
  typedef CudaImageDataManager< CudaImage >             CudaImageDataManagerType;
  typedef typename CudaImageDataManagerType::Superclass CudaImageDataSuperclass;
  typedef typename CudaImageDataSuperclass::Pointer     CudaImageDataSuperclassPointer;

  // call the superclass' implementation
  Superclass::Graft(data);

  // Pass regular pointer to Graft() instead of smart pointer due to type
  // casting problem
  CudaImageDataManagerType* ptr = dynamic_cast<CudaImageDataManagerType*>(
      (((CudaImage*)data)->GetCudaDataManager()).GetPointer());

  // call Cuda data graft function
  //m_DataManager->SetImagePointer(this); // hu! not necessary ?!
  m_DataManager->Graft(ptr);

  // Synchronize timestamp of CudaImage and CudaDataManager
  m_DataManager->SetTimeStamp(this->GetTimeStamp());
}

} // namespace itk

#endif
