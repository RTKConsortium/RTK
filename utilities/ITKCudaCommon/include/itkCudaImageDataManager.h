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
#ifndef itkCudaImageDataManager_h
#define itkCudaImageDataManager_h

#include <itkObject.h>
#include <itkLightObject.h>
#include <itkObjectFactory.h>
#include "itkCudaUtil.h"
#include "itkCudaDataManager.h"
#include "itkCudaContextManager.h"

namespace itk
{
/**
 * \class CudaImageDataManager
 *
 * DataManager for CudaImage. This class will take care of data synchronization
 * between CPU Image and Cuda Image.
 *
 * \ingroup ITKCudaCommon
 */
template <class ImageType>
class ITK_EXPORT CudaImageDataManager : public CudaDataManager
{
  // allow CudaKernelManager to access Cuda buffer pointer
  friend class CudaKernelManager;

public:
  using Self = CudaImageDataManager;
  using Superclass = CudaDataManager;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  using RegionType = typename ImageType::RegionType;
  using IndexType = typename ImageType::IndexType;
  using SizeType = typename ImageType::SizeType;

  itkNewMacro(Self);
  itkTypeMacro(CudaImageDataManager, CudaDataManager);

  itkGetModifiableObjectMacro(GPUBufferedRegionIndex, CudaDataManager);
  itkGetModifiableObjectMacro(GPUBufferedRegionSize, CudaDataManager);

  void
  SetImagePointer(ImageType * img);
  ImageType *
  GetImagePointer()
  {
    return this->m_Image;
  }

  /** actual Cuda->CPU memory copy takes place here */
  virtual void
  MakeCPUBufferUpToDate();

  /** actual CPU->Cuda memory copy takes place here */
  virtual void
  MakeGPUBufferUpToDate();

  /** Grafting Cuda Image Data */
  virtual void
  Graft(const CudaDataManager * data);

protected:
  CudaImageDataManager() {}
  virtual ~CudaImageDataManager() {}

  virtual void
  PrintSelf(std::ostream & os, Indent indent) const;

private:
  CudaImageDataManager(const Self &); // purposely not implemented
  void
  operator=(const Self &);

  ImageType *                       m_Image;
  IndexType                         m_BufferedRegionIndex;
  SizeType                          m_BufferedRegionSize;
  typename CudaDataManager::Pointer m_GPUBufferedRegionIndex;
  typename CudaDataManager::Pointer m_GPUBufferedRegionSize;
};

} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkCudaImageDataManager.hxx"
#endif

#endif
