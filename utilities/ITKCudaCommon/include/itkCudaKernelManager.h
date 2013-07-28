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
#ifndef __itkCudaKernelManager_h
#define __itkCudaKernelManager_h

#include <vector>
#include <itkLightObject.h>
#include <itkObjectFactory.h>
#include "itkCudaUtil.h"
#include "itkCudaImage.h"
#include "itkCudaContextManager.h"
#include "itkCudaDataManager.h"

namespace itk
{
/** \class CudaKernelManager
 * \brief Cuda kernel manager implemented using Cuda.
 *
 * This class is responsible for managing the Cuda kernel 
 *
 * \ingroup ITKCudaCommon
 */

class ITK_EXPORT CudaKernelManager : public LightObject
{
public:

  struct KernelArgumentList
    {
    bool m_IsReady;
    CudaDataManager::Pointer m_CudaDataManager;
    const void *m_Arg;
    };

  typedef CudaKernelManager        Self;
  typedef LightObject              Superclass;
  typedef SmartPointer<Self>       Pointer;
  typedef SmartPointer<const Self> ConstPointer;

  itkNewMacro(Self);
  itkTypeMacro(CudaKernelManager, LightObject);

  bool LoadProgramFromFile(const char* filename);
  bool LoadProgramFromString(const char* str);
  
  int CreateKernel(const char* kernelName);

  int CreateKernel(const char* kernelName, const std::type_info&);
  
  bool PushKernelArg(int kernelIdx, const void* argVal);
  
  void ClearKernelArgs(int kernelIdx);

  bool SetKernelArg(int kernelIdx, int argIdx, size_t argSize, const void* argVal);

  bool SetKernelArgWithImage(int kernelIdx, int argIdx, CudaDataManager::Pointer manager);

  template <class TPixel, unsigned int VImageDimension>
  bool SetKernelArgWithImage(int kernelIdx, int argIdx, SmartPointer<CudaImage<TPixel, VImageDimension> > image)
  {
    return SetKernelArgWithImage(kernelIdx, argIdx, image->GetCudaDataManager());
  }
  
  /** Pass to Cuda both the pixel buffer and the buffered region. */
  template< class TCudaImageDataManager >
  bool SetKernelArgWithImageAndBufferedRegion(
    int kernelIdx, int &argIdx,
    TCudaImageDataManager *manager)
  {
    return SetKernelArgWithImage(kernelIdx, argIdx++, manager) && 
      SetKernelArgWithImage(kernelIdx, argIdx++, manager->GetGPUBufferedRegionIndex()) &&
      SetKernelArgWithImage(kernelIdx, argIdx++, manager->GetGPUBufferedRegionSize());
  }

  bool LaunchKernel(int kernelIdx, int dim, 
                   size_t *globalWorkSize, size_t *localWorkSize, 
                   unsigned int sharedMemBytes = 0);

  bool LaunchKernel1D(int kernelIdx, 
                      size_t globalWorkSize, size_t localWorkSize, 
                      unsigned int sharedMemBytes = 0);

  bool LaunchKernel2D(int kernelIdx,
                      size_t globalWorkSizeX, size_t globalWorkSizeY,
                      size_t localWorkSizeX,  size_t localWorkSizeY, 
                      unsigned int sharedMemBytes = 0);

  bool LaunchKernel3D(int kernelIdx,
                      size_t globalWorkSizeX, size_t globalWorkSizeY, size_t globalWorkSizeZ,
                      size_t localWorkSizeX,  size_t localWorkSizeY, size_t localWorkSizeZ,
                      unsigned int sharedMemBytes = 0);
  
  void Synchronize();

protected:
  CudaKernelManager();
  virtual ~CudaKernelManager();

  bool CheckArgumentReady(int kernelIdx);

  void ResetArguments(int kernelIdx);

  bool GetKernelParams(int kernelIdx, std::vector<void*>& params);

private:
  CudaKernelManager(const Self&);   //purposely not implemented
  void operator=(const Self&);

  CUmodule m_Program;

  CudaContextManager * m_Manager;

  std::vector< CUfunction >                        m_KernelContainer;
  std::vector< std::vector< KernelArgumentList > > m_KernelArgumentReady;
};
}

#endif
