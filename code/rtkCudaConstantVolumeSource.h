/*=========================================================================
 *
 *  Copyright RTK Consortium
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

#ifndef rtkCudaConstantVolumeSource_h
#define rtkCudaConstantVolumeSource_h

#include "rtkConstantImageSource.h"
#include <itkCudaImageToImageFilter.h>
#include "rtkWin32Header.h"

namespace rtk
{

/** \class CudaConstantVolumeSource
 * \brief A 3D constant image source on GPU.
 *
 *
 *
 * \author Cyril Mory
 *
 * \ingroup CudaImageToImageFilter
 */

  class ITK_EXPORT CudaConstantVolumeSource :
        public itk::CudaImageToImageFilter< itk::CudaImage<float,3>, itk::CudaImage<float,3>,
    ConstantImageSource< itk::CudaImage<float,3> > >
{
public:
  /** Standard class typedefs. */
  typedef rtk::CudaConstantVolumeSource                               Self;
  typedef itk::CudaImage<float,3>                                     OutputImageType;
  typedef rtk::ConstantImageSource< OutputImageType >                 Superclass;
  typedef itk::SmartPointer<Self>                                     Pointer;
  typedef itk::SmartPointer<const Self>                               ConstPointer;

  /** Standard New method. */
  itkNewMacro(Self)

  /** Runtime information support. */
  itkTypeMacro(CudaConstantVolumeSource, ImageToImageFilter)

protected:
  rtkcuda_EXPORT CudaConstantVolumeSource();
  ~CudaConstantVolumeSource(){
  }

  virtual void GPUGenerateData();

private:
  CudaConstantVolumeSource(const Self&); //purposely not implemented
  void operator=(const Self&);         //purposely not implemented

}; // end of class

} // end namespace rtk

#endif
