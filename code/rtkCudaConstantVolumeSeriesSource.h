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

#ifndef rtkCudaConstantVolumeSeriesSource_h
#define rtkCudaConstantVolumeSeriesSource_h

#include "rtkConstantImageSource.h"
#include <itkCudaImageToImageFilter.h>
#include "rtkWin32Header.h"

namespace rtk
{

/** \class CudaConstantVolumeSeriesSource
 * \brief A 4D constant image source on GPU.
 *
 *
 *
 * \author Cyril Mory
 *
 * \ingroup CudaImageToImageFilter
 */

  class ITK_EXPORT CudaConstantVolumeSeriesSource :
        public itk::CudaImageToImageFilter< itk::CudaImage<float,4>, itk::CudaImage<float,4>,
    ConstantImageSource< itk::CudaImage<float,4> > >
{
public:
  /** Standard class typedefs. */
  typedef rtk::CudaConstantVolumeSeriesSource                               Self;
  typedef itk::CudaImage<float,3>                                     OutputImageType;
  typedef rtk::ConstantImageSource< OutputImageType >                 Superclass;
  typedef itk::SmartPointer<Self>                                     Pointer;
  typedef itk::SmartPointer<const Self>                               ConstPointer;

  /** Standard New method. */
  itkNewMacro(Self)

  /** Runtime information support. */
  itkTypeMacro(CudaConstantVolumeSeriesSource, ImageToImageFilter)

protected:
  rtkcuda_EXPORT CudaConstantVolumeSeriesSource();
  ~CudaConstantVolumeSeriesSource(){
  }

  virtual void GPUGenerateData();

private:
  CudaConstantVolumeSeriesSource(const Self&); //purposely not implemented
  void operator=(const Self&);         //purposely not implemented

}; // end of class

} // end namespace rtk

#endif
