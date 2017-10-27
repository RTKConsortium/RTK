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

#ifndef rtkCudaTotalVariationDenoisingBPDQImageFilter_h
#define rtkCudaTotalVariationDenoisingBPDQImageFilter_h

#include "rtkTotalVariationDenoisingBPDQImageFilter.h"
#include <itkCudaImageToImageFilter.h>
#include "rtkWin32Header.h"

namespace rtk
{

/** \class CudaTotalVariationDenoisingBPDQImageFilter
 * \brief Implements the TotalVariationDenoisingBPDQImageFilter on GPU.
 *
 *
 *
 * \author Cyril Mory
 *
 * \ingroup CudaImageToImageFilter
 */

  class ITK_EXPORT CudaTotalVariationDenoisingBPDQImageFilter :
        public itk::CudaImageToImageFilter< itk::CudaImage<float,3>, itk::CudaImage<float,3>,
    TotalVariationDenoisingBPDQImageFilter< itk::CudaImage<float,3>, itk::CudaImage< itk::CovariantVector < float, 3 >, 3 > > >
{
public:
  /** Standard class typedefs. */
  typedef rtk::CudaTotalVariationDenoisingBPDQImageFilter                               Self;
  typedef itk::CudaImage<float,3>                                                       OutputImageType;
  typedef itk::CudaImage< itk::CovariantVector < float, 3 >, 3 >                        GradientType;
  typedef rtk::TotalVariationDenoisingBPDQImageFilter< OutputImageType, GradientType >  Superclass;
  typedef itk::SmartPointer<Self>                                                       Pointer;
  typedef itk::SmartPointer<const Self>                                                 ConstPointer;

  /** Standard New method. */
  itkNewMacro(Self)

  /** Runtime information support. */
  itkTypeMacro(CudaTotalVariationDenoisingBPDQImageFilter, TotalVariationDenoisingBPDQImageFilter)

protected:
  rtkcuda_EXPORT CudaTotalVariationDenoisingBPDQImageFilter();
  ~CudaTotalVariationDenoisingBPDQImageFilter(){
  }

  virtual void GPUGenerateData();

private:
  CudaTotalVariationDenoisingBPDQImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);         //purposely not implemented

}; // end of class

} // end namespace rtk

#endif
