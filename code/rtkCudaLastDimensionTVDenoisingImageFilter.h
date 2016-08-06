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

#ifndef rtkCudaLastDimensionTVDenoisingImageFilter_h
#define rtkCudaLastDimensionTVDenoisingImageFilter_h

#include "rtkTotalVariationDenoisingBPDQImageFilter.h"
#include <itkCudaInPlaceImageFilter.h>
#include "rtkWin32Header.h"

namespace rtk
{

/** \class CudaLastDimensionTVDenoisingImageFilter
 * \brief Implements the TotalVariationDenoisingBPDQImageFilter on GPU
 * for a specific case : denoising only along the last dimension
 *
 *
 *
 * \author Cyril Mory
 *
 * \ingroup CudaImageToImageFilter
 */

  class ITK_EXPORT CudaLastDimensionTVDenoisingImageFilter :
        public itk::CudaInPlaceImageFilter< itk::CudaImage<float,4>, itk::CudaImage<float,4>,
    TotalVariationDenoisingBPDQImageFilter< itk::CudaImage<float,4>, itk::CudaImage< itk::CovariantVector < float, 1 >, 4 > > >
{
public:
  /** Standard class typedefs. */
  typedef rtk::CudaLastDimensionTVDenoisingImageFilter                               Self;
  typedef itk::CudaImage<float,4>                                                       OutputImageType;
  typedef itk::CudaImage< itk::CovariantVector < float, 1 >, 4 >                        GradientType;
  typedef rtk::TotalVariationDenoisingBPDQImageFilter< OutputImageType, GradientType >  Superclass;
  typedef itk::SmartPointer<Self>                                                       Pointer;
  typedef itk::SmartPointer<const Self>                                                 ConstPointer;

  /** Standard New method. */
  itkNewMacro(Self)

  /** Runtime information support. */
  itkTypeMacro(CudaLastDimensionTVDenoisingImageFilter, TotalVariationDenoisingBPDQImageFilter)

protected:
  rtkcuda_EXPORT CudaLastDimensionTVDenoisingImageFilter();
  ~CudaLastDimensionTVDenoisingImageFilter(){}

  virtual void GPUGenerateData();

private:
  CudaLastDimensionTVDenoisingImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);         //purposely not implemented

}; // end of class

} // end namespace rtk

#endif
