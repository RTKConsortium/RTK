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

#ifndef rtkCudaInterpolateImageFilter_h
#define rtkCudaInterpolateImageFilter_h

#include "rtkInterpolatorWithKnownWeightsImageFilter.h"
#include "itkCudaImage.h"
#include "itkCudaInPlaceImageFilter.h"
#include "rtkWin32Header.h"

namespace rtk
{

/** \class CudaInterpolateImageFilter
 * \brief Implements the InterpolateWithKnownWeightsImageFilter on GPU.
 *
 *
 *
 * \author Cyril Mory
 *
 * \ingroup CudaImageToImageFilter
 */
class ITK_EXPORT CudaInterpolateImageFilter :
        public itk::CudaInPlaceImageFilter< itk::CudaImage<float,3>, itk::CudaImage<float,3>,
    InterpolatorWithKnownWeightsImageFilter< itk::CudaImage<float,3>, itk::CudaImage<float,4> > >
{
public:
  /** Standard class typedefs. */
  typedef rtk::CudaInterpolateImageFilter                             Self;
  typedef rtk::InterpolatorWithKnownWeightsImageFilter< OutputImageType, InputImageType > Superclass;
  typedef itk::SmartPointer<Self>                            Pointer;
  typedef itk::SmartPointer<const Self>                      ConstPointer;

  /** Standard New method. */
  itkNewMacro(Self)

  /** Runtime information support. */
  itkTypeMacro(CudaInterpolateImageFilter, InterpolatorWithKnownWeightsImageFilter)

protected:
  rtkcuda_EXPORT CudaInterpolateImageFilter();
  ~CudaInterpolateImageFilter(){
  }

  virtual void GPUGenerateData();

private:
  CudaInterpolateImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);         //purposely not implemented

}; // end of class

} // end namespace rtk

#endif
