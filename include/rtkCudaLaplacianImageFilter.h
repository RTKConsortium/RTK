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

#ifndef rtkCudaLaplacianImageFilter_h
#define rtkCudaLaplacianImageFilter_h

#include "rtkLaplacianImageFilter.h"
#include <itkCudaInPlaceImageFilter.h>
#include "rtkWin32Header.h"

namespace rtk
{

/** \class CudaLaplacianImageFilter
 * \brief Implements the 3D float LaplacianImageFilter on GPU
 *
 * \author Cyril Mory
 *
 * \ingroup CudaImageToImageFilter
 */

  class ITK_EXPORT CudaLaplacianImageFilter :
        public itk::CudaImageToImageFilter< itk::CudaImage<float,3>, itk::CudaImage<float,3>,
    LaplacianImageFilter< itk::CudaImage<float,3>, itk::CudaImage<itk::CovariantVector<float, 3>,3> > >
{
public:
  /** Standard class typedefs. */
  typedef rtk::CudaLaplacianImageFilter                                   Self;
  typedef itk::CudaImage<float,3>                                         OutputImageType;
  typedef itk::CudaImage<itk::CovariantVector<float, 3>,3>                GradientImageType;
  typedef rtk::LaplacianImageFilter< OutputImageType, GradientImageType>  Superclass;
  typedef itk::SmartPointer<Self>                                         Pointer;
  typedef itk::SmartPointer<const Self>                                   ConstPointer;

  /** Standard New method. */
  itkNewMacro(Self)

  /** Runtime information support. */
  itkTypeMacro(CudaLaplacianImageFilter, LaplacianImageFilter)

protected:
  rtkcuda_EXPORT CudaLaplacianImageFilter();
  ~CudaLaplacianImageFilter(){}

  virtual void GPUGenerateData();

private:
  CudaLaplacianImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);         //purposely not implemented

}; // end of class

} // end namespace rtk

#endif
