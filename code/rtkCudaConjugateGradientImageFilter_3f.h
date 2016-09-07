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

#ifndef rtkCudaConjugateGradientImageFilter_3f_h
#define rtkCudaConjugateGradientImageFilter_3f_h

#include "rtkConjugateGradientImageFilter.h"
#include <itkCudaImageToImageFilter.h>
#include "rtkWin32Header.h"

namespace rtk
{

/** \class CudaConjugateGradientImageFilter_3f
 * \brief A 3D float conjugate gradient image filter on GPU.
 *
 *
 *
 * \author Cyril Mory
 *
 * \ingroup CudaImageToImageFilter
 */

  class ITK_EXPORT CudaConjugateGradientImageFilter_3f :
        public itk::CudaImageToImageFilter< itk::CudaImage<float,3>, itk::CudaImage<float,3>,
    ConjugateGradientImageFilter< itk::CudaImage<float,3> > >
{
public:
  /** Standard class typedefs. */
  typedef rtk::CudaConjugateGradientImageFilter_3f                    Self;
  typedef itk::CudaImage<float,3>                                     OutputImageType;
  typedef rtk::ConjugateGradientImageFilter< OutputImageType >        Superclass;
  typedef itk::SmartPointer<Self>                                     Pointer;
  typedef itk::SmartPointer<const Self>                               ConstPointer;

  /** Standard New method. */
  itkNewMacro(Self)

  /** Runtime information support. */
  itkTypeMacro(CudaConjugateGradientImageFilter_3f, ConjugateGradientImageFilter)

protected:
  rtkcuda_EXPORT CudaConjugateGradientImageFilter_3f();
  ~CudaConjugateGradientImageFilter_3f(){
  }

  virtual void GPUGenerateData();

private:
  CudaConjugateGradientImageFilter_3f(const Self&); //purposely not implemented
  void operator=(const Self&);         //purposely not implemented

}; // end of class

} // end namespace rtk

#endif
