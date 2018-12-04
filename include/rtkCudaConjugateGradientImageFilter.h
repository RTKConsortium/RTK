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

#ifndef rtkCudaConjugateGradientImageFilter_h
#define rtkCudaConjugateGradientImageFilter_h

#include "rtkConfiguration.h"
//Conditional definition of the class to pass ITKHeaderTest
#ifdef RTK_USE_CUDA

#include "rtkConjugateGradientImageFilter.h"
#include <itkCudaImageToImageFilter.h>
#include "RTKExport.h"

namespace rtk
{

/** \class CudaConjugateGradientImageFilter
 * \brief A 3D float conjugate gradient image filter on GPU.
 *
 *
 *
 * \author Cyril Mory
 *
 * \ingroup RTK CudaImageToImageFilter
 */

template < class TImage >
class CudaConjugateGradientImageFilter :
  public itk::CudaImageToImageFilter< TImage, TImage,
         ConjugateGradientImageFilter< TImage > >
{
public:
  /** Standard class typedefs. */
  typedef rtk::CudaConjugateGradientImageFilter<TImage>      Self;
  typedef rtk::ConjugateGradientImageFilter< TImage >        Superclass;
  typedef itk::SmartPointer<Self>                            Pointer;
  typedef itk::SmartPointer<const Self>                      ConstPointer;

  /** Standard New method. */
  itkNewMacro(Self)

  /** Runtime information support. */
  itkTypeMacro(CudaConjugateGradientImageFilter, ConjugateGradientImageFilter)

protected:
  CudaConjugateGradientImageFilter();
  ~CudaConjugateGradientImageFilter(){
  }

  virtual void GPUGenerateData();

private:
  CudaConjugateGradientImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);         //purposely not implemented

}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkCudaConjugateGradientImageFilter.hxx"
#endif


#endif //end conditional definition of the class

#endif
