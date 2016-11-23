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

#ifndef __InPlaceImageTestFilter_h
#define __InPlaceImageTestFilter_h

#include "itkCudaInPlaceImageFilter.h"
#include "itkConceptChecking.h"
#include "itkBarrier.h"
#include <vector>
#include "itkMultiplyImageTestFilter.h"

namespace itk {

template <class TInputImage, class TOutputImage>
class ITK_EXPORT InPlaceImageTestFilter:
  public CudaInPlaceImageFilter<TInputImage,TOutputImage,MultiplyImageTestFilter<TInputImage,TOutputImage>>
{
public:

  /** Standard class typedefs. */
  typedef InPlaceImageTestFilter                            Self;
  typedef itk::CudaInPlaceImageFilter<TInputImage,TOutputImage,MultiplyImageTestFilter<TInputImage,TOutputImage>> Superclass;
  typedef itk::SmartPointer<Self>                           Pointer;
  typedef itk::SmartPointer<const Self>                     ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(InPlaceImageTestFilter, CudaInPlaceImageFilter);

protected:
  InPlaceImageTestFilter();
  virtual ~InPlaceImageTestFilter() {};

  virtual void GPUGenerateData();

private:
  InPlaceImageTestFilter(const Self&);      //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkInPlaceImageTestFilter.txx"
#endif

#endif
