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

#ifndef __itkMultiplyImageTestFilter_h
#define __itkMultiplyImageTestFilter_h

#include "itkConceptChecking.h"
#include "itkBarrier.h"
#include <vector>
#include "itkImageToImageFilter.h"

namespace itk {

template <class TInputImage, class TOutputImage>
class ITK_EXPORT MultiplyImageTestFilter:
 public itk::InPlaceImageFilter<TInputImage,TOutputImage>
{
public:

  /** Standard class typedefs. */
  typedef MultiplyImageTestFilter                           Self;
  typedef itk::InPlaceImageFilter<TInputImage,TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                           Pointer;
  typedef itk::SmartPointer<const Self>                     ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(MultiplyImageTestFilter, InPlaceImageFilter);

protected:
  MultiplyImageTestFilter();
  virtual ~MultiplyImageTestFilter() {};

  virtual void BeforeThreadedGenerateData();

  /** Generates a FOV mask which is applied to the reconstruction
   * A call to this function will assume modification of the function.*/
  virtual void ThreadedGenerateData( const OutputImageRegionType& outputRegionForThread, ThreadIdType threadId );

private:
  MultiplyImageTestFilter(const Self&);      //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkMultiplyImageTestFilter.txx"
#endif

#endif
