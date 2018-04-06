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
#ifndef rtkConjugateGradientOperator_h
#define rtkConjugateGradientOperator_h

#include "itkImageToImageFilter.h"

namespace rtk
{
template< typename OutputImageType>
class ConjugateGradientOperator : public itk::ImageToImageFilter< OutputImageType, OutputImageType>
{
public:
 
  /** Standard class typedefs. */
  typedef ConjugateGradientOperator                                   Self;
  typedef itk::ImageToImageFilter< OutputImageType, OutputImageType > Superclass;
  typedef itk::SmartPointer< Self >                                   Pointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self)

  /** Run-time type information (and related methods). */
  itkTypeMacro(ConjugateGradientOperator, itk::ImageToImageFilter)

  /** The 4D image to be updated.*/
  virtual void SetX(const OutputImageType* OutputImage);

protected:
  ConjugateGradientOperator();
  ~ConjugateGradientOperator() {}

private:
  ConjugateGradientOperator(const Self &); //purposely not implemented
  void operator=(const Self &);  //purposely not implemented

};
} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkConjugateGradientOperator.hxx"
#endif

#endif
