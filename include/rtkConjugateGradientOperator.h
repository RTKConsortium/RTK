/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
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
/** \class ConjugateGradientOperator
 *
 * \author Cyril Mory
 *
 * \ingroup RTK
 */
template <typename OutputImageType>
class ITK_TEMPLATE_EXPORT ConjugateGradientOperator : public itk::ImageToImageFilter<OutputImageType, OutputImageType>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ConjugateGradientOperator);

  /** Standard class type alias. */
  using Self = ConjugateGradientOperator;
  using Superclass = itk::ImageToImageFilter<OutputImageType, OutputImageType>;
  using Pointer = itk::SmartPointer<Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(ConjugateGradientOperator);

  /** The image to be updated.*/
  virtual void
  SetX(const OutputImageType * OutputImage);

protected:
  ConjugateGradientOperator();
  ~ConjugateGradientOperator() override = default;
};
} // namespace rtk


#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkConjugateGradientOperator.hxx"
#endif

#endif
