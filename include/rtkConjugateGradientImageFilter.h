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

#ifndef rtkConjugateGradientImageFilter_h
#define rtkConjugateGradientImageFilter_h

#include <itkSubtractImageFilter.h>
#include <itkStatisticsImageFilter.h>
#include <itkTimeProbe.h>

#include "rtkSumOfSquaresImageFilter.h"

#include "rtkConjugateGradientGetR_kPlusOneImageFilter.h"
#include "rtkConjugateGradientGetX_kPlusOneImageFilter.h"
#include "rtkConjugateGradientGetP_kPlusOneImageFilter.h"
#include "rtkConjugateGradientOperator.h"

namespace rtk
{

/** \class ConjugateGradientImageFilter
 * \brief Solves AX = B by conjugate gradient
 *
 * ConjugateGradientImageFilter implements the algorithm described
 * in https://en.wikipedia.org/wiki/Conjugate_gradient_method
 *
 * \ingroup RTK
 */

template <typename OutputImageType>
class ITK_TEMPLATE_EXPORT ConjugateGradientImageFilter
  : public itk::InPlaceImageFilter<OutputImageType, OutputImageType>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ConjugateGradientImageFilter);

  /** Standard class type alias. */
  using Self = ConjugateGradientImageFilter;
  using Superclass = itk::InPlaceImageFilter<OutputImageType, OutputImageType>;
  using Pointer = itk::SmartPointer<Self>;
  using ConjugateGradientOperatorType = ConjugateGradientOperator<OutputImageType>;
  using OutputImagePointer = typename OutputImageType::Pointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(ConjugateGradientImageFilter);

  /** Get and Set macro*/
  itkGetMacro(NumberOfIterations, int);
  itkSetMacro(NumberOfIterations, int);

  void
  SetA(ConjugateGradientOperatorType * _arg);

  /** The input image to be updated.*/
  void
  SetX(const OutputImageType * OutputImage);

  /** The image called "B" in the CG algorithm.*/
  void
  SetB(const OutputImageType * OutputImage);

protected:
  ConjugateGradientImageFilter();
  ~ConjugateGradientImageFilter() override = default;

  OutputImagePointer
  GetX();
  OutputImagePointer
  GetB();

  /** Does the real work. */
  void
  GenerateData() override;

  /** Conjugate gradient requires the whole image */
  void
  GenerateInputRequestedRegion() override;
  void
  GenerateOutputInformation() override;

  ConjugateGradientOperatorType * m_A;

  int m_NumberOfIterations;
};
} // namespace rtk


#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkConjugateGradientImageFilter.hxx"
#endif

#endif
