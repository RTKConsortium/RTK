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

#ifndef rtkConjugateGradientImageFilter_h
#define rtkConjugateGradientImageFilter_h

#include <itkSubtractImageFilter.h>
#include <itkStatisticsImageFilter.h>
#include <itkTimeProbe.h>

#include "rtkDotProductImageFilter.h"
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
 * in http://en.wikipedia.org/wiki/Conjugate_gradient_method
 *
*/

template< typename OutputImageType>
class ConjugateGradientImageFilter : public itk::InPlaceImageFilter< OutputImageType,  OutputImageType>
{
public:
   
  /** Standard class typedefs. */
  typedef ConjugateGradientImageFilter                                              Self;
  typedef itk::InPlaceImageFilter< OutputImageType, OutputImageType>                Superclass;
  typedef itk::SmartPointer< Self >                                                 Pointer;
  typedef ConjugateGradientOperator<OutputImageType>                                ConjugateGradientOperatorType;
  typedef typename ConjugateGradientOperatorType::Pointer                           ConjugateGradientOperatorPointerType;
  typedef typename OutputImageType::Pointer                                         OutputImagePointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self)

  /** Run-time type information (and related methods). */
  itkTypeMacro(ConjugateGradientImageFilter, itk::InPlaceImageFilter)

  /** Get and Set macro*/
  itkGetMacro(NumberOfIterations, int)
  itkSetMacro(NumberOfIterations, int)

  /** Displays the conjugate gradient cost function at each iteration. */
  itkGetMacro(IterationCosts, bool)
  itkSetMacro(IterationCosts, bool)

  void SetA(ConjugateGradientOperatorPointerType _arg );

  /** The input image to be updated.*/
  void SetX(const OutputImageType* OutputImage);

  /** The image called "B" in the CG algorithm.*/
  void SetB(const OutputImageType* OutputImage);

  /** Set and Get the constant quantity BtWB for residual costs calculation */
  void SetC(const double _arg);
  double GetC();

  /** Getter for ResidualCosts storing array **/
  const std::vector<double> &GetResidualCosts();
  
protected:
  ConjugateGradientImageFilter();
  virtual ~ConjugateGradientImageFilter() ITK_OVERRIDE {}

  OutputImagePointer GetX();
  OutputImagePointer GetB();

  /** Does the real work. */
  void GenerateData() ITK_OVERRIDE;

  /** Conjugate gradient requires the whole image */
  void GenerateInputRequestedRegion() ITK_OVERRIDE;
  void GenerateOutputInformation() ITK_OVERRIDE;

  ConjugateGradientOperatorPointerType m_A;

  int                 m_NumberOfIterations;
  bool                m_IterationCosts;
  std::vector<double> m_ResidualCosts;
  double              m_C;

private:
  ConjugateGradientImageFilter(const Self &); //purposely not implemented
  void operator=(const Self &);  //purposely not implemented
};
} //namespace RTK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkConjugateGradientImageFilter.hxx"
#endif

#endif
