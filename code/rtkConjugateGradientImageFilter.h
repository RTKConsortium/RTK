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

#ifndef __rtkConjugateGradientImageFilter_h
#define __rtkConjugateGradientImageFilter_h

#include "itkImageToImageFilter.h"
#include "itkSubtractImageFilter.h"

#include "rtkConjugateGradientGetR_kPlusOneImageFilter.h"
#include "rtkConjugateGradientGetX_kPlusOneImageFilter.h"
#include "rtkConjugateGradientGetP_kPlusOneImageFilter.h"

#include "rtkConjugateGradientOperator.h"
#include "itkTimeProbe.h"

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
class ConjugateGradientImageFilter : public itk::ImageToImageFilter< OutputImageType,  OutputImageType>
{
public:
    /** Standard class typedefs. */
    typedef ConjugateGradientImageFilter                                              Self;
    typedef itk::ImageToImageFilter< OutputImageType, OutputImageType>                Superclass;
    typedef itk::SmartPointer< Self >                                                 Pointer;
    typedef itk::SubtractImageFilter<OutputImageType,OutputImageType,OutputImageType> SubtractFilterType;
    typedef ConjugateGradientOperator<OutputImageType>                                ConjugateGradientOperatorType;
    typedef typename ConjugateGradientOperatorType::Pointer                           ConjugateGradientOperatorPointerType;
    typedef typename OutputImageType::Pointer                                         OutputImagePointer;
    typedef typename rtk::ConjugateGradientGetP_kPlusOneImageFilter<OutputImageType>  GetP_kPlusOne_FilterType;
    typedef typename rtk::ConjugateGradientGetR_kPlusOneImageFilter<OutputImageType>  GetR_kPlusOne_FilterType;
    typedef typename rtk::ConjugateGradientGetX_kPlusOneImageFilter<OutputImageType>  GetX_kPlusOne_FilterType;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(ConjugateGradientImageFilter, itk::ImageToImageFilter)

    /** Get and Set macro*/
    itkGetMacro(NumberOfIterations, int)
    itkSetMacro(NumberOfIterations, int)

    itkSetMacro(MeasureExecutionTimes, bool)
    itkGetMacro(MeasureExecutionTimes, bool)

    void SetA(ConjugateGradientOperatorPointerType _arg );

    /** The input image to be updated.*/
    void SetX(const OutputImageType* OutputImage);

    /** The image called "B" in the CG algorithm.*/
    void SetB(const OutputImageType* OutputImage);

protected:
    ConjugateGradientImageFilter();
    ~ConjugateGradientImageFilter(){}

    OutputImagePointer GetX();
    OutputImagePointer GetB();

    /** Does the real work. */
    virtual void GenerateData();

    /** Conjugate gradient requires the whole image */
    void GenerateInputRequestedRegion();

private:
    ConjugateGradientImageFilter(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented

    ConjugateGradientOperatorPointerType m_A;

    int m_NumberOfIterations;
    bool m_MeasureExecutionTimes;

};
} //namespace RTK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkConjugateGradientImageFilter.txx"
#endif

#endif
