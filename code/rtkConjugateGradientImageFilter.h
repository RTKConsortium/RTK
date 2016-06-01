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
#include "itkStatisticsImageFilter.h"
#include "itkImageFileWriter.h"

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
  typedef itk::MultiplyImageFilter<OutputImageType,OutputImageType,OutputImageType> MultiplyFilterType;
  typedef itk::StatisticsImageFilter<OutputImageType>                               StatisticsImageFilterType;
  typedef itk::ImageFileWriter<OutputImageType >                                    WriterType;
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

  itkGetMacro(IterationCosts, bool)
  itkSetMacro(IterationCosts, bool)
  
//  itkSetMacro(MeasureExecutionTimes, bool)
//  itkGetMacro(MeasureExecutionTimes, bool)

  /** If IterFileName is given, reconstructed images will be saved at each iteration */
  /** IterFileName must be "pathname/filename.extension" so it can be parsed as pathname/filename + extension */
  itkSetStringMacro(IterFileName);
  itkGetStringMacro(IterFileName);

  void SetA(ConjugateGradientOperatorPointerType _arg );

  /** The input image to be updated.*/
  void SetX(const OutputImageType* OutputImage);

  /** The image called "B" in the CG algorithm.*/
  void SetB(const OutputImageType* OutputImage);

  /** Set and Get the constant quantity BtWB for residual costs calculation */
  void SetC(const double _arg);
  const double GetC();

  /** Setter and getter for ResidualCosts storing array **/
  const std::vector<double> &GetResidualCosts();
  
protected:
  ConjugateGradientImageFilter();
  ~ConjugateGradientImageFilter(){}

  OutputImagePointer GetX();
  OutputImagePointer GetB();

  /** Does the real work. */
  virtual void GenerateData();

  /** Conjugate gradient requires the whole image */
  void GenerateInputRequestedRegion();
  void GenerateOutputInformation();

  ConjugateGradientOperatorPointerType m_A;

  int  m_NumberOfIterations;
  bool m_IterationCosts;
  std::vector<double> m_ResidualCosts;
  double m_C;

  void CalculateResidualCosts(OutputImagePointer R_kPlusOne, OutputImagePointer X_kPlusOne);
  void IterateImageWriter(OutputImagePointer X_kPlusOne, const int iter, const std::string FileName, const std::string Ext);

private:
  ConjugateGradientImageFilter(const Self &); //purposely not implemented
  void operator=(const Self &);  //purposely not implemented
  std::string m_IterFileName;
//  bool m_MeasureExecutionTimes;
};
} //namespace RTK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkConjugateGradientImageFilter.hxx"
#endif

#endif
