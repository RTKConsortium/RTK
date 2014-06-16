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

#ifndef __rtkTotalVariationDenoisingBPDQImageFilter_h
#define __rtkTotalVariationDenoisingBPDQImageFilter_h


#include "rtkForwardDifferenceGradientImageFilter.h"
#include "rtkBackwardDifferenceDivergenceImageFilter.h"
#include "rtkMagnitudeThresholdImageFilter.h"
#include "rtkMacro.h"

//#include <itkImageToImageFilter.h>
#include <itkCastImageFilter.h>
//#include <itkImage.h>
#include <itkSubtractImageFilter.h>
#include <itkMultiplyImageFilter.h>
#include <itkPeriodicBoundaryCondition.h>

namespace rtk
{
/** \class TotalVariationDenoisingBPDQImageFilter
 * \brief Applies a total variation denoising, only along the dimensions specified, on an image.
 *
 * This filter finds the minimum of || f - f_0 ||_2^2 + gamma * TV(f)
 * using basis pursuit dequantization, where f is the current image, f_0 the
 * input image, and TV the total variation calculated with only the gradients
 * along the dimensions specified. This filter can be used, for example, to
 * perform 3D total variation denoising on a 4D dataset
 * (by calling SetDimensionsProcessed([true true true false]).
 * More information on the algorithm can be found at
 * http://wiki.epfl.ch/bpdq#download
 *
 * \dot
 * digraph TotalVariationDenoisingBPDQImageFilter {
 *
 * Input [label="Input"];
 * Input [shape=Mdiamond];
 * Output [label="Output"];
 * Output [shape=Mdiamond];
 *
 * node [shape=box];
 * ZeroMultiply [ label="itk::MultiplyImageFilter (by zero)" URL="\ref itk::MultiplyImageFilter"];
 * ZeroGradient [ label="rtk::ForwardDifferenceGradientImageFilter" URL="\ref rtk::ForwardDifferenceGradientImageFilter"];
 * Divergence [ label="rtk::BackwardDifferenceDivergenceImageFilter" URL="\ref rtk::BackwardDifferenceDivergenceImageFilter"];
 * Subtract [ label="itk::SubtractImageFilter" URL="\ref itk::SubtractImageFilter"];
 * Multiply [ label="itk::MultiplyImageFilter (by beta)" URL="\ref itk::MultiplyImageFilter"];
 * Gradient [ label="rtk::ForwardDifferenceGradientImageFilter" URL="\ref rtk::ForwardDifferenceGradientImageFilter"];
 * SubtractGradient [ label="itk::SubtractImageFilter" URL="\ref itk::SubtractImageFilter"];
 * MagnitudeThreshold [ label="rtk::MagnitudeThresholdImageFilter" URL="\ref rtk::MagnitudeThresholdImageFilter"];
 * OutOfZeroGradient [label="", fixedsize="false", width=0, height=0, shape=none];
 *
 * Input -> ZeroMultiply;
 * Input -> Subtract;
 * ZeroMultiply -> ZeroGradient;
 * ZeroGradient -> OutOfZeroGradient [arrowhead=None];
 * OutOfZeroGradient -> Divergence;
 * OutOfZeroGradient -> SubtractGradient;
 * Divergence -> Subtract;
 * Subtract -> Multiply;
 * Multiply -> Gradient;
 * Gradient -> SubtractGradient;
 * SubtractGradient -> MagnitudeThreshold;
 * MagnitudeThreshold -> OutOfZeroGradient [style=dashed];
 * Subtract -> Output;
 * }
 * \enddot
 *
 * \author Cyril Mory
 *
 * \ingroup IntensityImageFilters
 */

template< typename TOutputImage, typename TGradientOutputImage = 
    itk::Image< itk::CovariantVector < typename TOutputImage::ValueType, TOutputImage::ImageDimension >, 
    TOutputImage::ImageDimension > >
class TotalVariationDenoisingBPDQImageFilter :
        public itk::ImageToImageFilter< TOutputImage, TOutputImage >
{
public:

  /** Standard class typedefs. */
  typedef TotalVariationDenoisingBPDQImageFilter               Self;
  typedef itk::ImageToImageFilter< TOutputImage, TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                              Pointer;
  typedef itk::SmartPointer<const Self>                        ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self)

  /** Run-time type information (and related methods). */
  itkTypeMacro(TotalVariationDenoisingBPDQImageFilter, ImageToImageFilter)

  /** Sub filter type definitions */
  typedef ForwardDifferenceGradientImageFilter<TOutputImage, 
            typename TOutputImage::ValueType, typename TOutputImage::ValueType, 
            TGradientOutputImage>                                                     GradientFilterType;
  typedef itk::MultiplyImageFilter<TOutputImage>                                      MultiplyFilterType;
  typedef itk::SubtractImageFilter<TOutputImage>                                      SubtractImageFilterType;
  typedef itk::SubtractImageFilter<TGradientOutputImage>                              SubtractGradientFilterType;
  typedef MagnitudeThresholdImageFilter<TGradientOutputImage>                         MagnitudeThresholdFilterType;
  typedef BackwardDifferenceDivergenceImageFilter<TGradientOutputImage, TOutputImage> DivergenceFilterType;

  itkGetMacro(NumberOfIterations, int)
  itkSetMacro(NumberOfIterations, int)

  itkSetMacro(Gamma, double)
  itkGetMacro(Gamma, double)

  void SetDimensionsProcessed(bool* arg);

  /** In some cases, regularization must use periodic boundary condition */
  void SetBoundaryConditionToPeriodic();

protected:
  TotalVariationDenoisingBPDQImageFilter();
  virtual ~TotalVariationDenoisingBPDQImageFilter();

  virtual void GenerateData();

  virtual void GenerateOutputInformation();

  /** Sub filter pointers */
  typename GradientFilterType::Pointer             m_GradientFilter;
  typename GradientFilterType::Pointer             m_ZeroGradientFilter;
  typename MultiplyFilterType::Pointer             m_MultiplyFilter;
  typename MultiplyFilterType::Pointer             m_ZeroMultiplyFilter;
  typename SubtractImageFilterType::Pointer        m_SubtractFilter;
  typename SubtractGradientFilterType::Pointer     m_SubtractGradientFilter;
  typename MagnitudeThresholdFilterType::Pointer   m_MagnitudeThresholdFilter;
  typename DivergenceFilterType::Pointer           m_DivergenceFilter;

  double m_Gamma;
  int    m_NumberOfIterations;
  bool   m_DimensionsProcessed[TOutputImage::ImageDimension];

  // In some cases, regularization must use periodic boundary condition
  typename itk::ImageBoundaryCondition<TOutputImage, TOutputImage>         * m_BoundaryConditionForGradientFilter;
  typename itk::ImageBoundaryCondition<TGradientOutputImage, TGradientOutputImage> * m_BoundaryConditionForDivergenceFilter;

private:
  TotalVariationDenoisingBPDQImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  double m_Beta;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkTotalVariationDenoisingBPDQImageFilter.txx"
#endif

#endif //__rtkTotalVariationDenoisingBPDQImageFilter__
