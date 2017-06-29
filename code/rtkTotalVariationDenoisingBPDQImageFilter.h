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

#ifndef rtkTotalVariationDenoisingBPDQImageFilter_h
#define rtkTotalVariationDenoisingBPDQImageFilter_h

#include "rtkDenoisingBPDQImageFilter.h"
#include "rtkMagnitudeThresholdImageFilter.h"

#include <itkPeriodicBoundaryCondition.h>

namespace rtk
{
/** \class TotalVariationDenoisingBPDQImageFilter
 * \brief Applies a total variation denoising, only alm_SingularValueThresholdFilterong the dimensions specified, on an image.
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
 * digraph TotalVariationDenoisingBPDQImageFilter
 * {
 *
 * subgraph clusterFirstIteration
 *   {
 *   label="First iteration"
 *
 *   FI_Input [label="Input"];
 *   FI_Input [shape=Mdiamond];
 *
 *   node [shape=box];
 *   FI_Multiply [ label="itk::MultiplyImageFilter (by beta)" URL="\ref itk::MultiplyImageFilter"];
 *   FI_Gradient [ label="rtk::ForwardDifferenceGradientImageFilter" URL="\ref rtk::ForwardDifferenceGradientImageFilter"];
 *   FI_MagnitudeThreshold [ label="rtk::MagnitudeThresholdImageFilter" URL="\ref rtk::MagnitudeThresholdImageFilter"];
 *   FI_OutOfMagnitudeTreshold [label="", fixedsize="false", width=0, height=0, shape=none];
 *
 *   FI_Input -> FI_Multiply;
 *   FI_Multiply -> FI_Gradient;
 *   FI_Gradient -> FI_MagnitudeThreshold;
 *   FI_MagnitudeThreshold -> FI_OutOfMagnitudeTreshold [style=dashed];
 *   }
 *
 * subgraph clusterAfterFirstIteration
 *   {
 *   label="After first iteration"
 *
 *   Input [label="Input"];
 *   Input [shape=Mdiamond];
 *   Output [label="Output"];
 *   Output [shape=Mdiamond];
 *
 *   node [shape=box];
 *   Divergence [ label="rtk::BackwardDifferenceDivergenceImageFilter" URL="\ref rtk::BackwardDifferenceDivergenceImageFilter"];
 *   Subtract [ label="itk::SubtractImageFilter" URL="\ref itk::SubtractImageFilter"];
 *   Multiply [ label="itk::MultiplyImageFilter (by beta)" URL="\ref itk::MultiplyImageFilter"];
 *   Gradient [ label="rtk::ForwardDifferenceGradientImageFilter" URL="\ref rtk::ForwardDifferenceGradientImageFilter"];
 *   SubtractGradient [ label="itk::SubtractImageFilter" URL="\ref itk::SubtractImageFilter"];
 *   MagnitudeThreshold [ label="rtk::MagnitudeThresholdImageFilter" URL="\ref rtk::MagnitudeThresholdImageFilter"];
 *   OutOfSubtract [label="", fixedsize="false", width=0, height=0, shape=none];
 *   OutOfMagnitudeTreshold [label="", fixedsize="false", width=0, height=0, shape=none];
 *   BeforeDivergence [label="", fixedsize="false", width=0, height=0, shape=none];
 *
 *   Input -> Subtract;
 *   Divergence -> Subtract;
 *   Subtract -> OutOfSubtract;
 *   OutOfSubtract -> Output;
 *   OutOfSubtract -> Multiply;
 *   Multiply -> Gradient;
 *   Gradient -> SubtractGradient;
 *   SubtractGradient -> MagnitudeThreshold;
 *   MagnitudeThreshold -> OutOfMagnitudeTreshold;
 *   OutOfMagnitudeTreshold -> BeforeDivergence [style=dashed, constraint=false];
 *   BeforeDivergence -> Divergence;
 *   BeforeDivergence -> SubtractGradient;
 *   }
 *
 * }
 * \enddot
 *
 * \author Cyril Mory
 *
 * \ingroup IntensityImageFilters
 */

template< typename TOutputImage, typename TGradientImage =
    itk::Image< itk::CovariantVector < typename TOutputImage::ValueType, TOutputImage::ImageDimension >, 
    TOutputImage::ImageDimension > >
class TotalVariationDenoisingBPDQImageFilter :
        public rtk::DenoisingBPDQImageFilter< TOutputImage, TGradientImage >
{
public:

  /** Standard class typedefs. */
  typedef TotalVariationDenoisingBPDQImageFilter                        Self;
  typedef rtk::DenoisingBPDQImageFilter< TOutputImage, TGradientImage > Superclass;
  typedef itk::SmartPointer<Self>                                       Pointer;
  typedef itk::SmartPointer<const Self>                                 ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self)

  /** Run-time type information (and related methods). */
  itkTypeMacro(TotalVariationDenoisingBPDQImageFilter, DenoisingBPDQImageFilter)

  /** Sub filter type definitions */
  typedef MagnitudeThresholdImageFilter<TGradientImage>                 MagnitudeThresholdFilterType;

  void SetDimensionsProcessed(bool* arg);

  /** In some cases, regularization must use periodic boundary condition */
  void SetBoundaryConditionToPeriodic();

protected:
  TotalVariationDenoisingBPDQImageFilter();
  ~TotalVariationDenoisingBPDQImageFilter() {}

  void GenerateOutputInformation() ITK_OVERRIDE;

  /** Sub filter pointers */
  typename MagnitudeThresholdFilterType::Pointer   m_ThresholdFilter;
  virtual typename Superclass::ThresholdFilterType* GetThresholdFilter() ITK_OVERRIDE
  {
    return dynamic_cast<typename Superclass::ThresholdFilterType*>(this->m_ThresholdFilter.GetPointer());
  }

private:
  TotalVariationDenoisingBPDQImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkTotalVariationDenoisingBPDQImageFilter.hxx"
#endif

#endif //__rtkTotalVariationDenoisingBPDQImageFilter__
