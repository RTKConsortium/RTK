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

#ifndef rtkTotalNuclearVariationDenoisingBPDQImageFilter_h
#define rtkTotalNuclearVariationDenoisingBPDQImageFilter_h

#include "rtkSingularValueThresholdImageFilter.h"
#include "rtkDenoisingBPDQImageFilter.h"

namespace rtk
{
/** \class TotalNuclearVariationDenoisingBPDQImageFilter
 * \brief Performs total nuclear variation denoising
 *
 * This filter implements "Joint reconstruction of multi-channel, spectral CT data
 * via constrained total nuclear variation minimization", by Rigie & LaRivière, in
 * Physics in Medicine and Biology 2015.
 *
 * It uses basis pursuit dequantization, and is (mathematically) only a generalization
 * of the TotalVariationDenoisingBPDQImageFilter to process multiple channel images.
 * It outputs a multiple channel image close to the input one, for which the spatial
 * gradient of each channel is sparser, and the gradient vectors are more similar (ie. colinear) across channels,
 * than in the input.
 *
 * The order of the channels is not taken into account, which makes this regularization
 * more suitable when the channels describe materials (i.e. in spectral CT) or
 * colors (i.e. in RGB images) than when they describe time frames (i.e. in dynamic CT).
 *
 * \dot
 * digraph TotalNuclearVariationDenoisingBPDQImageFilter
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
 *   FI_SingularValueThreshold [ label="rtk::SingularValueThresholdImageFilter" URL="\ref rtk::SingularValueThresholdImageFilter"];
 *   FI_OutOfSingularValueThreshold [label="", fixedsize="false", width=0, height=0, shape=none];
 *
 *   FI_Input -> FI_Multiply;
 *   FI_Multiply -> FI_Gradient;
 *   FI_Gradient -> FI_SingularValueThreshold;
 *   FI_SingularValueThreshold -> FI_OutOfSingularValueThreshold [style=dashed];
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
 *   SingularValueThreshold [ label="rtk::SingularValueThresholdImageFilter" URL="\ref rtk::SingularValueThresholdImageFilter"];
 *   OutOfSubtract [label="", fixedsize="false", width=0, height=0, shape=none];
 *   OutOfSingularValueThreshold [label="", fixedsize="false", width=0, height=0, shape=none];
 *   BeforeDivergence [label="", fixedsize="false", width=0, height=0, shape=none];
 *
 *   Input -> Subtract;
 *   Divergence -> Subtract;
 *   Subtract -> OutOfSubtract;
 *   OutOfSubtract -> Output;
 *   OutOfSubtract -> Multiply;
 *   Multiply -> Gradient;
 *   Gradient -> SubtractGradient;
 *   SubtractGradient -> SingularValueThreshold;
 *   SingularValueThreshold -> OutOfSingularValueThreshold;
 *   OutOfSingularValueThreshold -> BeforeDivergence [style=dashed, constraint=false];
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
    itk::Image< itk::CovariantVector < typename TOutputImage::ValueType, TOutputImage::ImageDimension - 1>,
    TOutputImage::ImageDimension > >
class TotalNuclearVariationDenoisingBPDQImageFilter :
        public rtk::DenoisingBPDQImageFilter< TOutputImage, TGradientImage >
{
public:

  /** Standard class typedefs. */
  typedef TotalNuclearVariationDenoisingBPDQImageFilter                 Self;
  typedef rtk::DenoisingBPDQImageFilter< TOutputImage, TGradientImage > Superclass;
  typedef itk::SmartPointer<Self>                                       Pointer;
  typedef itk::SmartPointer<const Self>                                 ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self)

  /** Run-time type information (and related methods). */
  itkTypeMacro(TotalNuclearVariationDenoisingBPDQImageFilter, DenoisingBPDQImageFilter)

  /** Sub filter type definitions */
  typedef SingularValueThresholdImageFilter<TGradientImage>             SingularValueThresholdFilterType;

protected:
  TotalNuclearVariationDenoisingBPDQImageFilter();
  ~TotalNuclearVariationDenoisingBPDQImageFilter() {}

  void GenerateOutputInformation() ITK_OVERRIDE;

  /** Sub filter pointers */
  typename SingularValueThresholdFilterType::Pointer    m_ThresholdFilter;
  virtual typename Superclass::ThresholdFilterType* GetThresholdFilter() ITK_OVERRIDE
  {
    return dynamic_cast<typename Superclass::ThresholdFilterType*>(this->m_ThresholdFilter.GetPointer());
  }

private:
  TotalNuclearVariationDenoisingBPDQImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkTotalNuclearVariationDenoisingBPDQImageFilter.hxx"
#endif

#endif //__rtkTotalNuclearVariationDenoisingBPDQImageFilter__
