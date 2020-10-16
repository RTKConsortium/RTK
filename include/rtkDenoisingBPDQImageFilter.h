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

#ifndef rtkDenoisingBPDQImageFilter_h
#define rtkDenoisingBPDQImageFilter_h

#include "rtkForwardDifferenceGradientImageFilter.h"
#include "rtkBackwardDifferenceDivergenceImageFilter.h"

#include <itkCastImageFilter.h>
#include <itkSubtractImageFilter.h>
#include <itkMultiplyImageFilter.h>
#include <itkInPlaceImageFilter.h>

namespace rtk
{
/** \class DenoisingBPDQImageFilter
 * \brief Base class for Basis Pursuit DeQuantization denoising filters
 *
 * \author Cyril Mory
 *
 * \ingroup RTK IntensityImageFilters
 */

template <typename TOutputImage, typename TGradientImage>
class DenoisingBPDQImageFilter : public itk::InPlaceImageFilter<TOutputImage, TOutputImage>
{
public:
#if ITK_VERSION_MAJOR == 5 && ITK_VERSION_MINOR == 1
  ITK_DISALLOW_COPY_AND_ASSIGN(DenoisingBPDQImageFilter);
#else
  ITK_DISALLOW_COPY_AND_MOVE(DenoisingBPDQImageFilter);
#endif

  /** Standard class type alias. */
  using Self = DenoisingBPDQImageFilter;
  using Superclass = itk::InPlaceImageFilter<TOutputImage, TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(DenoisingBPDQImageFilter, ImageToImageFilter);

  /** Sub filter type definitions */
  typedef ForwardDifferenceGradientImageFilter<TOutputImage,
                                               typename TOutputImage::ValueType,
                                               typename TOutputImage::ValueType,
                                               TGradientImage>
    GradientFilterType;
  using MultiplyFilterType = itk::MultiplyImageFilter<TOutputImage>;
  using SubtractImageFilterType = itk::SubtractImageFilter<TOutputImage>;
  using SubtractGradientFilterType = itk::SubtractImageFilter<TGradientImage>;
  using ThresholdFilterType = itk::InPlaceImageFilter<TGradientImage>;
  using DivergenceFilterType = BackwardDifferenceDivergenceImageFilter<TGradientImage, TOutputImage>;

  itkGetMacro(NumberOfIterations, int);
  itkSetMacro(NumberOfIterations, int);

  itkSetMacro(Gamma, double);
  itkGetMacro(Gamma, double);

protected:
  DenoisingBPDQImageFilter();
  ~DenoisingBPDQImageFilter() override = default;

  void
  GenerateData() override;

  void
  GenerateOutputInformation() override;

  virtual ThresholdFilterType *
  GetThresholdFilter()
  {
    return nullptr;
  }

  /** Sub filter pointers */
  typename GradientFilterType::Pointer         m_GradientFilter;
  typename MultiplyFilterType::Pointer         m_MultiplyFilter;
  typename SubtractImageFilterType::Pointer    m_SubtractFilter;
  typename SubtractGradientFilterType::Pointer m_SubtractGradientFilter;
  typename DivergenceFilterType::Pointer       m_DivergenceFilter;

  double m_Gamma;
  double m_Beta;
  double m_MinSpacing;
  int    m_NumberOfIterations;
  bool   m_DimensionsProcessed[TOutputImage::ImageDimension];

private:
  virtual void
  SetPipelineForFirstIteration();
  virtual void
  SetPipelineAfterFirstIteration();
};

} // namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkDenoisingBPDQImageFilter.hxx"
#endif

#endif //__rtkDenoisingBPDQImageFilter__
