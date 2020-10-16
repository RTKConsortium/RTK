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

#ifndef rtkDivergenceOfGradientConjugateGradientOperator_h
#define rtkDivergenceOfGradientConjugateGradientOperator_h

#include "rtkConjugateGradientOperator.h"

#include "rtkForwardDifferenceGradientImageFilter.h"
#include "rtkBackwardDifferenceDivergenceImageFilter.h"

namespace rtk
{
/** \class DivergenceOfGradientConjugateGradientOperator
 * \brief Computes the divergence of the gradient of an image. To be used
 * with the ConjugateGradientImageFilter
 *
 * \author Cyril Mory
 *
 * \ingroup RTK IntensityImageFilters
 */
template <class TInputImage>
class DivergenceOfGradientConjugateGradientOperator : public ConjugateGradientOperator<TInputImage>
{
public:
#if ITK_VERSION_MAJOR == 5 && ITK_VERSION_MINOR == 1
  ITK_DISALLOW_COPY_AND_ASSIGN(DivergenceOfGradientConjugateGradientOperator);
#else
  ITK_DISALLOW_COPY_AND_MOVE(DivergenceOfGradientConjugateGradientOperator);
#endif

  /** Extract dimension from input and output image. */
  itkStaticConstMacro(InputImageDimension, unsigned int, TInputImage::ImageDimension);

  /** Convenient type alias for simplifying declarations. */
  using InputImageType = TInputImage;

  /** Standard class type alias. */
  using Self = DivergenceOfGradientConjugateGradientOperator;
  using Superclass = itk::ImageToImageFilter<InputImageType, InputImageType>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(DivergenceOfGradientConjugateGradientOperator, ImageToImageFilter);

  /** Image type alias support. */
  using InputPixelType = typename InputImageType::PixelType;
  using InputImageRegionType = typename InputImageType::RegionType;
  using InputSizeType = typename InputImageType::SizeType;

  /** Sub filter type definitions */
  using GradientFilterType = ForwardDifferenceGradientImageFilter<TInputImage>;
  using GradientImageType = typename GradientFilterType::OutputImageType;
  using DivergenceFilterType = BackwardDifferenceDivergenceImageFilter<GradientImageType>;

  void
  SetDimensionsProcessed(bool * arg);

protected:
  DivergenceOfGradientConjugateGradientOperator();
  ~DivergenceOfGradientConjugateGradientOperator() override = default;

  void
  GenerateData() override;

  void
  GenerateOutputInformation() override;

  /** Sub filter pointers */
  typename GradientFilterType::Pointer   m_GradientFilter;
  typename DivergenceFilterType::Pointer m_DivergenceFilter;

  bool m_DimensionsProcessed[TInputImage::ImageDimension];
};

} // namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkDivergenceOfGradientConjugateGradientOperator.hxx"
#endif

#endif //__rtkDivergenceOfGradientConjugateGradientOperator__
