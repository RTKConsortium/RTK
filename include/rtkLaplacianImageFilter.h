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

#ifndef rtkLaplacianImageFilter_h
#define rtkLaplacianImageFilter_h

#include "rtkForwardDifferenceGradientImageFilter.h"
#include "rtkBackwardDifferenceDivergenceImageFilter.h"
#include "itkMultiplyImageFilter.h"

#ifdef RTK_USE_CUDA
#  include <itkCudaImage.h>
#endif

namespace rtk
{

/** \class LaplacianImageFilter
 * \brief Computes the laplacian of the input image
 *
 * Computes the gradient of the input image, then the divergence of
 * this gradient. The exact definition of the gradient and divergence filters can
 * be found in Chambolle, Antonin. "An Algorithm for Total
 * Variation Minimization and Applications." J. Math. Imaging Vis. 20,
 * no. 1-2 (January 2004): 89-97. The border conditions are described there.
 *
 * \ingroup RTK IntensityImageFilters
 */

template <typename TOutputImage>
class ITK_TEMPLATE_EXPORT LaplacianImageFilter : public itk::ImageToImageFilter<TOutputImage, TOutputImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(LaplacianImageFilter);

  /** Standard class type alias. */
  using Self = LaplacianImageFilter;
  using Superclass = itk::ImageToImageFilter<TOutputImage, TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using OutputImagePointer = typename TOutputImage::Pointer;
  using CPUImageType = itk::Image<typename TOutputImage::PixelType, TOutputImage::ImageDimension>;
  using VectorPixelType = itk::CovariantVector<typename TOutputImage::ValueType, TOutputImage::ImageDimension>;

#ifdef RTK_USE_CUDA
  using GradientImageType = typename std::conditional_t<std::is_same_v<TOutputImage, CPUImageType>,
                                                        itk::Image<VectorPixelType, TOutputImage::ImageDimension>,
                                                        itk::CudaImage<VectorPixelType, TOutputImage::ImageDimension>>;
#else
  using GradientImageType = itk::Image<VectorPixelType, TOutputImage::ImageDimension>;
#endif
  using GradientFilterType = rtk::ForwardDifferenceGradientImageFilter<TOutputImage,
                                                                       typename TOutputImage::ValueType,
                                                                       typename TOutputImage::ValueType,
                                                                       GradientImageType>;
  using DivergenceFilterType = rtk::BackwardDifferenceDivergenceImageFilter<GradientImageType, TOutputImage>;
  using MultiplyImageFilterType = itk::MultiplyImageFilter<GradientImageType, TOutputImage>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(LaplacianImageFilter);

  void
  SetWeights(const TOutputImage * weights);
  typename TOutputImage::ConstPointer
  GetWeights();

protected:
  LaplacianImageFilter();
  ~LaplacianImageFilter() override = default;

  /** Does the real work. */
  void
  GenerateData() override;

  /** Handle regions */
  void
  GenerateOutputInformation() override;

  typename GradientFilterType::Pointer      m_Gradient;
  typename DivergenceFilterType::Pointer    m_Divergence;
  typename MultiplyImageFilterType::Pointer m_Multiply;
};
} // namespace rtk


#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkLaplacianImageFilter.hxx"
#endif

#endif
