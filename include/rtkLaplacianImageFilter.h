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

#ifndef rtkLaplacianImageFilter_h
#define rtkLaplacianImageFilter_h

#include "rtkForwardDifferenceGradientImageFilter.h"
#include "rtkBackwardDifferenceDivergenceImageFilter.h"

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

template <typename OutputImageType, typename GradientImageType>
class LaplacianImageFilter : public itk::ImageToImageFilter<OutputImageType, OutputImageType>
{
public:
#if ITK_VERSION_MAJOR == 5 && ITK_VERSION_MINOR == 1
  ITK_DISALLOW_COPY_AND_ASSIGN(LaplacianImageFilter);
#else
  ITK_DISALLOW_COPY_AND_MOVE(LaplacianImageFilter);
#endif

  /** Standard class type alias. */
  using Self = LaplacianImageFilter;
  using Superclass = itk::ImageToImageFilter<OutputImageType, OutputImageType>;
  using Pointer = itk::SmartPointer<Self>;
  using OutputImagePointer = typename OutputImageType::Pointer;
  using GradientFilterType = rtk::ForwardDifferenceGradientImageFilter<OutputImageType,
                                                                       typename OutputImageType::ValueType,
                                                                       typename OutputImageType::ValueType,
                                                                       GradientImageType>;
  using DivergenceFilterType = rtk::BackwardDifferenceDivergenceImageFilter<GradientImageType, OutputImageType>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(LaplacianImageFilter, itk::ImageToImageFilter);

protected:
  LaplacianImageFilter();
  ~LaplacianImageFilter() override = default;

  /** Does the real work. */
  void
  GenerateData() override;

  /** Handle regions */
  void
  GenerateOutputInformation() override;

  typename GradientFilterType::Pointer   m_Gradient;
  typename DivergenceFilterType::Pointer m_Divergence;
};
} // namespace rtk


#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkLaplacianImageFilter.hxx"
#endif

#endif
