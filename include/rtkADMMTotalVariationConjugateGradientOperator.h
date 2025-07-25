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

#ifndef rtkADMMTotalVariationConjugateGradientOperator_h
#define rtkADMMTotalVariationConjugateGradientOperator_h

#include <itkMultiplyImageFilter.h>
#include <itkSubtractImageFilter.h>

#include "rtkConjugateGradientOperator.h"
#include "rtkBackProjectionImageFilter.h"
#include "rtkForwardProjectionImageFilter.h"
#include "rtkForwardDifferenceGradientImageFilter.h"
#include "rtkBackwardDifferenceDivergenceImageFilter.h"
#include "rtkDisplacedDetectorImageFilter.h"
#include "rtkMultiplyByVectorImageFilter.h"
#include "rtkThreeDCircularProjectionGeometry.h"

#ifdef RTK_USE_CUDA
#  include <itkCudaImage.h>
#endif

namespace rtk
{

/** \class ADMMTotalVariationConjugateGradientOperator
 * \brief Implements the operator A used in the conjugate gradient step
 * of ADMM reconstruction with total variation regularization
 *
 * This filter implements the operator A used in the conjugate gradient step
 * of a reconstruction method based on compressed sensing.
 * The method attempts to find the f that minimizes
 * || sqrt(D) (Rf -p) ||_2^2 + alpha * TV(f),
 * with R the forward projection operator, p the measured projections,
 * D the displaced detector weighting matrix, and TV the total variation.
 * Details on the method and the calculations can be found in
 *
 * Mory, C., B. Zhang, V. Auvray, M. Grass, D. Schafer, F. Peyrin, S. Rit, P. Douek,
 * and L. Boussel. "ECG-Gated C-Arm Computed Tomography Using L1 Regularization."
 * In Proceedings of the 20th European Signal Processing Conference (EUSIPCO), 2728-32, 2012.
 *
 * This filter takes in input f and outputs R_t D R f - beta div(grad(f)).
 *
 * \dot
 * digraph ADMMTotalVariationConjugateGradientOperator {
 *
 * Input0 [ label="Input 0 (Volume)"];
 * Input0 [shape=Mdiamond];
 * Input1 [label="Input 1 (Projections)"];
 * Input1 [shape=Mdiamond];
 * Output [label="Output (Volume)"];
 * Output [shape=Mdiamond];
 *
 * node [shape=box];
 * ZeroMultiplyVolume [label="itk::MultiplyImageFilter (by zero)" URL="\ref itk::MultiplyImageFilter"];
 * ZeroMultiplyProjections [label="itk::MultiplyImageFilter (by zero)" URL="\ref itk::MultiplyImageFilter"];
 * BeforeZeroMultiplyVolume [label="", fixedsize="false", width=0, height=0, shape=none];
 * Subtract [ label="itk::SubtractImageFilter" URL="\ref itk::SubtractImageFilter"];
 * Multiply [ label="itk::MultiplyImageFilter (by lambda)" URL="\ref itk::MultiplyImageFilter"];
 * Gradient [ label="rtk::ForwardDifferenceGradientImageFilter" URL="\ref rtk::ForwardDifferenceGradientImageFilter"];
 * Divergence [ label="rtk::BackwardDifferenceDivergenceImageFilter"
 *              URL="\ref rtk::BackwardDifferenceDivergenceImageFilter"];
 * BackProjection [ label="rtk::BackProjectionImageFilter" URL="\ref rtk::BackProjectionImageFilter"];
 * ForwardProjection [ label="rtk::ForwardProjectionImageFilter" URL="\ref rtk::ForwardProjectionImageFilter"];
 * Displaced [ label="rtk::DisplacedDetectorImageFilter" URL="\ref rtk::DisplacedDetectorImageFilter"];
 *
 * Input0 -> BeforeZeroMultiplyVolume [arrowhead=none];
 * BeforeZeroMultiplyVolume -> ZeroMultiplyVolume;
 * BeforeZeroMultiplyVolume -> ForwardProjection;
 * BeforeZeroMultiplyVolume -> Gradient;
 * Input1 -> ZeroMultiplyProjections;
 * ZeroMultiplyProjections -> ForwardProjection;
 * ZeroMultiplyVolume -> BackProjection;
 * ForwardProjection -> Displaced;
 * Displaced -> BackProjection;
 * BackProjection -> Subtract;
 * Gradient -> Divergence;
 * Divergence -> Multiply;
 * Multiply -> Subtract;
 * Subtract -> Output;
 *
 * }
 * \enddot
 *
 * \test rtkadmmtotalvariationtest.cxx
 *
 * \author Cyril Mory
 *
 * \ingroup RTK ReconstructionAlgorithm
 */

template <typename TOutputImage>
class ITK_TEMPLATE_EXPORT ADMMTotalVariationConjugateGradientOperator : public ConjugateGradientOperator<TOutputImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ADMMTotalVariationConjugateGradientOperator);

  /** Standard class type alias. */
  using Self = ADMMTotalVariationConjugateGradientOperator;
  using Superclass = ConjugateGradientOperator<TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(ADMMTotalVariationConjugateGradientOperator);

  using BackProjectionFilterType = rtk::BackProjectionImageFilter<TOutputImage, TOutputImage>;
  using BackProjectionFilterPointer = typename BackProjectionFilterType::Pointer;

  using ForwardProjectionFilterType = rtk::ForwardProjectionImageFilter<TOutputImage, TOutputImage>;
  using ForwardProjectionFilterPointer = typename ForwardProjectionFilterType::Pointer;

  using MultiplyFilterType = itk::MultiplyImageFilter<TOutputImage>;
  using SubtractFilterType = itk::SubtractImageFilter<TOutputImage>;

  using VectorPixelType = itk::CovariantVector<typename TOutputImage::ValueType, TOutputImage::ImageDimension>;
  using CPUImageType = itk::Image<typename TOutputImage::PixelType, TOutputImage::ImageDimension>;
#ifdef RTK_USE_CUDA
  typedef
    typename std::conditional<std::is_same<TOutputImage, CPUImageType>::value,
                              itk::Image<VectorPixelType, TOutputImage::ImageDimension>,
                              itk::CudaImage<VectorPixelType, TOutputImage::ImageDimension>>::type GradientImageType;
#else
  using GradientImageType = itk::Image<VectorPixelType, TOutputImage::ImageDimension>;
#endif

  using GradientFilterType = ForwardDifferenceGradientImageFilter<TOutputImage,
                                                                  typename TOutputImage::ValueType,
                                                                  typename TOutputImage::ValueType,
                                                                  GradientImageType>;
  using DivergenceFilterType = rtk::BackwardDifferenceDivergenceImageFilter<GradientImageType, TOutputImage>;
  using DisplacedDetectorFilterType = rtk::DisplacedDetectorImageFilter<TOutputImage>;
  using GatingWeightsFilterType = rtk::MultiplyByVectorImageFilter<TOutputImage>;

  /** Set the backprojection filter*/
  void
  SetBackProjectionFilter(BackProjectionFilterType * _arg);

  /** Set the forward projection filter*/
  void
  SetForwardProjectionFilter(ForwardProjectionFilterType * _arg);

  /** Pass the geometry to all filters needing it */
  itkSetObjectMacro(Geometry, ThreeDCircularProjectionGeometry);

  /** Set the regularization parameter */
  itkSetMacro(Beta, float);

  /** In the case of a gated reconstruction, set the gating weights */
  void
  SetGatingWeights(std::vector<float> weights);

  /** Set / Get whether the displaced detector filter should be disabled */
  itkSetMacro(DisableDisplacedDetectorFilter, bool);
  itkGetMacro(DisableDisplacedDetectorFilter, bool);

protected:
  ADMMTotalVariationConjugateGradientOperator();
  ~ADMMTotalVariationConjugateGradientOperator() override = default;

  /** Checks that inputs are correctly set. */
  void
  VerifyPreconditions() const override;

  /** Does the real work. */
  void
  GenerateData() override;

  /** Member pointers to the filters used internally (for convenience)*/
  BackProjectionFilterPointer    m_BackProjectionFilter;
  ForwardProjectionFilterPointer m_ForwardProjectionFilter;

  typename SubtractFilterType::Pointer          m_SubtractFilter;
  typename MultiplyFilterType::Pointer          m_MultiplyFilter;
  typename MultiplyFilterType::Pointer          m_ZeroMultiplyProjectionFilter;
  typename MultiplyFilterType::Pointer          m_ZeroMultiplyVolumeFilter;
  typename DivergenceFilterType::Pointer        m_DivergenceFilter;
  typename GradientFilterType::Pointer          m_GradientFilter;
  typename DisplacedDetectorFilterType::Pointer m_DisplacedDetectorFilter;
  typename GatingWeightsFilterType::Pointer     m_GatingWeightsFilter;

  ThreeDCircularProjectionGeometry::Pointer m_Geometry;
  float                                     m_Beta;
  bool                                      m_DisableDisplacedDetectorFilter;

  /** Have gating weights been set ? If so, apply them, otherwise ignore
   * the gating weights filter */
  bool               m_IsGated;
  std::vector<float> m_GatingWeights;

  /** When the inputs have the same type, ITK checks whether they occupy the
   * same physical space or not. Obviously they dont, so we have to remove this check
   */
  void
  VerifyInputInformation() const override
  {}

  /** The volume and the projections must have different requested regions
   */
  void
  GenerateInputRequestedRegion() override;
  void
  GenerateOutputInformation() override;
};
} // namespace rtk


#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkADMMTotalVariationConjugateGradientOperator.hxx"
#endif

#endif
