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

#ifndef rtkReconstructionConjugateGradientOperator_h
#define rtkReconstructionConjugateGradientOperator_h

#include <itkMultiplyImageFilter.h>
#include <itkAddImageFilter.h>

#include "rtkConstantImageSource.h"

#include "rtkConjugateGradientOperator.h"
#include "rtkBackProjectionImageFilter.h"
#include "rtkForwardProjectionImageFilter.h"

#include "rtkThreeDCircularProjectionGeometry.h"
#include "rtkLaplacianImageFilter.h"
#include "rtkBlockDiagonalMatrixVectorMultiplyImageFilter.h"

#ifdef RTK_USE_CUDA
#  include "rtkCudaConstantVolumeSource.h"
#  include "rtkCudaLaplacianImageFilter.h"
#endif

namespace rtk
{

/** \class ReconstructionConjugateGradientOperator
 * \brief Implements the operator A used in conjugate gradient reconstruction
 *
 * This filter implements the operator A used in the conjugate gradient reconstruction method,
 * which attempts to find the f that minimizes
 * || sqrt(D) (Rf -p) ||_2^2 + gamma || grad f ||_2^2 + Tikhonov || f ||_2^2,
 * with R the forward projection operator,
 * p the measured projections, and D the displaced detector weighting operator.
 *
 * With gamma=0, this it is similar to the ART and SART methods. The difference lies
 * in the algorithm employed to minimize this cost function. ART uses the
 * Kaczmarz method (projects and back projects one ray at a time),
 * SART the block-Kaczmarz method (projects and back projects one projection
 * at a time), and ConjugateGradient a conjugate gradient method
 * (projects and back projects all projections together).
 *
 * This filter takes in input f and outputs R_t D R f + gamma Laplacian f + Tikhonov f
 *
 * \dot
 * digraph ReconstructionConjugateGradientOperator {
 *
 * Input0 [ label="Input 0 (Volume)"];
 * Input0 [shape=Mdiamond];
 * Input1 [label="Input 1 (Projections)"];
 * Input1 [shape=Mdiamond];
 * Input2 [label="Input 2 (Weights)"];
 * Input2 [shape=Mdiamond];
 * Input3 [label="Input Support Mask"];
 * Input3 [shape=Mdiamond];
 * Output [label="Output (Volume)"];
 * Output [shape=Mdiamond];
 *
 * node [shape=box];
 * ConstantVolumeSource [label="rtk::ConstantImageSource" URL="\ref rtk::ConstantImageSource"];
 * ConstantProjectionsSource [label="rtk::ConstantImageSource" URL="\ref rtk::ConstantImageSource"];
 * BackProjection [ label="rtk::BackProjectionImageFilter" URL="\ref rtk::BackProjectionImageFilter"];
 * ForwardProjection [ label="rtk::ForwardProjectionImageFilter" URL="\ref rtk::ForwardProjectionImageFilter"];
 * Multiply [ label="itk::MultiplyImageFilter" URL="\ref itk::MultiplyImageFilter"];
 * MultiplyInput [ label="itk::MultiplyImageFilter" URL="\ref itk::MultiplyImageFilter"];
 * MultiplyOutput [ label="itk::MultiplyImageFilter" URL="\ref itk::MultiplyImageFilter"];
 * Laplacian [ label="rtk::LaplacianImageFilter" URL="\ref rtk::LaplacianImageFilter"];
 * MultiplyLaplacian [ label="itk::MultiplyImageFilter (by gamma)" URL="\ref itk::MultiplyImageFilter"];
 * MultiplyTikhonov [ label="itk::MultiplyImageFilter (by Tikhonov parameter)" URL="\ref itk::MultiplyImageFilter"];
 * AddLaplacian [ label="itk::AddImageFilter" URL="\ref itk::AddImageFilter"];
 * AddTikhonov [ label="itk::AddImageFilter" URL="\ref itk::AddImageFilter"];
 *
 * Input0 -> MultiplyInput;
 * MultiplyInput -> MultiplyTikhonov;
 * Input3 -> MultiplyInput;
 * MultiplyInput -> ForwardProjection;
 * ConstantProjectionsSource -> ForwardProjection;
 * ConstantVolumeSource -> BackProjection;
 * ForwardProjection -> Multiply;
 * Input2 -> Multiply;
 * Multiply -> BackProjection;
 * BackProjection -> AddLaplacian;
 * Input3 -> MultiplyOutput;
 * MultiplyInput -> Laplacian;
 * Laplacian -> MultiplyLaplacian;
 * MultiplyLaplacian -> AddLaplacian;
 * AddLaplacian -> AddTikhonov;
 * MultiplyTikhonov -> AddTikhonov;
 * AddTikhonov -> MultiplyOutput;
 * MultiplyOutput -> Output;
 * }
 * \enddot
 *
 * \test rtkconjugategradienttest.cxx
 *
 * \author Cyril Mory
 *
 * \ingroup RTK ReconstructionAlgorithm
 */

template <typename TOutputImage, typename TSingleComponentImage = TOutputImage, typename TWeightsImage = TOutputImage>
class ReconstructionConjugateGradientOperator : public ConjugateGradientOperator<TOutputImage>
{
public:
  ITK_DISALLOW_COPY_AND_ASSIGN(ReconstructionConjugateGradientOperator);

  /** Standard class type alias. */
  using Self = ReconstructionConjugateGradientOperator;
  using Superclass = ConjugateGradientOperator<TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;
#ifdef RTK_USE_CUDA
  using GradientImageType =
    itk::CudaImage<itk::CovariantVector<typename TOutputImage::PixelType, TOutputImage::ImageDimension>,
                   TOutputImage::ImageDimension>;
#else
  using GradientImageType =
    itk::Image<itk::CovariantVector<typename TOutputImage::PixelType, TOutputImage::ImageDimension>,
               TOutputImage::ImageDimension>;
#endif

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Setters for the inputs */
  void
  SetInputVolume(const TOutputImage * vol);
  void
  SetInputProjectionStack(const TOutputImage * projs);
  void
  SetInputWeights(const TWeightsImage * weights);

  /** Run-time type information (and related methods). */
  itkTypeMacro(rtkReconstructionConjugateGradientOperator, ConjugateGradientOperator);

  using BackProjectionFilterType = rtk::BackProjectionImageFilter<TOutputImage, TOutputImage>;
  using BackProjectionFilterPointer = typename BackProjectionFilterType::Pointer;

  using ForwardProjectionFilterType = rtk::ForwardProjectionImageFilter<TOutputImage, TOutputImage>;
  using ForwardProjectionFilterPointer = typename ForwardProjectionFilterType::Pointer;

  using ConstantSourceType = rtk::ConstantImageSource<TOutputImage>;
  using MultiplyFilterType = itk::MultiplyImageFilter<TOutputImage, TSingleComponentImage>;
  using AddFilterType = itk::AddImageFilter<TOutputImage>;

  // If TOutputImage is an itk::Image of floats or double, so are the weights, and a simple Multiply filter is required
  // If TOutputImage is an itk::Image of itk::Vector<float (or double)>, a BlockDiagonalMatrixVectorMultiply filter
  // is needed. Thus the meta-programming construct
  using MatrixVectorMultiplyFilterType = rtk::BlockDiagonalMatrixVectorMultiplyImageFilter<TOutputImage, TWeightsImage>;
  using PlainMultiplyFilterType = itk::MultiplyImageFilter<TOutputImage, TOutputImage, TOutputImage>;
  typedef typename std::conditional<std::is_same<TSingleComponentImage, TOutputImage>::value,
                                    PlainMultiplyFilterType,
                                    MatrixVectorMultiplyFilterType>::type MultiplyWithWeightsFilterType;

  using OutputImagePointer = typename TOutputImage::Pointer;

  /** Set the backprojection filter*/
  void
  SetBackProjectionFilter(const BackProjectionFilterPointer _arg);

  /** Set the forward projection filter*/
  void
  SetForwardProjectionFilter(const ForwardProjectionFilterPointer _arg);

  /** Set the support mask, if any, for support constraint in reconstruction */
  void
  SetSupportMask(const TSingleComponentImage * SupportMask);
  typename TSingleComponentImage::ConstPointer
  GetSupportMask();

  /** Set the geometry of both m_BackProjectionFilter and m_ForwardProjectionFilter */
  itkSetConstObjectMacro(Geometry, ThreeDCircularProjectionGeometry);

  /** Perform laplacian-based and/or Tikhonov regularization during
   *  reconstruction (gamma is the strength of laplacian the regularization) */
  itkSetMacro(Gamma, float);
  itkGetMacro(Gamma, float);
  itkSetMacro(Tikhonov, float);
  itkGetMacro(Tikhonov, float);

protected:
  ReconstructionConjugateGradientOperator();
  ~ReconstructionConjugateGradientOperator() override = default;

  /** Does the real work. */
  void
  GenerateData() override;

  template <typename ImageType>
  typename std::enable_if<std::is_same<TSingleComponentImage, ImageType>::value, ImageType>::type::Pointer
  ConnectGradientRegularization();

  template <typename ImageType>
  typename std::enable_if<!std::is_same<TSingleComponentImage, ImageType>::value, ImageType>::type::Pointer
  ConnectGradientRegularization();

  /** Member pointers to the filters used internally (for convenience)*/
  BackProjectionFilterPointer    m_BackProjectionFilter;
  ForwardProjectionFilterPointer m_ForwardProjectionFilter;

  typename ConstantSourceType::Pointer                                  m_ConstantProjectionsSource;
  typename ConstantSourceType::Pointer                                  m_ConstantVolumeSource;
  typename MultiplyFilterType::Pointer                                  m_MultiplyOutputVolumeFilter;
  typename MultiplyFilterType::Pointer                                  m_MultiplyInputVolumeFilter;
  typename MultiplyFilterType::Pointer                                  m_MultiplyLaplacianFilter;
  typename MultiplyFilterType::Pointer                                  m_MultiplyTikhonovFilter;
  typename AddFilterType::Pointer                                       m_AddLaplacianFilter;
  typename AddFilterType::Pointer                                       m_AddTikhonovFilter;
  typename itk::ImageToImageFilter<TOutputImage, TOutputImage>::Pointer m_LaplacianFilter;
  typename MultiplyWithWeightsFilterType::Pointer                       m_MultiplyWithWeightsFilter;

  /** Member attributes */
  rtk::ThreeDCircularProjectionGeometry::ConstPointer m_Geometry{ nullptr };
  float                                               m_Gamma{ 0 };    // Strength of the laplacian regularization
  float                                               m_Tikhonov{ 0 }; // Strength of the Tikhonov regularization

  /** Pointers to intermediate images, used to simplify complex branching */
  typename TOutputImage::Pointer m_FloatingInputPointer, m_FloatingOutputPointer;

  /** When the inputs have the same type, ITK checks whether they occupy the
   * same physical space or not. Obviously they dont, so we have to remove this check */
  void
  VerifyInputInformation() const override
  {}

  /** The volume and the projections must have different requested regions */
  void
  GenerateInputRequestedRegion() override;
  void
  GenerateOutputInformation() override;

  /** Getters for the inputs */
  typename TOutputImage::ConstPointer
  GetInputVolume();
  typename TOutputImage::ConstPointer
  GetInputProjectionStack();
  typename TWeightsImage::ConstPointer
  GetInputWeights();
};
} // namespace rtk


#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkReconstructionConjugateGradientOperator.hxx"
#endif

#endif
