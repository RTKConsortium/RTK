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

#ifndef rtkUnwarpSequenceImageFilter_h
#define rtkUnwarpSequenceImageFilter_h

#include "rtkConjugateGradientImageFilter.h"
#include "rtkUnwarpSequenceConjugateGradientOperator.h"
#include "rtkWarpSequenceImageFilter.h"
#include "rtkConstantImageSource.h"

#ifdef RTK_USE_CUDA
#  include "rtkCudaConjugateGradientImageFilter.h"
#  include "rtkCudaConstantVolumeSeriesSource.h"
#endif

namespace rtk
{
/** \class UnwarpSequenceImageFilter
 * \brief Finds the image sequence that, once warped, equals the input image sequence.
 *
 * This filter attempts to invert a deformation by Conjugate Gradient optimization.
 *
 * \dot
 * digraph UnwarpSequenceImageFilter {
 *
 * Input0 [ label="Input 0 (4D volume sequence)"];
 * Input0 [shape=Mdiamond];
 * Input1 [label="Input 1 (4D DVF)"];
 * Input1 [shape=Mdiamond];
 * Output [label="Output (4D volume sequence)"];
 * Output [shape=Mdiamond];
 *
 * node [shape=box];
 * ConstantSource [label="rtk::ConstantImageSource (4D volume sequence)" URL="\ref rtk::WarpSequenceImageFilter"];
 * WarpSequenceForward [label="rtk::WarpSequenceImageFilter (forward mapping)"
 *                      URL="\ref rtk::WarpSequenceImageFilter"];
 * ConjugateGradient[ label="rtk::ConjugateGradientImageFilter" URL="\ref rtk::ConjugateGradientImageFilter"];
 * CyclicDeformation [label="rtk::CyclicDeformationImageFilter (for DVFs)"
 *                    URL="\ref rtk::CyclicDeformationImageFilter"];
 *
 * Input0 -> WarpSequenceForward;
 * Input1 -> CyclicDeformation;
 * CyclicDeformation -> WarpSequenceForward;
 * ConstantSource -> ConjugateGradient;
 * WarpSequenceForward -> ConjugateGradient;
 * ConjugateGradient -> Output;
 * }
 * \enddot
 *
 * \test rtkunwarpsequencetest.cxx
 *
 * \author Cyril Mory
 *
 * \ingroup RTK ReconstructionAlgorithm
 */

template <typename TImageSequence,
          typename TDVFImageSequence =
            itk::Image<itk::CovariantVector<typename TImageSequence::ValueType, TImageSequence::ImageDimension - 1>,
                       TImageSequence::ImageDimension>,
          typename TImage = itk::Image<typename TImageSequence::ValueType, TImageSequence::ImageDimension - 1>,
          typename TDVFImage =
            itk::Image<itk::CovariantVector<typename TImageSequence::ValueType, TImageSequence::ImageDimension - 1>,
                       TImageSequence::ImageDimension - 1>>
class ITK_TEMPLATE_EXPORT UnwarpSequenceImageFilter : public itk::ImageToImageFilter<TImageSequence, TImageSequence>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(UnwarpSequenceImageFilter);

  /** Standard class type alias. */
  using Self = UnwarpSequenceImageFilter;
  using Superclass = itk::ImageToImageFilter<TImageSequence, TImageSequence>;
  using Pointer = itk::SmartPointer<Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(UnwarpSequenceImageFilter);

  using CGOperatorFilterType =
    UnwarpSequenceConjugateGradientOperator<TImageSequence, TDVFImageSequence, TImage, TDVFImage>;
  using WarpForwardFilterType = WarpSequenceImageFilter<TImageSequence, TDVFImageSequence, TImage, TDVFImage>;
  using ConjugateGradientFilterType = ConjugateGradientImageFilter<TImageSequence>;

  /** SFINAE type alias, depending on whether a CUDA image is used. */
  using CPUImageSequence = typename itk::Image<typename TImageSequence::PixelType, TImageSequence::ImageDimension>;
#ifdef RTK_USE_CUDA
  using ConstantSourceType = typename std::conditional_t<std::is_same_v<TImageSequence, CPUImageSequence>,
                                                         ConstantImageSource<TImageSequence>,
                                                         CudaConstantVolumeSeriesSource>;
  using CudaConjugateGradientType = typename std::conditional_t<std::is_same_v<TImageSequence, CPUImageSequence>,
                                                                ConjugateGradientFilterType,
                                                                CudaConjugateGradientImageFilter<TImageSequence>>;
#else
  using ConstantSourceType = ConstantImageSource<TImageSequence>;
  using CudaConjugateGradientType = ConjugateGradientFilterType;
#endif

  /** Set the motion vector field used in input 1 */
  void
  SetDisplacementField(const TDVFImageSequence * DVFs);

  /** Get the motion vector field used in input 1 */
  typename TDVFImageSequence::Pointer
  GetDisplacementField();

  /** Number of conjugate gradient iterations */
  itkSetMacro(NumberOfIterations, float);
  itkGetMacro(NumberOfIterations, float);

  /** Phase shift to simulate phase estimation errors */
  itkSetMacro(PhaseShift, float);
  itkGetMacro(PhaseShift, float);

  itkSetMacro(UseNearestNeighborInterpolationInWarping, bool);
  itkGetMacro(UseNearestNeighborInterpolationInWarping, bool);

  itkSetMacro(CudaConjugateGradient, bool);
  itkGetMacro(CudaConjugateGradient, bool);

  /** Set and Get for the UseCudaCyclicDeformation variable */
  itkSetMacro(UseCudaCyclicDeformation, bool);
  itkGetMacro(UseCudaCyclicDeformation, bool);

protected:
  UnwarpSequenceImageFilter();
  ~UnwarpSequenceImageFilter() override = default;

  /** Does the real work. */
  void
  GenerateData() override;

  /** Member pointers to the filters used internally (for convenience)*/
  typename ConjugateGradientFilterType::Pointer m_ConjugateGradientFilter;
  typename CGOperatorFilterType::Pointer        m_CGOperator;
  typename WarpForwardFilterType::Pointer       m_WarpForwardFilter;
  typename ConstantSourceType::Pointer          m_ConstantSource;

  /** Member variables */
  float m_PhaseShift;

  /** The inputs of this filter have the same type (float, 3) but not the same meaning
   * It is normal that they do not occupy the same physical space. Therefore this check
   * must be removed */
  void
  VerifyInputInformation() const override
  {}

  /** The volume and the projections must have different requested regions
   */
  void
  GenerateInputRequestedRegion() override;
  void
  GenerateOutputInformation() override;

  bool m_UseNearestNeighborInterpolationInWarping; // Default is false, linear interpolation is used instead
  bool m_CudaConjugateGradient;
  bool m_UseCudaCyclicDeformation;

private:
  unsigned int m_NumberOfIterations;
};
} // namespace rtk


#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkUnwarpSequenceImageFilter.hxx"
#endif

#endif
