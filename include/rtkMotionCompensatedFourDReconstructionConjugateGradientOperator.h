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
#ifndef rtkMotionCompensatedFourDReconstructionConjugateGradientOperator_h
#define rtkMotionCompensatedFourDReconstructionConjugateGradientOperator_h

#include "rtkFourDReconstructionConjugateGradientOperator.h"
#include "rtkCyclicDeformationImageFilter.h"
#include "rtkJosephForwardProjectionImageFilter.h"

#ifdef RTK_USE_CUDA
#  include "rtkCudaWarpForwardProjectionImageFilter.h"
#  include "rtkCudaWarpBackProjectionImageFilter.h"
#  include "rtkCudaCyclicDeformationImageFilter.h"
#endif

namespace rtk
{
/** \class MotionCompensatedFourDReconstructionConjugateGradientOperator
 * \brief Like FourDReconstructionConjugateGradientOperator, but motion-compensated
 *
 * \dot
 * digraph MotionCompensatedFourDReconstructionConjugateGradientOperator {
 *
 * Input0 [ label="Input 0 (Input: 4D sequence of volumes)"];
 * Input0 [shape=Mdiamond];
 * Input1 [label="Input 1 (Projections)"];
 * Input1 [shape=Mdiamond];
 * Input2 [label="Input 2 (Projection weights)"];
 * Input2 [shape=Mdiamond];
 * Input3 [label="Input 3 (4D Sequence of DVFs)"];
 * Input3 [shape=Mdiamond];
 * Input4 [label="Input 4 (4D Sequence of inverse DVFs)"];
 * Input4 [shape=Mdiamond];
 * Output [label="Output (Reconstruction: 4D sequence of volumes)"];
 * Output [shape=Mdiamond];
 *
 * node [shape=box];
 * SourceVol [ label="rtk::ConstantImageSource (volume)" URL="\ref rtk::ConstantImageSource"];
 * SourceVol2 [ label="rtk::ConstantImageSource (volume)" URL="\ref rtk::ConstantImageSource"];
 * SourceProj [ label="rtk::ConstantImageSource (projections)" URL="\ref rtk::ConstantImageSource"];
 * Source4D [ label="rtk::ConstantImageSource (4D)" URL="\ref rtk::ConstantImageSource"];
 * CyclicDeformation [label="rtk::CyclicDeformationImageFilter (for DVFs)"
 *                    URL="\ref rtk::CyclicDeformationImageFilter"];
 * CyclicDeformationInv [label="rtk::CyclicDeformationImageFilter (for inverse DVFs)"
 *                       URL="\ref rtk::CyclicDeformationImageFilter"];
 * ForwardProj [ label="rtk::CudaWarpForwardProjectionImageFilter"
 *               URL="\ref rtk::CudaWarpForwardProjectionImageFilter"];
 * Interpolation [ label="InterpolatorWithKnownWeightsImageFilter"
 *                 URL="\ref rtk::InterpolatorWithKnownWeightsImageFilter"];
 * BackProj [ label="rtk::CudaWarpBackProjectionImageFilter" URL="\ref rtk::CudaWarpBackProjectionImageFilter"];
 * Splat [ label="rtk::SplatWithKnownWeightsImageFilter" URL="\ref rtk::SplatWithKnownWeightsImageFilter"];
 * AfterSplat [label="", fixedsize="false", width=0, height=0, shape=none];
 * AfterInput0 [label="", fixedsize="false", width=0, height=0, shape=none];
 * AfterSource4D [label="", fixedsize="false", width=0, height=0, shape=none];
 *
 * Input3 -> CyclicDeformation;
 * Input4 -> CyclicDeformationInv;
 * CyclicDeformationInv -> ForwardProj;
 * CyclicDeformation -> BackProj;
 * Input0 -> Interpolation;
 * SourceVol -> Interpolation;
 * Interpolation -> ForwardProj;
 * SourceVol2 -> BackProj;
 * ForwardProj -> Multiply;
 * Input2 -> Multiply;
 * Multiply -> BackProj;
 * BackProj -> Splat;
 * Splat -> AfterSplat[arrowhead=none];
 * AfterSplat -> Output;
 * AfterSplat -> AfterSource4D[style=dashed, constraint=false];
 * Source4D -> AfterSource4D[arrowhead=none];
 * AfterSource4D -> Splat;
 * SourceProj -> ForwardProj;
 * }
 * \enddot
 *
 * \test rtkfourdconjugategradienttest.cxx
 *
 * \author Cyril Mory
 *
 * \ingroup RTK ReconstructionAlgorithm
 */

template <typename VolumeSeriesType, typename ProjectionStackType>
class MotionCompensatedFourDReconstructionConjugateGradientOperator
  : public FourDReconstructionConjugateGradientOperator<VolumeSeriesType, ProjectionStackType>
{
public:
  ITK_DISALLOW_COPY_AND_ASSIGN(MotionCompensatedFourDReconstructionConjugateGradientOperator);

  /** Standard class type alias. */
  using Self = MotionCompensatedFourDReconstructionConjugateGradientOperator;
  using Superclass = FourDReconstructionConjugateGradientOperator<VolumeSeriesType, ProjectionStackType>;
  using Pointer = itk::SmartPointer<Self>;

  /** Convenient type alias */
  using VolumeType = ProjectionStackType;
  using VectorForDVF = itk::CovariantVector<typename VolumeSeriesType::ValueType, VolumeSeriesType::ImageDimension - 1>;

  /** SFINAE type alias, depending on whether a CUDA image is used. */
  using CPUVolumeSeriesType =
    typename itk::Image<typename VolumeSeriesType::PixelType, VolumeSeriesType::ImageDimension>;
#ifdef RTK_USE_CUDA
  typedef typename std::conditional<std::is_same<VolumeSeriesType, CPUVolumeSeriesType>::value,
                                    itk::Image<VectorForDVF, VolumeSeriesType::ImageDimension>,
                                    itk::CudaImage<VectorForDVF, VolumeSeriesType::ImageDimension>>::type
    DVFSequenceImageType;
  typedef
    typename std::conditional<std::is_same<VolumeSeriesType, CPUVolumeSeriesType>::value,
                              itk::Image<VectorForDVF, VolumeSeriesType::ImageDimension - 1>,
                              itk::CudaImage<VectorForDVF, VolumeSeriesType::ImageDimension - 1>>::type DVFImageType;
  typedef typename std::conditional<std::is_same<VolumeSeriesType, CPUVolumeSeriesType>::value,
                                    JosephForwardProjectionImageFilter<ProjectionStackType, ProjectionStackType>,
                                    CudaWarpForwardProjectionImageFilter>::type WarpForwardProjectionImageFilterType;
  typedef typename std::conditional<std::is_same<VolumeSeriesType, CPUVolumeSeriesType>::value,
                                    BackProjectionImageFilter<VolumeType, VolumeType>,
                                    CudaWarpBackProjectionImageFilter>::type    WarpBackProjectionImageFilterType;
#else
  using DVFSequenceImageType = itk::Image<VectorForDVF, VolumeSeriesType::ImageDimension>;
  using DVFImageType = itk::Image<VectorForDVF, VolumeSeriesType::ImageDimension - 1>;
  using WarpForwardProjectionImageFilterType =
    JosephForwardProjectionImageFilter<ProjectionStackType, ProjectionStackType>;
  using WarpBackProjectionImageFilterType = BackProjectionImageFilter<VolumeType, VolumeType>;
#endif
  using CPUDVFInterpolatorType = CyclicDeformationImageFilter<DVFSequenceImageType, DVFImageType>;
#ifdef RTK_USE_CUDA
  typedef typename std::conditional<std::is_same<VolumeSeriesType, CPUVolumeSeriesType>::value,
                                    CPUDVFInterpolatorType,
                                    CudaCyclicDeformationImageFilter>::type CudaCyclicDeformationImageFilterType;
#else
  using CudaCyclicDeformationImageFilterType = CPUDVFInterpolatorType;
#endif

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(MotionCompensatedFourDReconstructionConjugateGradientOperator,
               FourDReconstructionConjugateGradientOperator);

  /** The forward and back projection filters cannot be set by the user */
  void
  SetForwardProjectionFilter(const typename Superclass::ForwardProjectionFilterType::Pointer itkNotUsed(_arg))
  {
    itkExceptionMacro(<< "ForwardProjection cannot be changed");
  }
  void
  SetBackProjectionFilter(const typename Superclass::BackProjectionFilterType::Pointer itkNotUsed(_arg))
  {
    itkExceptionMacro(<< "BackProjection cannot be changed");
  }

  /** The ND + time motion vector field */
  void
  SetDisplacementField(const DVFSequenceImageType * DVFs);
  void
  SetInverseDisplacementField(const DVFSequenceImageType * DVFs);
  typename DVFSequenceImageType::ConstPointer
  GetInverseDisplacementField();
  typename DVFSequenceImageType::ConstPointer
  GetDisplacementField();

  /** Set the vector containing the signal in the sub-filters */
  void
  SetSignal(const std::vector<double> signal) override;

  /** Set and Get for the UseCudaCyclicDeformation variable */
  itkSetMacro(UseCudaCyclicDeformation, bool);
  itkGetMacro(UseCudaCyclicDeformation, bool);

protected:
  MotionCompensatedFourDReconstructionConjugateGradientOperator();
  ~MotionCompensatedFourDReconstructionConjugateGradientOperator() override = default;

  /** Builds the pipeline and computes output information */
  void
  GenerateOutputInformation() override;

  /** The inputs should not be in the same space so there is nothing to verify */
  void
  VerifyInputInformation() const override
  {}

  /** Does the real work. */
  void
  GenerateData() override;

  /** Member pointers to the filters used internally (for convenience)*/
  typename CPUDVFInterpolatorType::Pointer m_DVFInterpolatorFilter;
  typename CPUDVFInterpolatorType::Pointer m_InverseDVFInterpolatorFilter;
  std::vector<double>                      m_Signal;
  bool                                     m_UseCudaCyclicDeformation;
};
} // namespace rtk


#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkMotionCompensatedFourDReconstructionConjugateGradientOperator.hxx"
#endif

#endif
