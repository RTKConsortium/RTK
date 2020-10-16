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

#ifndef rtkMotionCompensatedFourDConjugateGradientConeBeamReconstructionFilter_h
#define rtkMotionCompensatedFourDConjugateGradientConeBeamReconstructionFilter_h

#include "rtkFourDConjugateGradientConeBeamReconstructionFilter.h"
#include "rtkMotionCompensatedFourDReconstructionConjugateGradientOperator.h"
#include "rtkWarpProjectionStackToFourDImageFilter.h"
#ifdef RTK_USE_CUDA
#  include "rtkCudaWarpForwardProjectionImageFilter.h"
#  include "rtkCudaWarpBackProjectionImageFilter.h"
#endif

namespace rtk
{
/** \class MotionCompensatedFourDConjugateGradientConeBeamReconstructionFilter
 * \brief Implements motion compensated 4D reconstruction by conjugate gradient
 *
 * \dot
 * digraph MotionCompensatedFourDConjugateGradientConeBeamReconstructionFilter {
 *
 * Input0 [label="Input 0 (Input: 4D sequence of volumes)"];
 * Input0 [shape=Mdiamond];
 * Input1 [label="Input 1 (Projections)"];
 * Input1 [shape=Mdiamond];
 * Input2 [label="Input 2 (4D Sequence of DVFs)"];
 * Input2 [shape=Mdiamond];
 * Output [label="Output (Reconstruction: 4D sequence of volumes)"];
 * Output [shape=Mdiamond];
 *
 * node [shape=box];
 * AfterInput0 [label="", fixedsize="false", width=0, height=0, shape=none];
 * ConjugateGradient [ label="rtk::ConjugateGradientImageFilter" URL="\ref rtk::ConjugateGradientImageFilter"];
 * PSTFD [ label="rtk::WarpProjectionStackToFourDImageFilter" URL="\ref rtk::WarpProjectionStackToFourDImageFilter"];
 *
 * Input0 -> AfterInput0 [arrowhead=none];
 * AfterInput0 -> ConjugateGradient;
 * Input0 -> PSTFD;
 * Input1 -> PSTFD;
 * Input1 -> PSTFD;
 * PSTFD -> ConjugateGradient;
 * ConjugateGradient -> Output;
 * }
 * \enddot
 *
 * \test rtkmotioncompensatedfourdconjugategradienttest.cxx
 *
 * \author Cyril Mory
 *
 * \ingroup RTK ReconstructionAlgorithm
 */

template <typename VolumeSeriesType, typename ProjectionStackType>
class ITK_EXPORT MotionCompensatedFourDConjugateGradientConeBeamReconstructionFilter
  : public rtk::FourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
{
public:
#if ITK_VERSION_MAJOR == 5 && ITK_VERSION_MINOR == 1
  ITK_DISALLOW_COPY_AND_ASSIGN(MotionCompensatedFourDConjugateGradientConeBeamReconstructionFilter);
#else
  ITK_DISALLOW_COPY_AND_MOVE(MotionCompensatedFourDConjugateGradientConeBeamReconstructionFilter);
#endif

  /** Standard class type alias. */
  using Self = MotionCompensatedFourDConjugateGradientConeBeamReconstructionFilter;
  using Superclass = FourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Some convenient type alias. */
  using InputImageType = VolumeSeriesType;
  using OutputImageType = VolumeSeriesType;
  using VolumeType = ProjectionStackType;
  using VectorForDVF = itk::CovariantVector<typename VolumeSeriesType::ValueType, VolumeSeriesType::ImageDimension - 1>;

  using ForwardProjectionType = typename Superclass::ForwardProjectionType;
  using BackProjectionType = typename Superclass::BackProjectionType;

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
#else
  using DVFSequenceImageType = itk::Image<VectorForDVF, VolumeSeriesType::ImageDimension>;
  using DVFImageType = itk::Image<VectorForDVF, VolumeSeriesType::ImageDimension - 1>;
#endif

  /** Typedefs of each subfilter of this composite filter */

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(MotionCompensatedFourDConjugateGradientConeBeamReconstructionFilter,
               FourDConjugateGradientConeBeamReconstructionFilter);

  /** Neither the Forward nor the Back projection filters can be set by the user */
  void
  SetForwardProjectionFilter(ForwardProjectionType itkNotUsed(_arg)) override
  {
    itkExceptionMacro(<< "ForwardProjection cannot be changed");
  }
  void
  SetBackProjectionFilter(BackProjectionType itkNotUsed(_arg)) override
  {
    itkExceptionMacro(<< "BackProjection cannot be changed");
  }

  /** The ND + time motion vector field */
  void
  SetDisplacementField(const DVFSequenceImageType * DisplacementField);
  void
  SetInverseDisplacementField(const DVFSequenceImageType * InverseDisplacementField);
  typename DVFSequenceImageType::ConstPointer
  GetDisplacementField();
  typename DVFSequenceImageType::ConstPointer
  GetInverseDisplacementField();

  /** Set the vector containing the signal in the sub-filters */
  void
  SetSignal(const std::vector<double> signal) override;

  // Sub filters type alias
  using MCProjStackToFourDType = rtk::WarpProjectionStackToFourDImageFilter<VolumeSeriesType, ProjectionStackType>;
  using MCCGOperatorType =
    rtk::MotionCompensatedFourDReconstructionConjugateGradientOperator<VolumeSeriesType, ProjectionStackType>;

  /** Set and Get for the UseCudaCyclicDeformation variable */
  itkSetMacro(UseCudaCyclicDeformation, bool);
  itkGetMacro(UseCudaCyclicDeformation, bool);

protected:
  MotionCompensatedFourDConjugateGradientConeBeamReconstructionFilter();
  ~MotionCompensatedFourDConjugateGradientConeBeamReconstructionFilter() override = default;

  void
  GenerateOutputInformation() override;
  void
  GenerateInputRequestedRegion() override;

  bool m_UseCudaCyclicDeformation;

}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkMotionCompensatedFourDConjugateGradientConeBeamReconstructionFilter.hxx"
#endif

#endif
