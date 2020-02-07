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

#ifndef rtkWarpSequenceImageFilter_h
#define rtkWarpSequenceImageFilter_h

#include "rtkConstantImageSource.h"

#include <itkExtractImageFilter.h>
#include <itkPasteImageFilter.h>
#include <itkCastImageFilter.h>
#include <itkNearestNeighborInterpolateImageFunction.h>

#ifdef RTK_USE_CUDA
#  include "rtkCudaWarpImageFilter.h"
#  include "rtkCudaForwardWarpImageFilter.h"
#  include "rtkCudaCyclicDeformationImageFilter.h"
#else
#  include <itkWarpImageFilter.h>
#  include "rtkForwardWarpImageFilter.h"
#  include "rtkCyclicDeformationImageFilter.h"
#endif

namespace rtk
{
/** \class WarpSequenceImageFilter
 * \brief Applies an N-D + time Motion Vector Field to an N-D + time sequence of images
 *
 * Most of the work in this filter is performed by the underlying itkWarpImageFilter.
 * The only difference is that this filter manages the last dimension specifically as time.
 *
 * \dot
 * digraph WarpSequenceImageFilter {
 *
 * Input0 [ label="Input 0 (Sequence of images)"];
 * Input0 [shape=Mdiamond];
 * Input1 [label="Input 1 (Sequence of DVFs)"];
 * Input1 [shape=Mdiamond];
 * Output [label="Output (Sequence of images)"];
 * Output [shape=Mdiamond];
 *
 * node [shape=box];
 * Extract [label="itk::ExtractImageFilter (for images)" URL="\ref itk::ExtractImageFilter"];
 * CyclicDeformation [label="rtk::CyclicDeformationImageFilter (for DVFs)"
 *                    URL="\ref rtk::CyclicDeformationImageFilter"];
 * Warp [ label="itk::WarpImageFilter" URL="\ref itk::WarpImageFilter"];
 * Cast [ label="itk::CastImageFilter" URL="\ref itk::CastImageFilter"];
 * Paste [ label="itk::PasteImageFilter" URL="\ref itk::PasteImageFilter"];
 * ConstantSource [ label="rtk::ConstantImageSource" URL="\ref rtk::ConstantImageSource"];
 * BeforePaste [label="", fixedsize="false", width=0, height=0, shape=none];
 * AfterPaste [label="", fixedsize="false", width=0, height=0, shape=none];
 *
 * Input0 -> Extract;
 * Input1 -> CyclicDeformation;
 * Extract -> Warp;
 * CyclicDeformation -> Warp;
 * Warp -> Cast;
 * Cast -> Paste;
 * ConstantSource -> BeforePaste [arrowhead=none];
 * BeforePaste -> Paste;
 * Paste -> AfterPaste [arrowhead=none];
 * AfterPaste -> BeforePaste [style=dashed, constraint=false];
 * AfterPaste -> Output [style=dashed];
 * }
 * \enddot
 *
 * \test rtkfourdroostertest.cxx
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
class WarpSequenceImageFilter : public itk::ImageToImageFilter<TImageSequence, TImageSequence>
{
public:
  ITK_DISALLOW_COPY_AND_ASSIGN(WarpSequenceImageFilter);

  /** Standard class type alias. */
  using Self = WarpSequenceImageFilter;
  using Superclass = itk::ImageToImageFilter<TImageSequence, TImageSequence>;
  using Pointer = itk::SmartPointer<Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** SFINAE type alias, depending on whether a CUDA image is used. */
  using CPUImageType = typename itk::Image<typename TImage::PixelType, TImage::ImageDimension>;
  using CPUWarpFilterType = typename itk::WarpImageFilter<TImage, TImage, TDVFImage>;
#ifdef RTK_USE_CUDA
  typedef
    typename std::conditional<std::is_same<TImage, CPUImageType>::value, CPUWarpFilterType, CudaWarpImageFilter>::type
                                                                            WarpFilterType;
  typedef typename std::conditional<std::is_same<TImage, CPUImageType>::value,
                                    ForwardWarpImageFilter<TImage, TImage, TDVFImage>,
                                    CudaForwardWarpImageFilter>::type       ForwardWarpFilterType;
  typedef typename std::conditional<std::is_same<TImage, CPUImageType>::value,
                                    CyclicDeformationImageFilter<TDVFImageSequence, TDVFImage>,
                                    CudaCyclicDeformationImageFilter>::type CudaCyclicDeformationImageFilterType;
#else
  using WarpFilterType = CPUWarpFilterType;
  using ForwardWarpFilterType = ForwardWarpImageFilter<TImage, TImage, TDVFImage>;
  using CudaCyclicDeformationImageFilterType = CyclicDeformationImageFilter<TDVFImageSequence, TDVFImage>;
#endif

  /** Run-time type information (and related methods). */
  itkTypeMacro(WarpSequenceImageFilter, IterativeConeBeamReconstructionFilter);

  /** Set the motion vector field used in input 1 */
  void
  SetDisplacementField(const TDVFImageSequence * DVFs);

  /** Get the motion vector field used in input 1 */
  typename TDVFImageSequence::Pointer
  GetDisplacementField();

  /** Set/Get for m_ForwardWarp */
  itkGetMacro(ForwardWarp, bool);
  itkSetMacro(ForwardWarp, bool);

  /** Phase shift to simulate phase estimation errors */
  itkSetMacro(PhaseShift, float);
  itkGetMacro(PhaseShift, float);

  /** Information for the CUDA warp filter, to avoid using RTTI */
  itkSetMacro(UseNearestNeighborInterpolationInWarping, bool);
  itkGetMacro(UseNearestNeighborInterpolationInWarping, bool);

  /** Set and Get for the UseCudaCyclicDeformation variable */
  itkSetMacro(UseCudaCyclicDeformation, bool);
  itkGetMacro(UseCudaCyclicDeformation, bool);

  /** Typedefs of internal filters */
  using LinearInterpolatorType = itk::LinearInterpolateImageFunction<TImage, double>;
  using NearestNeighborInterpolatorType = itk::NearestNeighborInterpolateImageFunction<TImage, double>;
  using ExtractFilterType = itk::ExtractImageFilter<TImageSequence, TImage>;
  using DVFInterpolatorType = rtk::CyclicDeformationImageFilter<TDVFImageSequence, TDVFImage>;
  using PasteFilterType = itk::PasteImageFilter<TImageSequence, TImageSequence>;
  using CastFilterType = itk::CastImageFilter<TImage, TImageSequence>;
  using ConstantImageSourceType = rtk::ConstantImageSource<TImageSequence>;

protected:
  WarpSequenceImageFilter();
  ~WarpSequenceImageFilter() override = default;

  /** Does the real work. */
  void
  GenerateData() override;

  /** Member pointers to the filters used internally (for convenience)*/
  typename CPUWarpFilterType::Pointer       m_WarpFilter;
  typename ExtractFilterType::Pointer       m_ExtractFilter;
  typename DVFInterpolatorType::Pointer     m_DVFInterpolatorFilter;
  typename PasteFilterType::Pointer         m_PasteFilter;
  typename CastFilterType::Pointer          m_CastFilter;
  typename ConstantImageSourceType::Pointer m_ConstantSource;

  /** Extraction regions for both extract filters */
  typename TImageSequence::RegionType m_ExtractAndPasteRegion;

  /** Perform a forward warping (using splat) instead of the standard backward warping */
  bool  m_ForwardWarp;
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
  GenerateOutputInformation() override;
  void
  GenerateInputRequestedRegion() override;

  bool m_UseNearestNeighborInterpolationInWarping; // Default is false, linear interpolation is used instead
  bool m_UseCudaCyclicDeformation;
};
} // namespace rtk


#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkWarpSequenceImageFilter.hxx"
#endif

#endif
