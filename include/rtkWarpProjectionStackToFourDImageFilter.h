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
#ifndef rtkWarpProjectionStackToFourDImageFilter_h
#define rtkWarpProjectionStackToFourDImageFilter_h

#include "rtkCyclicDeformationImageFilter.h"
#include "rtkProjectionStackToFourDImageFilter.h"

#ifdef RTK_USE_CUDA
#  include "rtkCudaWarpBackProjectionImageFilter.h"
#  include "rtkCudaCyclicDeformationImageFilter.h"
#endif

namespace rtk
{
/** \class WarpProjectionStackToFourDImageFilter
 * \brief Back projection part for motion compensated iterative 4D reconstruction
 *
 * This filter is similar to ProjectionStackToFourDImageFilter, except
 * that it uses a motion-compensated backprojection. A 4D displacement
 * vector field is therefore required, and its back projection filter
 * cannot be changed.
 *
 * \dot
 * digraph WarpProjectionStackToFourDImageFilter {
 *
 * Input0 [label="Input 0 (4D sequence of volumes)"];
 * Input0 [shape=Mdiamond];
 * Input1 [label="Input 1 (Projections)"];
 * Input1 [shape=Mdiamond];
 * Input2 [label="Input 2 (4D Sequence of DVFs)"];
 * Input2 [shape=Mdiamond];
 * Output [label="Output (4D sequence of volumes)"];
 * Output [shape=Mdiamond];
 *
 * node [shape=box];
 * Extract [ label="itk::ExtractImageFilter" URL="\ref itk::ExtractImageFilter"];
 * VolumeSeriesSource [ label="rtk::ConstantImageSource (4D)" URL="\ref rtk::ConstantImageSource"];
 * AfterSource4D [label="", fixedsize="false", width=0, height=0, shape=none];
 * Source [ label="rtk::ConstantImageSource" URL="\ref rtk::ConstantImageSource"];
 * CyclicDeformation [label="rtk::CyclicDeformationImageFilter (for DVFs)"
 *                    URL="\ref rtk::CyclicDeformationImageFilter"];
 * Backproj [ label="rtk::CudaWarpBackProjectionImageFilter" URL="\ref rtk::CudaWarpBackProjectionImageFilter"];
 * Splat [ label="rtk::SplatWithKnownWeightsImageFilter" URL="\ref rtk::SplatWithKnownWeightsImageFilter"];
 * AfterSplat [label="", fixedsize="false", width=0, height=0, shape=none];
 *
 * Input1 -> Extract;
 * Input0 -> VolumeSeriesSource [style=invis];
 * Input2 -> CyclicDeformation;
 * VolumeSeriesSource -> AfterSource4D[arrowhead=none];
 * AfterSource4D -> Splat;
 * Extract -> Backproj;
 * CyclicDeformation -> Backproj;
 * Source -> Backproj;
 * Backproj -> Splat;
 * Splat -> AfterSplat[arrowhead=none];
 * AfterSplat -> Output;
 * AfterSplat -> AfterSource4D[style=dashed, constraint=none];
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
class WarpProjectionStackToFourDImageFilter
  : public ProjectionStackToFourDImageFilter<VolumeSeriesType, ProjectionStackType>
{
public:
  ITK_DISALLOW_COPY_AND_ASSIGN(WarpProjectionStackToFourDImageFilter);

  /** Standard class type alias. */
  using Self = WarpProjectionStackToFourDImageFilter;
  using Superclass = ProjectionStackToFourDImageFilter<VolumeSeriesType, ProjectionStackType>;
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
                                    BackProjectionImageFilter<VolumeType, VolumeType>,
                                    CudaWarpBackProjectionImageFilter>::type WarpBackProjectionImageFilter;
  using CPUDVFInterpolatorType = CyclicDeformationImageFilter<DVFSequenceImageType, DVFImageType>;
  typedef typename std::conditional<std::is_same<VolumeSeriesType, CPUVolumeSeriesType>::value,
                                    CPUDVFInterpolatorType,
                                    CudaCyclicDeformationImageFilter>::type CudaCyclicDeformationImageFilterType;
#else
  using DVFSequenceImageType = itk::Image<VectorForDVF, VolumeSeriesType::ImageDimension>;
  using DVFImageType = itk::Image<VectorForDVF, VolumeSeriesType::ImageDimension - 1>;
  using WarpBackProjectionImageFilter = BackProjectionImageFilter<VolumeType, VolumeType>;
  using CPUDVFInterpolatorType = CyclicDeformationImageFilter<DVFSequenceImageType, DVFImageType>;
  using CudaCyclicDeformationImageFilterType = CPUDVFInterpolatorType;
#endif

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(WarpProjectionStackToFourDImageFilter, ProjectionStackToFourDImageFilter);

  using SignalVectorType = std::vector<double>;

  /** The back projection filter cannot be set by the user */
  void
  SetBackProjectionFilter(const typename Superclass::BackProjectionFilterType::Pointer itkNotUsed(_arg))
  {
    itkExceptionMacro(<< "BackProjection cannot be changed");
  }

  /** The ND + time motion vector field */
  void
  SetDisplacementField(const DVFSequenceImageType * DVFs);
  typename DVFSequenceImageType::ConstPointer
  GetDisplacementField();

  void
  SetSignal(const std::vector<double> signal) override;

  /** Set and Get for the UseCudaCyclicDeformation variable */
  itkSetMacro(UseCudaCyclicDeformation, bool);
  itkGetMacro(UseCudaCyclicDeformation, bool);

protected:
  WarpProjectionStackToFourDImageFilter();
  ~WarpProjectionStackToFourDImageFilter() override = default;

  /** Does the real work. */
  void
  GenerateData() override;

  void
  GenerateOutputInformation() override;

  /** The first two inputs should not be in the same space so there is nothing
   * to verify. */
  void
  VerifyInputInformation() const override
  {}

  /** Member pointers to the filters used internally (for convenience)*/
  typename CPUDVFInterpolatorType::Pointer m_DVFInterpolatorFilter;
  std::vector<double>                      m_Signal;
  bool                                     m_UseCudaCyclicDeformation;
};
} // namespace rtk


#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkWarpProjectionStackToFourDImageFilter.hxx"
#endif

#endif
