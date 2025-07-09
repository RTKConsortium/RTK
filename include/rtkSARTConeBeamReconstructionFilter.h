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

#ifndef rtkSARTConeBeamReconstructionFilter_h
#define rtkSARTConeBeamReconstructionFilter_h

#include "rtkBackProjectionImageFilter.h"
#include "rtkForwardProjectionImageFilter.h"

#include <itkExtractImageFilter.h>
#include <itkMultiplyImageFilter.h>
#include <itkSubtractImageFilter.h>
#include <itkAddImageAdaptor.h>
#include <itkAddImageFilter.h>
#include <itkDivideOrZeroOutImageFilter.h>
#include <itkThresholdImageFilter.h>

#include "rtkRayBoxIntersectionImageFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkIterativeConeBeamReconstructionFilter.h"
#include "rtkDisplacedDetectorImageFilter.h"

namespace rtk
{

/** \class SARTConeBeamReconstructionFilter
 * \brief Implements the Simultaneous Algebraic Reconstruction Technique [Andersen, 1984]
 *
 * SARTConeBeamReconstructionFilter is a composite filter which combines
 * the different steps of the SART cone-beam reconstruction, mainly:
 * - ExtractFilterType to work on one projection at a time
 * - ForwardProjectionImageFilter,
 * - SubtractImageFilter,
 * - BackProjectionImageFilter.
 * The input stack of projections is processed piece by piece (the size is
 * controlled with ProjectionSubsetSize) via the use of itk::ExtractImageFilter
 * to extract sub-stacks.
 *
 * Two weighting steps must be applied when processing a given projection:
 * - each pixel of the forward projection must be divided by the total length of the
 * intersection between the ray and the reconstructed volume. This weighting step
 * is performed using the part of the pipeline that contains RayBoxIntersectionImageFilter
 * - each voxel of the back projection must be divided by the value it would take if
 * a projection filled with ones was being reprojected. This weighting step is not
 * performed when using a voxel-based back projection, as the weights are all equal to one
 * in this case. When using a ray-based backprojector, typically Joseph,it must be performed.
 * It is implemented in NormalizedJosephBackProjectionImageFilter, which
 * is used in the SART pipeline.
 *
 * \dot
 * digraph SARTConeBeamReconstructionFilter {
 *
 * Input0 [ label="Input 0 (Volume)"];
 * Input0 [shape=Mdiamond];
 * Input1 [label="Input 1 (Projections)"];
 * Input1 [shape=Mdiamond];
 * Output [label="Output (Reconstruction)"];
 * Output [shape=Mdiamond];
 *
 * node [shape=box];
 * ForwardProject [ label="rtk::ForwardProjectionImageFilter" URL="\ref rtk::ForwardProjectionImageFilter"];
 * Extract [ label="itk::ExtractImageFilter" URL="\ref itk::ExtractImageFilter"];
 * MultiplyByZero [ label="itk::MultiplyImageFilter (by zero)" URL="\ref itk::MultiplyImageFilter"];
 * AfterExtract [label="", fixedsize="false", width=0, height=0, shape=none];
 * Subtract [ label="itk::SubtractImageFilter" URL="\ref itk::SubtractImageFilter"];
 * MultiplyByLambda [ label="itk::MultiplyImageFilter (by lambda)" URL="\ref itk::MultiplyImageFilter"];
 * DivideProj [ label="itk::DivideOrZeroOutImageFilter" URL="\ref itk::DivideOrZeroOutImageFilter"];
 * DivideVol [ label="itk::DivideOrZeroOutImageFilter" URL="\ref itk::DivideOrZeroOutImageFilter"];
 * GatingWeight [ label="itk::MultiplyImageFilter (by gating weight)"
 *                URL="\ref itk::MultiplyImageFilter", style=dashed];
 * Displaced [ label="rtk::DisplacedDetectorImageFilter" URL="\ref rtk::DisplacedDetectorImageFilter"];
 * ConstantProjectionStack [ label="rtk::ConstantImageSource (0)" URL="\ref rtk::ConstantImageSource"];
 * ExtractConstantProjection [ label="itk::ExtractImageFilter" URL="\ref itk::ExtractImageFilter"];
 * RayBox [ label="rtk::RayBoxIntersectionImageFilter" URL="\ref rtk::RayBoxIntersectionImageFilter"];
 * ConstantVolume [ label="rtk::ConstantImageSource (0)" URL="\ref rtk::ConstantImageSource"];
 * BackProjection [ label="rtk::BackProjectionImageFilter" URL="\ref rtk::BackProjectionImageFilter"];
 * ConstantProjDenom [ label="rtk::ConstantImageSource (1)" URL="\ref rtk::ConstantImageSource"];
 * BackProjectionDenom [ label="rtk::BackProjectionImageFilter" URL="\ref rtk::BackProjectionImageFilter"];
 * Add [ label="itk::AddImageFilter" URL="\ref itk::AddImageFilter"];
 * OutofInput0 [label="", fixedsize="false", width=0, height=0, shape=none];
 * Threshold [ label="itk::ThresholdImageFilter" URL="\ref itk::ThresholdImageFilter"];
 * OutofThreshold [label="", fixedsize="false", width=0, height=0, shape=none];
 * OutofBP [label="", fixedsize="false", width=0, height=0, shape=none];
 * BeforeBP [label="", fixedsize="false", width=0, height=0, shape=none];
 * BeforeAdd [label="", fixedsize="false", width=0, height=0, shape=none];
 * Input0 -> OutofInput0 [arrowhead=none];
 * OutofInput0 -> ForwardProject;
 * OutofInput0 -> BeforeAdd [arrowhead=none];
 * BeforeAdd -> Add;
 * ConstantVolume -> BeforeBP [arrowhead=none];
 * BeforeBP -> BackProjection;
 * Extract -> AfterExtract[arrowhead=none];
 * AfterExtract -> MultiplyByZero;
 * AfterExtract -> Subtract;
 * MultiplyByZero -> ForwardProject;
 * Input1 -> Extract;
 * ForwardProject -> Subtract;
 * Subtract -> MultiplyByLambda;
 * MultiplyByLambda -> DivideProj;
 * DivideProj -> GatingWeight;
 * GatingWeight -> Displaced;
 * ConstantProjectionStack -> ExtractConstantProjection;
 * ExtractConstantProjection -> RayBox;
 * RayBox -> DivideProj;
 * Displaced -> BackProjection;
 * BackProjection -> OutofBP [arrowhead=none];
 * ConstantProjDenom -> BackProjectionDenom;
 * OutofBP -> DivideVol
 * BeforeBP -> BackProjectionDenom;
 * BackProjectionDenom -> DivideVol;
 * DivideVol -> Add;
 * Add -> Threshold;
 * Threshold -> OutofThreshold [arrowhead=none];
 * OutofThreshold -> OutofInput0 [headport="se", style=dashed];
 * OutofThreshold -> Output;
 * }
 * \enddot
 *
 * \test rtksarttest.cxx
 *
 * \author Simon Rit
 *
 * \ingroup RTK ReconstructionAlgorithm
 */
template <class TVolumeImage, class TProjectionImage = TVolumeImage>
class ITK_TEMPLATE_EXPORT SARTConeBeamReconstructionFilter
  : public rtk::IterativeConeBeamReconstructionFilter<TVolumeImage, TProjectionImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(SARTConeBeamReconstructionFilter);

  /** Standard class type alias. */
  using Self = SARTConeBeamReconstructionFilter;
  using Superclass = IterativeConeBeamReconstructionFilter<TVolumeImage, TProjectionImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Some convenient type alias. */
  using VolumeType = TVolumeImage;
  using ProjectionType = TProjectionImage;
  using ProjectionPixelType = typename ProjectionType::PixelType;

  /** Typedefs of each subfilter of this composite filter */
  using ExtractFilterType = itk::ExtractImageFilter<ProjectionType, ProjectionType>;
  using MultiplyFilterType = itk::MultiplyImageFilter<ProjectionType, ProjectionType, ProjectionType>;
  using ForwardProjectionFilterType = rtk::ForwardProjectionImageFilter<ProjectionType, VolumeType>;
  using SubtractFilterType = itk::SubtractImageFilter<ProjectionType, ProjectionType>;
  using AddFilterType = itk::AddImageFilter<VolumeType, VolumeType>;
  using BackProjectionFilterType = rtk::BackProjectionImageFilter<VolumeType, ProjectionType>;
  using RayBoxIntersectionFilterType = rtk::RayBoxIntersectionImageFilter<ProjectionType, ProjectionType>;
  using DivideProjectionFilterType = itk::DivideOrZeroOutImageFilter<ProjectionType, ProjectionType, ProjectionType>;
  using DivideVolumeFilterType = itk::DivideOrZeroOutImageFilter<VolumeType, VolumeType, VolumeType>;
  using ConstantVolumeSourceType = rtk::ConstantImageSource<VolumeType>;
  using ConstantProjectionSourceType = rtk::ConstantImageSource<ProjectionType>;
  using ThresholdFilterType = itk::ThresholdImageFilter<VolumeType>;
  using DisplacedDetectorFilterType = rtk::DisplacedDetectorImageFilter<ProjectionType>;
  using GatingWeightsFilterType = itk::MultiplyImageFilter<ProjectionType, ProjectionType, ProjectionType>;

  using ForwardProjectionType = typename Superclass::ForwardProjectionType;
  using BackProjectionType = typename Superclass::BackProjectionType;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkOverrideGetNameOfClassMacro(SARTConeBeamReconstructionFilter);

  /** Get / Set the object pointer to projection geometry */
  itkGetModifiableObjectMacro(Geometry, ThreeDCircularProjectionGeometry);
  itkSetObjectMacro(Geometry, ThreeDCircularProjectionGeometry);

  /** Get / Set the number of iterations. Default is 3. */
  itkGetMacro(NumberOfIterations, unsigned int);
  itkSetMacro(NumberOfIterations, unsigned int);

  /** Get / Set the number of projections per subset. Default is 1. */
  itkGetMacro(NumberOfProjectionsPerSubset, unsigned int);
  itkSetMacro(NumberOfProjectionsPerSubset, unsigned int);

  /** Get / Set the convergence factor. Default is 0.3. */
  itkGetMacro(Lambda, double);
  itkSetMacro(Lambda, double);

  /** Get / Set the positivity enforcement behaviour */
  itkGetMacro(EnforcePositivity, bool);
  itkSetMacro(EnforcePositivity, bool);

  /** In the case of a gated SART, set the gating weights */
  void
  SetGatingWeights(std::vector<float> weights);

  /** Set / Get whether the displaced detector filter should be disabled */
  itkSetMacro(DisableDisplacedDetectorFilter, bool);
  itkGetMacro(DisableDisplacedDetectorFilter, bool);

  /** Set the threshold below which pixels in the denominator in the projection space are considered zero. The division
   * by zero will then be evaluated at zero. Avoid noise magnification from low projections values when working with
   * noisy and/or simulated data.
   */
  itkSetMacro(DivisionThreshold, ProjectionPixelType);
  itkGetMacro(DivisionThreshold, ProjectionPixelType);

protected:
  SARTConeBeamReconstructionFilter();
  ~SARTConeBeamReconstructionFilter() override = default;

  /** Checks that inputs are correctly set. */
  void
  VerifyPreconditions() const override;

  void
  GenerateInputRequestedRegion() override;

  void
  GenerateOutputInformation() override;

  void
  GenerateData() override;

  /** The two inputs should not be in the same space so there is nothing
   * to verify. */
  void
  VerifyInputInformation() const override
  {}

  /** Pointers to each subfilter of this composite filter */
  typename ExtractFilterType::Pointer            m_ExtractFilter;
  typename ExtractFilterType::Pointer            m_ExtractFilterRayBox;
  typename MultiplyFilterType::Pointer           m_ZeroMultiplyFilter;
  typename ForwardProjectionFilterType::Pointer  m_ForwardProjectionFilter;
  typename SubtractFilterType::Pointer           m_SubtractFilter;
  typename AddFilterType::Pointer                m_AddFilter;
  typename MultiplyFilterType::Pointer           m_MultiplyFilter;
  typename BackProjectionFilterType::Pointer     m_BackProjectionFilter;
  typename BackProjectionFilterType::Pointer     m_BackProjectionNormalizationFilter;
  typename RayBoxIntersectionFilterType::Pointer m_RayBoxFilter;
  typename DivideProjectionFilterType::Pointer   m_DivideProjectionFilter;
  typename DivideVolumeFilterType::Pointer       m_DivideVolumeFilter;
  typename ConstantProjectionSourceType::Pointer m_ConstantProjectionStackSource;
  typename ConstantProjectionSourceType::Pointer m_OneConstantProjectionStackSource;
  typename ConstantVolumeSourceType::Pointer     m_ConstantVolumeSource;
  typename ThresholdFilterType::Pointer          m_ThresholdFilter;
  typename DisplacedDetectorFilterType::Pointer  m_DisplacedDetectorFilter;
  typename GatingWeightsFilterType::Pointer      m_GatingWeightsFilter;

  ProjectionPixelType m_DivisionThreshold;

  bool m_EnforcePositivity;
  bool m_DisableDisplacedDetectorFilter{};

private:
  /** Number of projections processed before the volume is updated (1 for SART,
   * several for OS-SART, all for SIRT) */
  unsigned int m_NumberOfProjectionsPerSubset{1};

  /** Geometry object */
  ThreeDCircularProjectionGeometry::Pointer m_Geometry;

  /** Number of iterations */
  unsigned int m_NumberOfIterations;

  /** Convergence factor according to Andersen's publications which relates
   * to the step size of the gradient descent. Default 0.3, Must be in (0,2). */
  double m_Lambda;

  /** Have gating weights been set ? If so, apply them, otherwise ignore
   * the gating weights filter */
  bool               m_IsGated{};
  std::vector<float> m_GatingWeights;
}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkSARTConeBeamReconstructionFilter.hxx"
#endif

#endif
