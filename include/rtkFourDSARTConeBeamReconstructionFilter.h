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

#ifndef rtkFourDSARTConeBeamReconstructionFilter_h
#define rtkFourDSARTConeBeamReconstructionFilter_h

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
#include "rtkProjectionStackToFourDImageFilter.h"
#include "rtkFourDToProjectionStackImageFilter.h"
#include "rtkDisplacedDetectorImageFilter.h"

namespace rtk
{

/** \class FourDSARTConeBeamReconstructionFilter
 * \brief Implements the 4D Simultaneous Algebraic Reconstruction Technique
 *
 * FourDSARTConeBeamReconstructionFilter is a composite filter. The pipeline
 * is essentially the same as in SARTConeBeamReconstructionFilter, with
 * the ForwardProjectionImageFilter replaced by 4DToProjectionStackImageFilter
 * and the BackProjectionImageFilter replaced by ProjectionStackTo4DImageFilter.
 *
 * \dot
 * digraph FourDSARTConeBeamReconstructionFilter {
 *
 * Input0 [ label="Input 0 (Volume)"];
 * Input0 [shape=Mdiamond];
 * Input1 [label="Input 1 (Projections)"];
 * Input1 [shape=Mdiamond];
 * Output [label="Output (Reconstruction)"];
 * Output [shape=Mdiamond];
 *
 * node [shape=box];
 * FourDToProjectionStack [ label="rtk::FourDToProjectionStackImageFilter"
 *                          URL="\ref rtk::FourDToProjectionStackImageFilter"];
 * Extract [ label="itk::ExtractImageFilter" URL="\ref itk::ExtractImageFilter"];
 * MultiplyByZero [ label="itk::MultiplyImageFilter (by zero)" URL="\ref itk::MultiplyImageFilter"];
 * AfterExtract [label="", fixedsize="false", width=0, height=0, shape=none];
 * Subtract [ label="itk::SubtractImageFilter" URL="\ref itk::SubtractImageFilter"];
 * MultiplyByLambda [ label="itk::MultiplyImageFilter (by lambda)" URL="\ref itk::MultiplyImageFilter"];
 * Divide [ label="itk::DivideOrZeroOutImageFilter" URL="\ref itk::DivideOrZeroOutImageFilter"];
 * Displaced [ label="rtk::DisplacedDetectorImageFilter" URL="\ref rtk::DisplacedDetectorImageFilter"];
 * ConstantProjectionStack [ label="rtk::ConstantImageSource" URL="\ref rtk::ConstantImageSource"];
 * ExtractConstantProjection [ label="itk::ExtractImageFilter" URL="\ref itk::ExtractImageFilter"];
 * RayBox [ label="rtk::RayBoxIntersectionImageFilter" URL="\ref rtk::RayBoxIntersectionImageFilter"];
 * ConstantVolume [ label="rtk::ConstantImageSource" URL="\ref rtk::ConstantImageSource"];
 * ProjectionStackToFourD [ label="rtk::ProjectionStackToFourDImageFilter"
 *                          URL="\ref rtk::ProjectionStackToFourDImageFilter"];
 * Add [ label="itk::AddImageFilter (accumulates corrections)" URL="\ref itk::AddImageFilter"];
 * Add2 [ label="itk::AddImageFilter (adds correction to current 4D volume)" URL="\ref itk::AddImageFilter"];
 * OutofInput0 [label="", fixedsize="false", width=0, height=0, shape=none];
 * Threshold [ label="itk::ThresholdImageFilter" URL="\ref itk::ThresholdImageFilter"];
 * OutofThreshold [label="", fixedsize="false", width=0, height=0, shape=none];
 * OutofBP [label="", fixedsize="false", width=0, height=0, shape=none];
 * BeforeAdd [label="", fixedsize="false", width=0, height=0, shape=none];
 * BeforeAdd2 [label="", fixedsize="false", width=0, height=0, shape=none];
 * Input0 -> OutofInput0 [arrowhead=none];
 * OutofInput0 -> FourDToProjectionStack;
 * OutofInput0 -> Add2;
 * BeforeAdd -> Add;
 * ConstantVolume -> BeforeAdd [arrowhead=none];
 * OutofInput0 -> ProjectionStackToFourD;
 * Extract -> AfterExtract[arrowhead=none];
 * AfterExtract -> MultiplyByZero;
 * AfterExtract -> Subtract;
 * MultiplyByZero -> FourDToProjectionStack;
 * Input1 -> Extract;
 * FourDToProjectionStack -> Subtract;
 * Subtract -> MultiplyByLambda;
 * MultiplyByLambda -> Divide;
 * ConstantProjectionStack -> ExtractConstantProjection;
 * ExtractConstantProjection -> RayBox;
 * RayBox -> Divide;
 * Divide -> Displaced;
 * Displaced -> ProjectionStackToFourD;
 * ProjectionStackToFourD -> Add;
 * Add -> BeforeAdd2 [arrowhead=none];
 * BeforeAdd2 -> Add2;
 * BeforeAdd2 -> BeforeAdd [style=dashed, constraint=false];
 * Add2 -> Threshold;
 * Threshold -> OutofThreshold [arrowhead=none];
 * OutofThreshold -> OutofInput0 [headport="se", style=dashed];
 * OutofThreshold -> Output;
 * }
 * \enddot
 *
 * \test rtkfourdsarttest.cxx
 *
 * \author Cyril Mory
 *
 * \ingroup RTK ReconstructionAlgorithm
 */
template <class VolumeSeriesType, class ProjectionStackType>
class ITK_TEMPLATE_EXPORT FourDSARTConeBeamReconstructionFilter
  : public rtk::IterativeConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(FourDSARTConeBeamReconstructionFilter);

  /** Standard class type alias. */
  using Self = FourDSARTConeBeamReconstructionFilter;
  using Superclass = IterativeConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Some convenient type alias. */
  using InputImageType = VolumeSeriesType;
  using OutputImageType = VolumeSeriesType;
  using VolumeType = ProjectionStackType;

  using ForwardProjectionType = typename Superclass::ForwardProjectionType;
  using BackProjectionType = typename Superclass::BackProjectionType;

  /** Typedefs of each subfilter of this composite filter */
  using ExtractFilterType = itk::ExtractImageFilter<ProjectionStackType, ProjectionStackType>;
  using ForwardProjectionFilterType = rtk::ForwardProjectionImageFilter<ProjectionStackType, ProjectionStackType>;
  using FourDToProjectionStackFilterType =
    rtk::FourDToProjectionStackImageFilter<ProjectionStackType, VolumeSeriesType>;
  using SubtractFilterType = itk::SubtractImageFilter<ProjectionStackType, ProjectionStackType>;
  using MultiplyFilterType = itk::MultiplyImageFilter<ProjectionStackType, ProjectionStackType, ProjectionStackType>;
  using AddFilterType = itk::AddImageFilter<VolumeSeriesType, VolumeSeriesType>;
  using BackProjectionFilterType = rtk::BackProjectionImageFilter<VolumeType, VolumeType>;
  using ProjectionStackToFourDFilterType =
    rtk::ProjectionStackToFourDImageFilter<VolumeSeriesType, ProjectionStackType>;
  using RayBoxIntersectionFilterType = rtk::RayBoxIntersectionImageFilter<ProjectionStackType, ProjectionStackType>;
  using DivideFilterType =
    itk::DivideOrZeroOutImageFilter<ProjectionStackType, ProjectionStackType, ProjectionStackType>;
  using DisplacedDetectorFilterType = rtk::DisplacedDetectorImageFilter<ProjectionStackType>;
  using ConstantVolumeSeriesSourceType = rtk::ConstantImageSource<VolumeSeriesType>;
  using ConstantProjectionStackSourceType = rtk::ConstantImageSource<ProjectionStackType>;
  using ThresholdFilterType = itk::ThresholdImageFilter<VolumeSeriesType>;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkOverrideGetNameOfClassMacro(FourDSARTConeBeamReconstructionFilter);

  /** The 4D image to be updated.*/
  void
  SetInputVolumeSeries(const VolumeSeriesType * VolumeSeries);
  typename VolumeSeriesType::ConstPointer
  GetInputVolumeSeries();

  /** The stack of measured projections */
  void
  SetInputProjectionStack(const ProjectionStackType * Projection);
  typename ProjectionStackType::Pointer
  GetInputProjectionStack();

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

  /** Pass the interpolation weights to subfilters */
  void
  SetWeights(const itk::Array2D<float> _arg);

  /** Store the phase signal in a member variable */
  virtual void
  SetSignal(const std::vector<double> signal);

  /** Set / Get whether the displaced detector filter should be disabled */
  itkSetMacro(DisableDisplacedDetectorFilter, bool);
  itkGetMacro(DisableDisplacedDetectorFilter, bool);

protected:
  FourDSARTConeBeamReconstructionFilter();
  ~FourDSARTConeBeamReconstructionFilter() override = default;

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
  typename ExtractFilterType::Pointer                 m_ExtractFilter;
  typename ExtractFilterType::Pointer                 m_ExtractFilterRayBox;
  typename MultiplyFilterType::Pointer                m_ZeroMultiplyFilter;
  typename ForwardProjectionFilterType::Pointer       m_ForwardProjectionFilter;
  typename FourDToProjectionStackFilterType::Pointer  m_FourDToProjectionStackFilter;
  typename SubtractFilterType::Pointer                m_SubtractFilter;
  typename AddFilterType::Pointer                     m_AddFilter;
  typename AddFilterType::Pointer                     m_AddFilter2;
  typename MultiplyFilterType::Pointer                m_MultiplyFilter;
  typename BackProjectionFilterType::Pointer          m_BackProjectionFilter;
  typename ProjectionStackToFourDFilterType::Pointer  m_ProjectionStackToFourDFilter;
  typename RayBoxIntersectionFilterType::Pointer      m_RayBoxFilter;
  typename DivideFilterType::Pointer                  m_DivideFilter;
  typename DisplacedDetectorFilterType::Pointer       m_DisplacedDetectorFilter;
  typename ConstantProjectionStackSourceType::Pointer m_ConstantProjectionStackSource;
  typename ConstantVolumeSeriesSourceType::Pointer    m_ConstantVolumeSeriesSource;
  typename ThresholdFilterType::Pointer               m_ThresholdFilter;

  /** Miscellaneous member variables */
  std::vector<unsigned int> m_ProjectionsOrder;
  bool                      m_ProjectionsOrderInitialized;
  bool                      m_EnforcePositivity;
  std::vector<double>       m_Signal;
  bool                      m_DisableDisplacedDetectorFilter;

private:
  /** Number of projections processed before the volume is updated (1 for SART,
   * several for OS-SART, all for SIRT) */
  unsigned int m_NumberOfProjectionsPerSubset;

  /** Geometry object */
  ThreeDCircularProjectionGeometry::Pointer m_Geometry;

  /** Number of iterations */
  unsigned int m_NumberOfIterations;

  /** Convergence factor according to Andersen's publications which relates
   * to the step size of the gradient descent. Default 0.3, Must be in (0,2). */
  double m_Lambda;
}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkFourDSARTConeBeamReconstructionFilter.hxx"
#endif

#endif
