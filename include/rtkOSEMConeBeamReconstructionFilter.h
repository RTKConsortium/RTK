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

#ifndef rtkOSEMConeBeamReconstructionFilter_h
#define rtkOSEMConeBeamReconstructionFilter_h

#include "rtkBackProjectionImageFilter.h"
#include "rtkForwardProjectionImageFilter.h"

#include <itkExtractImageFilter.h>
#include <itkMultiplyImageFilter.h>
#include <itkAddImageAdaptor.h>
#include <itkDivideImageFilter.h>
#include <itkDivideOrZeroOutImageFilter.h>

#include "rtkConstantImageSource.h"
#include "rtkIterativeConeBeamReconstructionFilter.h"

namespace rtk
{

/** \class OSEMConeBeamReconstructionFilter
 * \brief Implements the Ordered-Subset Expectation-Maximization algorithm.
 *
 * OSEMConeBeamReconstructionFilter is a composite filter which combines
 * the different steps of the OSEM cone-beam reconstruction, mainly:
 * - ExtractFilterType to create the subsets.
 * - ForwardProjectionImageFilter,
 * - DivideImageFilter,
 * - BackProjectionImageFilter.
 * The input stack of projections is processed piece by piece (the size is
 * controlled with ProjectionSubsetSize) via the use of itk::ExtractImageFilter
 * to extract sub-stacks.
 *
 * One weighting steps must be applied when processing a given subset:
 * - each voxel of the back projection must be divided by the value it would take if
 * a projection filled with ones was being reprojected.
 *
 * \dot
 * digraph OSEMConeBeamReconstructionFilter {
 *
 *  Input0 [ label="Input 0 (Volume)"];
 *  Input0 [shape=Mdiamond];
 *  Input1 [label="Input 1 (Projections)"];
 *  Input1 [shape=Mdiamond];
 *  Output [label="Output (Reconstruction)"];
 *  Output [shape=Mdiamond];
 *
 *  node [shape=box];
 *  ForwardProject [ label="rtk::ForwardProjectionImageFilter" URL="\ref rtk::ForwardProjectionImageFilter"];
 *  Extract [ label="itk::ExtractImageFilter" URL="\ref itk::ExtractImageFilter"];
 *  Divide1 [ label="itk::DivideImageFilter" URL="\ref itk::DivideImageFilter"];
 *  Divide [ label="itk::DivideImageFilter" URL="\ref itk::DivideImageFilter"];
 *  ProjectionZero [ label="rtk::ConstantImageSource (full of zero)" URL="\ref rtk::ConstantImageSource"];
 *  ConstantVolume2[ label="rtk::ConstantImageSource" URL="\ref rtk::ConstantImageSource"];
 *  ConstantVolume [ label="rtk::ConstantImageSource" URL="\ref rtk::ConstantImageSource"];
 *  ConstantProjectionStack [ label="rtk::ConstantImageSource (full of one)" URL="\ref rtk::ConstantImageSource"];
 *  BackProjection [ label="rtk::BackProjectionImageFilter" URL="\ref rtk::BackProjectionImageFilter"];
 *  BackProjection2 [ label="rtk::BackProjectionImageFilter" URL="\ref rtk::BackProjectionImageFilter"];
 *  Multiply [ label="itk::MultiplyImageFilter " URL="\ref itk::MultiplyImageFilter"];
 *  OutofInput0 [label="", fixedsize="false", width=0, height=0, shape=none];
 *  OutofMultiply [label="", fixedsize="false", width=0, height=0, shape=none];
 *  OutofBP [label="", fixedsize="false", width=0, height=0, shape=none];
 *  BeforeBP [label="", fixedsize="false", width=0, height=0, shape=none];
 *  BeforeMultiply [label="", fixedsize="false", width=0, height=0, shape=none];
 *  Input0 -> OutofInput0 [arrowhead=none];
 *  OutofInput0 -> ForwardProject;
 *  OutofInput0 -> BeforeMultiply [arrowhead=none];
 *  BeforeMultiply -> Multiply;
 *  Extract -> Divide1;
 *  ProjectionZero -> ForwardProject;
 *  Input1 -> Extract;
 *  ForwardProject -> Divide1;
 *  Divide1 -> BackProjection;
 *  ConstantVolume -> BeforeBP [arrowhead=none];
 *  BeforeBP -> BackProjection;
 *  ConstantProjectionStack -> BackProjection2;
 *  ConstantVolume2 -> BackProjection2;
 *  BackProjection2 -> Divide;
 *  BackProjection -> OutofBP [arrowhead=none];
 *  Divide -> Multiply
 *  OutofBP-> Divide;
 *  OutofBP -> BeforeBP [style=dashed, constraint=false];
 *  Multiply -> OutofMultiply;
 *  OutofMultiply -> OutofInput0 [headport="se", style=dashed];
 *  OutofMultiply -> Output;
 *  }
 * \enddot
 *
 * \test rtkosemtest.cxx
 *
 * \author Antoine Robert
 *
 * \ingroup RTK ReconstructionAlgorithm
 */
template <class TVolumeImage, class TProjectionImage = TVolumeImage>
class ITK_EXPORT OSEMConeBeamReconstructionFilter
  : public rtk::IterativeConeBeamReconstructionFilter<TVolumeImage, TProjectionImage>
{
public:
  ITK_DISALLOW_COPY_AND_ASSIGN(OSEMConeBeamReconstructionFilter);

  /** Standard class type alias. */
  typedef OSEMConeBeamReconstructionFilter Self;
  using Superclass = IterativeConeBeamReconstructionFilter<TVolumeImage, TProjectionImage>;
  typedef itk::SmartPointer<Self>       Pointer;
  typedef itk::SmartPointer<const Self> ConstPointer;

  /** Some convenient type alias. */
  using VolumeType = TVolumeImage;
  using ProjectionType = TProjectionImage;

  /** Typedefs of each subfilter of this composite filter */
  using ExtractFilterType = itk::ExtractImageFilter<ProjectionType, ProjectionType>;
  using MultiplyFilterType = itk::MultiplyImageFilter<VolumeType, VolumeType, VolumeType>;
  using ForwardProjectionFilterType = rtk::ForwardProjectionImageFilter<ProjectionType, VolumeType>;
  using BackProjectionFilterType = rtk::BackProjectionImageFilter<VolumeType, ProjectionType>;
  using DivideProjectionFilterType = itk::DivideOrZeroOutImageFilter<ProjectionType, ProjectionType, ProjectionType>;
  using DivideVolumeFilterType = itk::DivideOrZeroOutImageFilter<VolumeType, VolumeType, VolumeType>;
  using ConstantVolumeSourceType = rtk::ConstantImageSource<VolumeType>;
  using ConstantProjectionSourceType = rtk::ConstantImageSource<ProjectionType>;

  using ForwardProjectionType = typename Superclass::ForwardProjectionType;
  using BackProjectionType = typename Superclass::BackProjectionType;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(OSEMConeBeamReconstructionFilter, IterativeConeBeamReconstructionFilter);

  /** Get / Set the object pointer to projection geometry */
  itkGetModifiableObjectMacro(Geometry, ThreeDCircularProjectionGeometry);
  itkSetObjectMacro(Geometry, ThreeDCircularProjectionGeometry);

  /** Get / Set the number of iterations. Default is 3. */
  itkGetMacro(NumberOfIterations, unsigned int);
  itkSetMacro(NumberOfIterations, unsigned int);

  /** Get / Set the number of projections per subset. Default is 1. */
  itkGetMacro(NumberOfProjectionsPerSubset, unsigned int);
  itkSetMacro(NumberOfProjectionsPerSubset, unsigned int);

  /** Get / Set the sigma zero of the PSF. Default is 1.5417233052142099 */
  itkGetMacro(SigmaZero, float);
  itkSetMacro(SigmaZero, float);

  /** Get / Set the alpha of the PSF. Default is 0.016241189545787734 */
  itkGetMacro(Alpha, float);
  itkSetMacro(Alpha, float);

  /** Select the ForwardProjection filter */
  void
  SetForwardProjectionFilter(ForwardProjectionType _arg) override;

  /** Select the backprojection filter */
  void
  SetBackProjectionFilter(BackProjectionType _arg) override;

protected:
  OSEMConeBeamReconstructionFilter();
  ~OSEMConeBeamReconstructionFilter() override = default;

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
  typename ForwardProjectionFilterType::Pointer  m_ForwardProjectionFilter;
  typename MultiplyFilterType::Pointer           m_MultiplyFilter;
  typename BackProjectionFilterType::Pointer     m_BackProjectionFilter;
  typename BackProjectionFilterType::Pointer     m_BackProjectionNormalizationFilter;
  typename DivideProjectionFilterType::Pointer   m_DivideProjectionFilter;
  typename DivideVolumeFilterType::Pointer       m_DivideVolumeFilter;
  typename ConstantProjectionSourceType::Pointer m_ZeroConstantProjectionStackSource;
  typename ConstantProjectionSourceType::Pointer m_OneConstantProjectionStackSource;
  typename ConstantVolumeSourceType::Pointer     m_ConstantVolumeSource;

private:
  /** Number of projections processed before the volume is updated (several for OS-EM) */
  unsigned int m_NumberOfProjectionsPerSubset;

  /** Geometry object */
  ThreeDCircularProjectionGeometry::Pointer m_Geometry;

  /** Number of iterations */
  unsigned int m_NumberOfIterations;

  /** PSF correction coefficients */
  float m_SigmaZero;
  float m_Alpha;

}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkOSEMConeBeamReconstructionFilter.hxx"
#endif

#endif
