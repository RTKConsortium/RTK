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
#include <itkTimeProbe.h>


#include "rtkConstantImageSource.h"
#include "rtkIterativeConeBeamReconstructionFilter.h"

namespace rtk
{

/** \class OSEMConeBeamReconstructionFilter
 * \brief Implements the Simultaneous Algebraic Reconstruction Technique [Andersen, 1984]
 *
 * OSEMConeBeamReconstructionFilter is a composite filter which combines
 * the different steps of the OSEM cone-beam reconstruction, mainly:
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
 * is used in the OSEM pipeline.
 *
 * \dot
 * digraph OSEMConeBeamReconstructionFilter {
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
 * Divide [ label="itk::DivideOrZeroOutImageFilter" URL="\ref itk::DivideOrZeroOutImageFilter"];
 * GatingWeight [ label="itk::MultiplyImageFilter (by gating weight)" URL="\ref itk::MultiplyImageFilter", style=dashed];
 * Displaced [ label="rtk::DisplacedDetectorImageFilter" URL="\ref rtk::DisplacedDetectorImageFilter"];
 * ConstantProjectionStack [ label="rtk::ConstantImageSource" URL="\ref rtk::ConstantImageSource"];
 * ExtractConstantProjection [ label="itk::ExtractImageFilter" URL="\ref itk::ExtractImageFilter"];
 * RayBox [ label="rtk::RayBoxIntersectionImageFilter" URL="\ref rtk::RayBoxIntersectionImageFilter"];
 * ConstantVolume [ label="rtk::ConstantImageSource" URL="\ref rtk::ConstantImageSource"];
 * BackProjection [ label="rtk::BackProjectionImageFilter" URL="\ref rtk::BackProjectionImageFilter"];
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
 * MultiplyByLambda -> Divide;
 * Divide -> GatingWeight;
 * GatingWeight -> Displaced;
 * ConstantProjectionStack -> ExtractConstantProjection;
 * ExtractConstantProjection -> RayBox;
 * RayBox -> Divide;
 * Displaced -> BackProjection;
 * BackProjection -> OutofBP [arrowhead=none];
 * OutofBP -> Add;
 * OutofBP -> BeforeBP [style=dashed, constraint=false];
 * Add -> Threshold;
 * Threshold -> OutofThreshold [arrowhead=none];
 * OutofThreshold -> OutofInput0 [headport="se", style=dashed];
 * OutofThreshold -> Output;
 * }
 * \enddot
 *
 * \test rtkOSEMtest.cxx
 *
 * \author Simon Rit
 *
 * \ingroup ReconstructionAlgorithm
 */
template<class TVolumeImage, class TProjectionImage=TVolumeImage>
class ITK_EXPORT OSEMConeBeamReconstructionFilter :
  public rtk::IterativeConeBeamReconstructionFilter<TVolumeImage, TProjectionImage>
{
public:
  /** Standard class typedefs. */
  typedef OSEMConeBeamReconstructionFilter					Self;
  typedef IterativeConeBeamReconstructionFilter<TVolumeImage, TProjectionImage> Superclass;
  typedef itk::SmartPointer<Self>						Pointer;
  typedef itk::SmartPointer<const Self>						ConstPointer;

  /** Some convenient typedefs. */
  typedef TVolumeImage	   VolumeType;
  typedef TProjectionImage ProjectionType;

  /** Typedefs of each subfilter of this composite filter */
  typedef itk::ExtractImageFilter< ProjectionType, ProjectionType >		    ExtractFilterType;
  typedef itk::MultiplyImageFilter< VolumeType, VolumeType, VolumeType >	    MultiplyFilterType;
  typedef rtk::ForwardProjectionImageFilter< ProjectionType, VolumeType >	    ForwardProjectionFilterType;
  typedef rtk::BackProjectionImageFilter< VolumeType, ProjectionType >		    BackProjectionFilterType;
  typedef itk::DivideImageFilter<ProjectionType, ProjectionType, ProjectionType>    DivideProjectionFilterType;
  typedef itk::DivideImageFilter<VolumeType, VolumeType, VolumeType>		    DivideVolumeFilterType;
  typedef rtk::ConstantImageSource<VolumeType>					    ConstantVolumeSourceType;
  typedef rtk::ConstantImageSource<VolumeType>					    ConstantProjectionSourceType;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(OSEMConeBeamReconstructionFilter, IterativeConeBeamReconstructionFilter);

  /** Get / Set the object pointer to projection geometry */
  itkGetMacro(Geometry, ThreeDCircularProjectionGeometry::Pointer);
  itkSetMacro(Geometry, ThreeDCircularProjectionGeometry::Pointer);

  void PrintTiming(std::ostream& os) const;

  /** Get / Set the number of iterations. Default is 3. */
  itkGetMacro(NumberOfIterations, unsigned int);
  itkSetMacro(NumberOfIterations, unsigned int);

  /** Get / Set the number of projections per subset. Default is 1. */
  itkGetMacro(NumberOfProjectionsPerSubset, unsigned int);
  itkSetMacro(NumberOfProjectionsPerSubset, unsigned int);

//  /** Get / Set the convergence factor. Default is 0.3. */
//  itkGetMacro(Lambda, double);
//  itkSetMacro(Lambda, double);

//  /** Get / Set the positivity enforcement behaviour */
//  itkGetMacro(EnforcePositivity, bool);
//  itkSetMacro(EnforcePositivity, bool);

  /** Select the ForwardProjection filter */
  void SetForwardProjectionFilter (int _arg) ITK_OVERRIDE;

  /** Select the backprojection filter */
  void SetBackProjectionFilter (int _arg) ITK_OVERRIDE;

//  /** In the case of a gated OSEM, set the gating weights */
//  void SetGatingWeights(std::vector<float> weights);

//  /** Set / Get whether the displaced detector filter should be disabled */
//  itkSetMacro(DisableDisplacedDetectorFilter, bool)
//  itkGetMacro(DisableDisplacedDetectorFilter, bool)
protected:
  OSEMConeBeamReconstructionFilter();
  ~OSEMConeBeamReconstructionFilter() {}

  void GenerateInputRequestedRegion() ITK_OVERRIDE;

  void GenerateOutputInformation() ITK_OVERRIDE;

  void GenerateData() ITK_OVERRIDE;

  /** The two inputs should not be in the same space so there is nothing
   * to verify. */
  void VerifyInputInformation() ITK_OVERRIDE {}

  /** Pointers to each subfilter of this composite filter */
  typename ExtractFilterType::Pointer		 m_ExtractFilter;
  typename ForwardProjectionFilterType::Pointer	 m_ForwardProjectionFilter;
  typename MultiplyFilterType::Pointer		 m_MultiplyFilter;
  typename BackProjectionFilterType::Pointer	 m_BackProjectionFilter;
  typename BackProjectionFilterType::Pointer	 m_BackProjectionNormalizationFilter;
  typename DivideProjectionFilterType::Pointer	 m_DivideProjectionFilter;
  typename DivideVolumeFilterType::Pointer	 m_DivideVolumeFilter;
  typename ConstantProjectionSourceType::Pointer m_ZeroConstantProjectionStackSource;
  typename ConstantProjectionSourceType::Pointer m_OneConstantProjectionStackSource;
  typename ConstantVolumeSourceType::Pointer	 m_ConstantVolumeSource;

private:
  /** Number of projections processed before the volume is updated (several for OS-EM) */
  unsigned int m_NumberOfProjectionsPerSubset;

  //purposely not implemented
  OSEMConeBeamReconstructionFilter(const Self&);
  void operator=(const Self&);

  /** Geometry object */
  ThreeDCircularProjectionGeometry::Pointer m_Geometry;

  /** Number of iterations */
  unsigned int m_NumberOfIterations;

  /** Convergence factor according to Andersen's publications which relates
   * to the step size of the gradient descent. Default 0.3, Must be in (0,2). */
  double m_Lambda;

  /** Have gating weights been set ? If so, apply them, otherwise ignore
   * the gating weights filter */
  bool                m_IsGated;
  std::vector<float>  m_GatingWeights;

  /** Time probes */
//  itk::TimeProbe m_ExtractProbe;
//  itk::TimeProbe m_ZeroMultiplyProbe;
//  itk::TimeProbe m_ForwardProjectionProbe;
//  itk::TimeProbe m_SubtractProbe;
//  itk::TimeProbe m_DisplacedDetectorProbe;
//  itk::TimeProbe m_MultiplyProbe;
//  itk::TimeProbe m_RayBoxProbe;
//  itk::TimeProbe m_DivideProbe;
//  itk::TimeProbe m_BackProjectionProbe;
//  itk::TimeProbe m_ThresholdProbe;
//  itk::TimeProbe m_AddProbe;
//  itk::TimeProbe m_GatingProbe;

}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkOSEMConeBeamReconstructionFilter.hxx"
#endif

#endif
