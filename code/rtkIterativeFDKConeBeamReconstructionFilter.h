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

#ifndef __rtkIterativeFDKConeBeamReconstructionFilter_h
#define __rtkIterativeFDKConeBeamReconstructionFilter_h

#include <itkMultiplyImageFilter.h>
#include <itkSubtractImageFilter.h>
#include <itkTimeProbe.h>
#include <itkThresholdImageFilter.h>

#include "rtkConstantImageSource.h"
#include "rtkParkerShortScanImageFilter.h"
#include "rtkDisplacedDetectorForOffsetFieldOfViewImageFilter.h"
#include "rtkIterativeConeBeamReconstructionFilter.h"
#include "rtkFDKConeBeamReconstructionFilter.h"

namespace rtk
{

/** \class IterativeFDKConeBeamReconstructionFilter
 * \brief Implements the Iterative FDK
 *
 * IterativeFDKConeBeamReconstructionFilter is a composite filter which combines
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
 * digraph IterativeFDKConeBeamReconstructionFilter {
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
 * \test rtkiterativefdktest.cxx
 *
 * \author Simon Rit
 *
 * \ingroup ReconstructionAlgorithm
 */
template<class TInputImage, class TOutputImage=TInputImage, class TFFTPrecision=double>
class ITK_EXPORT IterativeFDKConeBeamReconstructionFilter :
  public rtk::IterativeConeBeamReconstructionFilter<TInputImage, TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef IterativeFDKConeBeamReconstructionFilter                         Self;
  typedef IterativeConeBeamReconstructionFilter<TInputImage, TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                                          Pointer;
  typedef itk::SmartPointer<const Self>                                    ConstPointer;

  /** Some convenient typedefs. */
  typedef TInputImage  InputImageType;
  typedef TOutputImage OutputImageType;

  /** Typedefs of each subfilter of this composite filter */
  typedef rtk::DisplacedDetectorImageFilter<OutputImageType, OutputImageType>                       DisplacedDetectorFilterType;
  typedef rtk::ParkerShortScanImageFilter<OutputImageType, OutputImageType>                         ParkerFilterType;
  typedef rtk::FDKConeBeamReconstructionFilter<OutputImageType, OutputImageType, TFFTPrecision>     FDKFilterType;
  typedef itk::MultiplyImageFilter< OutputImageType, OutputImageType, OutputImageType >             MultiplyFilterType;
  typedef itk::SubtractImageFilter< OutputImageType, OutputImageType >                              SubtractFilterType;
  typedef rtk::ConstantImageSource<OutputImageType>                                                 ConstantImageSourceType;
  typedef rtk::ForwardProjectionImageFilter< OutputImageType, OutputImageType >                     ForwardProjectionFilterType;
  typedef itk::ThresholdImageFilter<OutputImageType>                                                ThresholdFilterType;

/** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(IterativeFDKConeBeamReconstructionFilter, IterativeConeBeamReconstructionFilter);

  /** Get / Set the object pointer to projection geometry */
  itkGetMacro(Geometry, ThreeDCircularProjectionGeometry::Pointer);
  itkSetMacro(Geometry, ThreeDCircularProjectionGeometry::Pointer);

  void PrintTiming(std::ostream& os) const;

  /** Get / Set the number of iterations. Default is 3. */
  itkGetMacro(NumberOfIterations, unsigned int);
  itkSetMacro(NumberOfIterations, unsigned int);

  /** Get / Set the convergence factor. Default is 0.3. */
  itkGetMacro(Lambda, double);
  itkSetMacro(Lambda, double);

  /** Get / Set the positivity enforcement behaviour */
  itkGetMacro(EnforcePositivity, bool);
  itkSetMacro(EnforcePositivity, bool);

  /** Select the ForwardProjection filter */
  void SetForwardProjectionFilter (int _arg);

  /** Select the backprojection filter */
  void SetBackProjectionFilter (int _arg){}

  /** Get / Set the truncation correction */
  itkGetMacro(TruncationCorrection, double);
  itkSetMacro(TruncationCorrection, double);

  /** Get / Set the Hann cut frequency */
  itkGetMacro(HannCutFrequency, double);
  itkSetMacro(HannCutFrequency, double);

  /** Get / Set the Hann cut frequency on axis Y */
  itkGetMacro(HannCutFrequencyY, double);
  itkSetMacro(HannCutFrequencyY, double);

  /** Get / Set the number of iterations. Default is 3. */
  itkGetMacro(ProjectionSubsetSize, unsigned int);
  itkSetMacro(ProjectionSubsetSize, unsigned int);

protected:
  IterativeFDKConeBeamReconstructionFilter();
  ~IterativeFDKConeBeamReconstructionFilter(){}

  virtual void GenerateInputRequestedRegion();

  virtual void GenerateOutputInformation();

  virtual void GenerateData();

  /** The two inputs should not be in the same space so there is nothing
   * to verify. */
  virtual void VerifyInputInformation() {}

  /** Pointers to each subfilter of this composite filter */
  typename DisplacedDetectorFilterType::Pointer       m_DisplacedDetectorFilter;
  typename ParkerFilterType::Pointer                  m_ParkerFilter;
  typename FDKFilterType::Pointer                     m_FDKFilter;
  typename ThresholdFilterType::Pointer               m_ThresholdFilter;
  typename ForwardProjectionFilterType::Pointer       m_ForwardProjectionFilter;
  typename SubtractFilterType::Pointer                m_SubtractFilter;
  typename MultiplyFilterType::Pointer                m_MultiplyFilter;
  typename ConstantImageSourceType::Pointer           m_ConstantProjectionStackSource;

  bool m_EnforcePositivity;
  double m_TruncationCorrection;
  double m_HannCutFrequency;
  double m_HannCutFrequencyY;
  double m_ProjectionSubsetSize;

private:
  //purposely not implemented
  IterativeFDKConeBeamReconstructionFilter(const Self&);
  void operator=(const Self&);

  /** Geometry object */
  ThreeDCircularProjectionGeometry::Pointer m_Geometry;

  /** Number of iterations */
  unsigned int m_NumberOfIterations;

  /** Convergence factor. Default 0.3 */
  double m_Lambda;

  /** Time probes */
  itk::TimeProbe m_FDKProbe;
  itk::TimeProbe m_AddProbe;
  itk::TimeProbe m_ThresholdProbe;
  itk::TimeProbe m_ForwardProjectionProbe;
  itk::TimeProbe m_SubtractProbe;
  itk::TimeProbe m_MultiplyProbe;

}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkIterativeFDKConeBeamReconstructionFilter.txx"
#endif

#endif
