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

#ifndef rtkIterativeFDKConeBeamReconstructionFilter_h
#define rtkIterativeFDKConeBeamReconstructionFilter_h

#include <itkMultiplyImageFilter.h>
#include <itkSubtractImageFilter.h>
#include <itkTimeProbe.h>
#include <itkThresholdImageFilter.h>
#include <itkDivideOrZeroOutImageFilter.h>

#include "rtkConstantImageSource.h"
#include "rtkParkerShortScanImageFilter.h"
#include "rtkDisplacedDetectorForOffsetFieldOfViewImageFilter.h"
#include "rtkIterativeConeBeamReconstructionFilter.h"
#include "rtkFDKConeBeamReconstructionFilter.h"
#include "rtkRayBoxIntersectionImageFilter.h"

namespace rtk
{

/** \class IterativeFDKConeBeamReconstructionFilter
 * \brief Implements the Iterative FDK
 *
 * IterativeFDKConeBeamReconstructionFilter is a composite filter which combines
 * the different steps of the iterative FDK cone-beam reconstruction, mainly:
 * - FDK reconstruction,
 * - Forward projection,
 * - Difference between the calculated projections and the input ones,
 * - Multiplication by a small constant
 * This "projections correction" is used at the next iteration to improve the FDK.
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
 * Displaced [ label="rtk::DisplacedDetectorImageFilter" URL="\ref rtk::DisplacedDetectorImageFilter"];
 * Parker [ label="rtk::ParkerShortScanImageFilter" URL="\ref rtk::ParkerShortScanImageFilter"];
 * FDK [ label="rtk::FDKConeBeamReconstructionFilter" URL="\ref rtk::FDKConeBeamReconstructionFilter"];
 * Subtract [ label="itk::SubtractImageFilter" URL="\ref itk::SubtractImageFilter"];
 * Multiply [ label="itk::MultiplyImageFilter (by lambda)" URL="\ref itk::MultiplyImageFilter"];
 * ConstantProjectionStack [ label="rtk::ConstantImageSource (projections)" URL="\ref rtk::ConstantImageSource"];
 * ForwardProjection [ label="rtk::ForwardProjectionImageFilter" URL="\ref rtk::ForwardProjectionImageFilter"];
 * RayBox [ label="rtk::RayBoxIntersectionImageFilter" URL="\ref rtk::RayBoxIntersectionImageFilter"];
 * Threshold [ label="itk::ThresholdImageFilter" URL="\ref itk::ThresholdImageFilter"];
 * Divide [ label="itk::DivideOrZeroOutImageFilter" URL="\ref itk::DivideOrZeroOutImageFilter"];
 *
 * AfterInput1 [label="", fixedsize="false", width=0, height=0, shape=none];
 * AfterInput0 [label="", fixedsize="false", width=0, height=0, shape=none];
 * AfterThreshold [label="", fixedsize="false", width=0, height=0, shape=none];
 * AfterConstantSource [label="", fixedsize="false", width=0, height=0, shape=none];
 * AfterDivide [label="", fixedsize="false", width=0, height=0, shape=none];
 *
 * Input1 -> AfterInput1 [arrowhead=none];
 * AfterInput1 -> Displaced;
 * Displaced -> Parker;
 * Parker -> FDK;
 * Input0 -> AfterInput0;
 * AfterInput0 -> FDK;
 * ConstantProjectionStack -> AfterConstantSource [arrowhead=none];
 * AfterConstantSource -> ForwardProjection;
 * AfterConstantSource -> RayBox;
 * RayBox -> Divide;
 * FDK -> Threshold;
 * Threshold -> AfterThreshold [arrowhead=none];
 * AfterThreshold -> ForwardProjection;
 * AfterInput1 -> Subtract;
 * ForwardProjection -> Subtract;
 * Subtract -> Multiply;
 * Multiply -> Divide;
 * AfterThreshold -> Output;
 * AfterThreshold -> AfterInput0 [style=dashed, constraint=false];
 * Divide -> AfterDivide;
 * AfterDivide -> AfterInput1 [style=dashed, constraint=false];
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
  typedef itk::ThresholdImageFilter<OutputImageType>                                                ThresholdFilterType;
  typedef itk::DivideOrZeroOutImageFilter<OutputImageType>                                          DivideFilterType;
  typedef rtk::RayBoxIntersectionImageFilter<OutputImageType, OutputImageType>                      RayBoxIntersectionFilterType;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(IterativeFDKConeBeamReconstructionFilter, IterativeConeBeamReconstructionFilter);

  /** Get / Set the object pointer to projection geometry */
  itkGetMacro(Geometry, ThreeDCircularProjectionGeometry::Pointer);
  itkSetMacro(Geometry, ThreeDCircularProjectionGeometry::Pointer);

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
  void SetForwardProjectionFilter (int _arg) ITK_OVERRIDE;

  /** Select the backprojection filter */
  void SetBackProjectionFilter (int _arg) ITK_OVERRIDE {}

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

  /** Set / Get whether the displaced detector filter should be disabled */
  itkSetMacro(DisableDisplacedDetectorFilter, bool)
  itkGetMacro(DisableDisplacedDetectorFilter, bool)

protected:
  IterativeFDKConeBeamReconstructionFilter();
  ~IterativeFDKConeBeamReconstructionFilter() {}

  void GenerateInputRequestedRegion() ITK_OVERRIDE;

  void GenerateOutputInformation() ITK_OVERRIDE;

  void GenerateData() ITK_OVERRIDE;

  /** The two inputs should not be in the same space so there is nothing
   * to verify. */
  void VerifyInputInformation() ITK_OVERRIDE {}

  /** Pointers to each subfilter of this composite filter */
  typename Superclass::ForwardProjectionPointerType m_ForwardProjectionFilter;
  typename DisplacedDetectorFilterType::Pointer     m_DisplacedDetectorFilter;
  typename ParkerFilterType::Pointer                m_ParkerFilter;
  typename FDKFilterType::Pointer                   m_FDKFilter;
  typename ThresholdFilterType::Pointer             m_ThresholdFilter;
  typename SubtractFilterType::Pointer              m_SubtractFilter;
  typename MultiplyFilterType::Pointer              m_MultiplyFilter;
  typename ConstantImageSourceType::Pointer         m_ConstantProjectionStackSource;
  typename DivideFilterType::Pointer                m_DivideFilter;
  typename RayBoxIntersectionFilterType::Pointer    m_RayBoxFilter;

  bool   m_EnforcePositivity;
  double m_TruncationCorrection;
  double m_HannCutFrequency;
  double m_HannCutFrequencyY;
  double m_ProjectionSubsetSize;
  bool   m_DisableDisplacedDetectorFilter;

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
#include "rtkIterativeFDKConeBeamReconstructionFilter.hxx"
#endif

#endif
