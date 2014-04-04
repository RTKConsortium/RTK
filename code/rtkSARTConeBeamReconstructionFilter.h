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

#ifndef __rtkSARTConeBeamReconstructionFilter_h
#define __rtkSARTConeBeamReconstructionFilter_h

#include "rtkBackProjectionImageFilter.h"
#include "rtkForwardProjectionImageFilter.h"

#include <itkExtractImageFilter.h>
#include <itkMultiplyImageFilter.h>
#include <itkSubtractImageFilter.h>
#include <itkDivideOrZeroOutImageFilter.h>
#include <itkTimeProbe.h>
#include <itkThresholdImageFilter.h>

#include "rtkRayBoxIntersectionImageFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkIterativeConeBeamReconstructionFilter.h"

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
 * 1 [ label="rtk::ForwardProjectionImageFilter" URL="\ref rtk::ForwardProjectionImageFilter"];
 * 2 [ label="itk::ExtractImageFilter" URL="\ref itk::ExtractImageFilter"];
 * 3 [ label="itk::MultiplyImageFilter (by zero)" URL="\ref itk::MultiplyImageFilter"];
 * test [label="", fixedsize="false", width=0, height=0, shape=none];
 * 4 [ label="itk::SubtractImageFilter" URL="\ref itk::SubtractImageFilter"];
 * 5 [ label="itk::MultiplyImageFilter (by lambda)" URL="\ref itk::MultiplyImageFilter"];
 * 6 [ label="itk::DivideOrZeroOutImageFilter" URL="\ref itk::DivideOrZeroOutImageFilter"];
 * 7 [ label="rtk::ConstantImageSource" URL="\ref rtk::ConstantImageSource"];
 * 8 [ label="itk::ExtractImageFilter" URL="\ref itk::ExtractImageFilter"];
 * 9 [ label="rtk::RayBoxIntersectionImageFilter" URL="\ref rtk::RayBoxIntersectionImageFilter"];
 * BackProjection [ label="rtk::BackProjectionImageFilter" URL="\ref rtk::BackProjectionImageFilter"];
 * OutofInput0 [label="", fixedsize="false", width=0, height=0, shape=none];
 * Threshold [ label="itk::ThresholdImageFilter" URL="\ref itk::ThresholdImageFilter"];
 * OutofThreshold [label="", fixedsize="false", width=0, height=0, shape=none];
 * Input0 -> OutofInput0 [arrowhead=None];
 * OutofInput0 -> 1;
 * OutofInput0 -> BackProjection;
 * 2 -> test[arrowhead=None];
 * test -> 3;
 * test -> 4;
 * 3 -> 1;
 * Input1 -> 2;
 * 1 -> 4;
 * 4 -> 5;
 * 5 -> 6;
 * 7 -> 8;
 * 8 -> 9;
 * 9 -> 6;
 * 6 -> BackProjection;
 * BackProjection -> Threshold;
 * Threshold -> OutofThreshold [arrowhead=None];
 * OutofThreshold -> OutofInput0 [style=dashed];
 * OutofThreshold -> Output;
 * }
 * \enddot
 *
 * \test rtksarttest.cxx
 *
 * \author Simon Rit
 *
 * \ingroup ReconstructionAlgorithm
 */
template<class TInputImage, class TOutputImage=TInputImage>
class ITK_EXPORT SARTConeBeamReconstructionFilter :
  public rtk::IterativeConeBeamReconstructionFilter<TInputImage, TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef SARTConeBeamReconstructionFilter                   Self;
  typedef itk::ImageToImageFilter<TInputImage, TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                            Pointer;
  typedef itk::SmartPointer<const Self>                      ConstPointer;

  /** Some convenient typedefs. */
  typedef TInputImage  InputImageType;
  typedef TOutputImage OutputImageType;

  /** Typedefs of each subfilter of this composite filter */
  typedef itk::ExtractImageFilter< InputImageType, InputImageType >                          ExtractFilterType;
  typedef itk::MultiplyImageFilter< OutputImageType, OutputImageType, OutputImageType >      MultiplyFilterType;
  typedef rtk::ForwardProjectionImageFilter< OutputImageType, OutputImageType >              ForwardProjectionFilterType;
  typedef itk::SubtractImageFilter< OutputImageType, OutputImageType >                       SubtractFilterType;
  typedef rtk::BackProjectionImageFilter< OutputImageType, OutputImageType >                 BackProjectionFilterType;
//  typedef typename BackProjectionFilterType::Pointer                                         BackProjectionFilterPointer;
  typedef rtk::RayBoxIntersectionImageFilter<OutputImageType, OutputImageType>               RayBoxIntersectionFilterType;
  typedef itk::DivideOrZeroOutImageFilter<OutputImageType, OutputImageType, OutputImageType> DivideFilterType;
  typedef rtk::ConstantImageSource<OutputImageType>                                          ConstantImageSourceType;
  typedef itk::ThresholdImageFilter<OutputImageType>                                         ThresholdFilterType;

/** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(SARTConeBeamReconstructionFilter, itk::ImageToImageFilter);

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
  void SetBackProjectionFilter (int _arg);

protected:
  SARTConeBeamReconstructionFilter();
  ~SARTConeBeamReconstructionFilter(){}

  virtual void GenerateInputRequestedRegion();

  virtual void GenerateOutputInformation();

  virtual void GenerateData();

  /** The two inputs should not be in the same space so there is nothing
   * to verify. */
  virtual void VerifyInputInformation() {}

  /** Pointers to each subfilter of this composite filter */
  typename ExtractFilterType::Pointer            m_ExtractFilter;
  typename ExtractFilterType::Pointer            m_ExtractFilterRayBox;
  typename MultiplyFilterType::Pointer           m_ZeroMultiplyFilter;
  typename ForwardProjectionFilterType::Pointer  m_ForwardProjectionFilter;
  typename SubtractFilterType::Pointer           m_SubtractFilter;
  typename MultiplyFilterType::Pointer           m_MultiplyFilter;
  typename BackProjectionFilterType::Pointer     m_BackProjectionFilter;
  typename RayBoxIntersectionFilterType::Pointer m_RayBoxFilter;
  typename DivideFilterType::Pointer             m_DivideFilter;
  typename ConstantImageSourceType::Pointer      m_ConstantImageSource;
  typename ThresholdFilterType::Pointer          m_ThresholdFilter;

  bool m_EnforcePositivity;

private:
  //purposely not implemented
  SARTConeBeamReconstructionFilter(const Self&);
  void operator=(const Self&);

  /** Geometry object */
  ThreeDCircularProjectionGeometry::Pointer m_Geometry;

  /** Number of iterations */
  unsigned int m_NumberOfIterations;

  /** Convergence factor according to Andersen's publications which relates
   * to the step size of the gradient descent. Default 0.3, Must be in (0,2). */
  double m_Lambda;

  /** Internal variables storing the current forward
    and back projection methods */
  int m_CurrentForwardProjectionConfiguration;
  int m_CurrentBackProjectionConfiguration;

  /** Time probes */
  itk::TimeProbe m_ExtractProbe;
  itk::TimeProbe m_ZeroMultiplyProbe;
  itk::TimeProbe m_ForwardProjectionProbe;
  itk::TimeProbe m_SubtractProbe;
  itk::TimeProbe m_MultiplyProbe;
  itk::TimeProbe m_RayBoxProbe;
  itk::TimeProbe m_DivideProbe;
  itk::TimeProbe m_BackProjectionProbe;
  itk::TimeProbe m_ThresholdProbe;

}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkSARTConeBeamReconstructionFilter.txx"
#endif

#endif
