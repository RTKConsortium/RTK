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

#ifndef __rtkMotionCompensatedFourDConjugateGradientConeBeamReconstructionFilter_h
#define __rtkMotionCompensatedFourDConjugateGradientConeBeamReconstructionFilter_h

#include "rtkFourDConjugateGradientConeBeamReconstructionFilter.h"
#include "rtkMotionCompensatedFourDReconstructionConjugateGradientOperator.h"
#include "rtkWarpProjectionStackToFourDImageFilter.h"
#ifdef RTK_USE_CUDA
  #include "rtkCudaWarpForwardProjectionImageFilter.h"
  #include "rtkCudaWarpBackProjectionImageFilter.h"
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
   * \ingroup ReconstructionAlgorithm
   */

template< typename VolumeSeriesType,
          typename ProjectionStackType = itk::Image<typename VolumeSeriesType::ValueType, VolumeSeriesType::ImageDimension-1>,
          typename TMVFImageSequence = itk::Image< itk::CovariantVector < typename VolumeSeriesType::ValueType, VolumeSeriesType::ImageDimension-1 >, VolumeSeriesType::ImageDimension >,
          typename TMVFImage = itk::Image< itk::CovariantVector < typename VolumeSeriesType::ValueType, VolumeSeriesType::ImageDimension-1 >, VolumeSeriesType::ImageDimension -1 > >
class ITK_EXPORT MotionCompensatedFourDConjugateGradientConeBeamReconstructionFilter :
  public rtk::FourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
{
public:
  /** Standard class typedefs. */
  typedef MotionCompensatedFourDConjugateGradientConeBeamReconstructionFilter                           Self;
  typedef FourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>     Superclass;
  typedef itk::SmartPointer<Self>                                                                       Pointer;
  typedef itk::SmartPointer<const Self>                                                                 ConstPointer;

  /** Some convenient typedefs. */
  typedef VolumeSeriesType      InputImageType;
  typedef VolumeSeriesType      OutputImageType;
  typedef ProjectionStackType   VolumeType;

  /** Typedefs of each subfilter of this composite filter */

  /** Standard New method. */
  itkNewMacro(Self)

  /** Runtime information support. */
  itkTypeMacro(MotionCompensatedFourDConjugateGradientConeBeamReconstructionFilter, FourDConjugateGradientConeBeamReconstructionFilter)

  /** Pass the ForwardProjection filter to the conjugate gradient operator */
  void SetForwardProjectionFilter (int _arg) {}
  void SetBackProjectionFilter (int _arg) {}
//#ifdef RTK_USE_CUDA
//  virtual typename rtk::CudaWarpForwardProjectionImageFilter* GetForwardProjectionFilter();
//  virtual typename rtk::CudaWarpBackProjectionImageFilter* GetBackProjectionFilter();
//#endif

  /** The ND + time motion vector field */
  void SetDisplacementField(const TMVFImageSequence* MVFs);
  void SetInverseDisplacementField(const TMVFImageSequence* MVFs);
  typename TMVFImageSequence::ConstPointer GetDisplacementField();
  typename TMVFImageSequence::ConstPointer GetInverseDisplacementField();

  /** The file containing the phase at which each projection has been acquired */
  itkGetMacro(SignalFilename, std::string)
  virtual void SetSignalFilename (const std::string _arg);

  typedef rtk::WarpProjectionStackToFourDImageFilter< VolumeSeriesType, ProjectionStackType, TMVFImageSequence, TMVFImage>  MCProjStackToFourDType;
  typedef rtk::MotionCompensatedFourDReconstructionConjugateGradientOperator<VolumeSeriesType, ProjectionStackType, TMVFImageSequence, TMVFImage> MCCGOperatorType;

  virtual MCProjStackToFourDType* GetProjectionStackToFourDFilter();
  virtual MCCGOperatorType* GetConjugateGradientOperator();

protected:
  MotionCompensatedFourDConjugateGradientConeBeamReconstructionFilter();
  ~MotionCompensatedFourDConjugateGradientConeBeamReconstructionFilter(){}

  virtual void GenerateOutputInformation();

//  virtual void GenerateInputRequestedRegion();

//  virtual void GenerateData();

//  /** The inputs should not be in the same space so there is nothing
//   * to verify. */
//  virtual void VerifyInputInformation() {}

  typename MCProjStackToFourDType::Pointer        m_ProjStackToFourDFilter;
  typename MCCGOperatorType::Pointer              m_CGOperator;
  std::string                                     m_SignalFilename;
  std::vector<double>                             m_Signal;
//#ifdef RTK_USE_CUDA
//  rtk::CudaWarpForwardProjectionImageFilter::Pointer  m_ForwardProjectionFilter;
//  rtk::CudaWarpBackProjectionImageFilter::Pointer     m_BackProjectionFilter;
//#endif


private:
  //purposely not implemented
  MotionCompensatedFourDConjugateGradientConeBeamReconstructionFilter(const Self&);
  void operator=(const Self&);
}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkMotionCompensatedFourDConjugateGradientConeBeamReconstructionFilter.hxx"
#endif

#endif
