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

#ifndef rtkMotionCompensatedFourDConjugateGradientConeBeamReconstructionFilter_h
#define rtkMotionCompensatedFourDConjugateGradientConeBeamReconstructionFilter_h

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

template< typename VolumeSeriesType, typename ProjectionStackType>
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
  typedef itk::CovariantVector< typename VolumeSeriesType::ValueType, VolumeSeriesType::ImageDimension - 1>   VectorForDVF;

#ifdef RTK_USE_CUDA
  typedef itk::CudaImage<VectorForDVF, VolumeSeriesType::ImageDimension>          DVFSequenceImageType;
  typedef itk::CudaImage<VectorForDVF, VolumeSeriesType::ImageDimension - 1>      DVFImageType;
#else
  typedef itk::Image<VectorForDVF, VolumeSeriesType::ImageDimension>              DVFSequenceImageType;
  typedef itk::Image<VectorForDVF, VolumeSeriesType::ImageDimension - 1>          DVFImageType;
#endif

  /** Typedefs of each subfilter of this composite filter */

  /** Standard New method. */
  itkNewMacro(Self)

  /** Runtime information support. */
  itkTypeMacro(MotionCompensatedFourDConjugateGradientConeBeamReconstructionFilter, FourDConjugateGradientConeBeamReconstructionFilter)

  /** Neither the Forward nor the Back projection filters can be set by the user */
  void SetForwardProjectionFilter (int _arg) ITK_OVERRIDE {}
  void SetBackProjectionFilter (int _arg) ITK_OVERRIDE {}

  /** The ND + time motion vector field */
  void SetDisplacementField(const DVFSequenceImageType* DVFs);
  void SetInverseDisplacementField(const DVFSequenceImageType* DVFs);
  typename DVFSequenceImageType::ConstPointer GetDisplacementField();
  typename DVFSequenceImageType::ConstPointer GetInverseDisplacementField();

  /** Set the vector containing the signal in the sub-filters */
  void SetSignal(const std::vector<double> signal) ITK_OVERRIDE;

  // Sub filters typedefs
  typedef rtk::WarpProjectionStackToFourDImageFilter< VolumeSeriesType, ProjectionStackType>                        MCProjStackToFourDType;
  typedef rtk::MotionCompensatedFourDReconstructionConjugateGradientOperator<VolumeSeriesType, ProjectionStackType> MCCGOperatorType;

  /** Set and Get for the UseCudaCyclicDeformation variable */
  itkSetMacro(UseCudaCyclicDeformation, bool)
  itkGetMacro(UseCudaCyclicDeformation, bool)

protected:
  MotionCompensatedFourDConjugateGradientConeBeamReconstructionFilter();
  ~MotionCompensatedFourDConjugateGradientConeBeamReconstructionFilter() {}

  void GenerateOutputInformation() ITK_OVERRIDE;
  void GenerateInputRequestedRegion() ITK_OVERRIDE;

  bool                                                m_UseCudaCyclicDeformation;

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
