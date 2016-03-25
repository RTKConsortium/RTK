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
#ifndef __rtkMotionCompensatedFourDReconstructionConjugateGradientOperator_h
#define __rtkMotionCompensatedFourDReconstructionConjugateGradientOperator_h

#include "rtkFourDReconstructionConjugateGradientOperator.h"
#include "rtkCyclicDeformationImageFilter.h"

#ifdef RTK_USE_CUDA
  #include "rtkCudaWarpForwardProjectionImageFilter.h"
  #include "rtkCudaWarpBackProjectionImageFilter.h"
#endif

namespace rtk
{
  /** \class MotionCompensatedFourDReconstructionConjugateGradientOperator
   * \brief Like FourDReconstructionConjugateGradientOperator, but motion-compensated
   *
   * \dot
   * digraph MotionCompensatedFourDReconstructionConjugateGradientOperator {
   *
   * Input0 [ label="Input 0 (Input: 4D sequence of volumes)"];
   * Input0 [shape=Mdiamond];
   * Input1 [label="Input 1 (Projections)"];
   * Input1 [shape=Mdiamond];
   * Input2 [label="Input 2 (4D Sequence of DVFs)"];
   * Input2 [shape=Mdiamond];
   * Output [label="Output (Reconstruction: 4D sequence of volumes)"];
   * Output [shape=Mdiamond];
   *
   * node [shape=box];
   * SourceVol [ label="rtk::ConstantImageSource (volume)" URL="\ref rtk::ConstantImageSource"];
   * SourceVol2 [ label="rtk::ConstantImageSource (volume)" URL="\ref rtk::ConstantImageSource"];
   * SourceProj [ label="rtk::ConstantImageSource (projections)" URL="\ref rtk::ConstantImageSource"];
   * Source4D [ label="rtk::ConstantImageSource (4D)" URL="\ref rtk::ConstantImageSource"];
   * CyclicDeformation [label="rtk::CyclicDeformationImageFilter (for DVFs)" URL="\ref rtk::CyclicDeformationImageFilter"];
   * ForwardProj [ label="rtk::ForwardProjectionImageFilter" URL="\ref rtk::ForwardProjectionImageFilter"];
   * Interpolation [ label="InterpolatorWithKnownWeightsImageFilter" URL="\ref rtk::InterpolatorWithKnownWeightsImageFilter"];
   * BackProj [ label="rtk::BackProjectionImageFilter" URL="\ref rtk::BackProjectionImageFilter"];
   * Splat [ label="rtk::SplatWithKnownWeightsImageFilter" URL="\ref rtk::SplatWithKnownWeightsImageFilter"];
   * AfterSplat [label="", fixedsize="false", width=0, height=0, shape=none];
   * AfterInput0 [label="", fixedsize="false", width=0, height=0, shape=none];
   * AfterSource4D [label="", fixedsize="false", width=0, height=0, shape=none];
   * Displaced [ label="rtk::DisplacedDetectorImageFilter" URL="\ref rtk::DisplacedDetectorImageFilter"];
   *
   * Input2 -> CyclicDeformation;
   * CyclicDeformation -> ForwardProj;
   * CyclicDeformation -> BackProj;
   * Input0 -> Interpolation;
   * SourceVol -> Interpolation;
   * Interpolation -> ForwardProj;
   * SourceVol2 -> BackProj;
   * ForwardProj -> Displaced;
   * Displaced -> BackProj;
   * BackProj -> Splat;
   * Splat -> AfterSplat[arrowhead=none];
   * AfterSplat -> Output;
   * AfterSplat -> AfterSource4D[style=dashed, constraint=false];
   * Source4D -> AfterSource4D[arrowhead=none];
   * AfterSource4D -> Splat;
   * SourceProj -> ForwardProj;
   * }
   * \enddot
   *
   * \test rtkfourdconjugategradienttest.cxx
   *
   * \author Cyril Mory
   *
   * \ingroup ReconstructionAlgorithm
   */

template< typename VolumeSeriesType,
          typename ProjectionStackType = itk::Image<typename VolumeSeriesType::ValueType, VolumeSeriesType::ImageDimension-1>,
          typename TMVFImageSequence = itk::Image< itk::CovariantVector < typename VolumeSeriesType::ValueType, VolumeSeriesType::ImageDimension-1 >, VolumeSeriesType::ImageDimension >,
          typename TMVFImage = itk::Image< itk::CovariantVector < typename VolumeSeriesType::ValueType, VolumeSeriesType::ImageDimension-1 >, VolumeSeriesType::ImageDimension -1 > >
class MotionCompensatedFourDReconstructionConjugateGradientOperator : public FourDReconstructionConjugateGradientOperator< VolumeSeriesType, ProjectionStackType>
{
public:
    /** Standard class typedefs. */
    typedef MotionCompensatedFourDReconstructionConjugateGradientOperator                                      Self;
    typedef FourDReconstructionConjugateGradientOperator< VolumeSeriesType, ProjectionStackType>  Superclass;
    typedef itk::SmartPointer< Self >                                                             Pointer;

  /** Convenient typedef */
    typedef ProjectionStackType                                 VolumeType;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(MotionCompensatedFourDReconstructionConjugateGradientOperator, FourDReconstructionConjugateGradientOperator)

    typedef rtk::CyclicDeformationImageFilter<TMVFImage>                  MVFInterpolatorType;

    /** The forward and back projection filters cannot be set by the user */
    void SetForwardProjectionFilter (const typename Superclass::ForwardProjectionFilterType::Pointer _arg) {}
    void SetBackProjectionFilter (const typename Superclass::BackProjectionFilterType::Pointer _arg) {}

#ifdef RTK_USE_CUDA
    virtual typename rtk::CudaWarpForwardProjectionImageFilter* GetForwardProjectionFilter();
    virtual typename rtk::CudaWarpBackProjectionImageFilter* GetBackProjectionFilter();
#endif

    /** The ND + time motion vector field */
    void SetDisplacementField(const TMVFImageSequence* MVFs);
    void SetInverseDisplacementField(const TMVFImageSequence* MVFs);
    typename TMVFImageSequence::ConstPointer GetInverseDisplacementField();
    typename TMVFImageSequence::ConstPointer GetDisplacementField();

    /** The file containing the phase at which each projection has been acquired */
    itkGetMacro(SignalFilename, std::string)
    virtual void SetSignalFilename (const std::string _arg);

protected:
    MotionCompensatedFourDReconstructionConjugateGradientOperator();
    ~MotionCompensatedFourDReconstructionConjugateGradientOperator(){}

    /** Builds the pipeline and computes output information */
    virtual void GenerateOutputInformation();

    /** Computes the requested region of input images */
    virtual void GenerateInputRequestedRegion();

    /** Does the real work. */
    virtual void GenerateData();

    /** Member pointers to the filters used internally (for convenience)*/
    typename MVFInterpolatorType::Pointer               m_MVFInterpolatorFilter;
    typename MVFInterpolatorType::Pointer               m_InverseMVFInterpolatorFilter;
    std::string                                         m_SignalFilename;
    std::vector<double>                                 m_Signal;
#ifdef RTK_USE_CUDA
    rtk::CudaWarpForwardProjectionImageFilter::Pointer  m_ForwardProjectionFilter;
    rtk::CudaWarpBackProjectionImageFilter::Pointer m_BackProjectionFilter;
#endif

private:
    MotionCompensatedFourDReconstructionConjugateGradientOperator(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented
};
} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkMotionCompensatedFourDReconstructionConjugateGradientOperator.hxx"
#endif

#endif
