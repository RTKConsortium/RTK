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

#ifndef __rtkWarpForwardProjectSequenceImageFilter_h
#define __rtkWarpForwardProjectSequenceImageFilter_h

#include "rtkConstantImageSource.h"
#include "rtkCyclicDeformationImageFilter.h"
#include "rtkJosephForwardProjectionImageFilter.h"
#include "rtkInterpolatorWithKnownWeightsImageFilter.h"

#include <itkPasteImageFilter.h>
#include <itkNearestNeighborInterpolateImageFunction.h>

#ifdef RTK_USE_CUDA
  #include "rtkCudaWarpedForwardProjectionImageFilter.h"
  #include "rtkCudaInterpolateImageFilter.h"
#endif

namespace rtk
{
  /** \class WarpForwardProjectSequenceImageFilter
   * \brief Applies an N-D + time Motion Vector Field to an N-D + time sequence of images,
   * and forward projects through it. Combines the warping and the forward projection so as
   * to perform only one interpolation in space, thus avoiding some of the loss of resolution.
   *
   * Most of the work in this filter is performed by rtkCudaWarpedForwardProjectionImageFilter.
   *
   * \dot
   * digraph WarpForwardProjectSequenceImageFilter {
   *
   * Input0 [label="Input 0 (Projection stack)"];
   * Input0 [shape=Mdiamond];
   * Input1 [label="Input 1 (Sequence of images)"];
   * Input1 [shape=Mdiamond];
   * Input2 [label="Input 2 (Sequence of DVFs)"];
   * Input2 [shape=Mdiamond];
   * Output [label="Output (Projection stack)"];
   * Output [shape=Mdiamond];
   *
   * node [shape=box];forwardProjection
   * Interpolator [label="rtk::rtkInterpolatorWithKnownWeightsImageFilter" URL="\ref rtk::rtkInterpolatorWithKnownWeightsImageFilter"];
   * CyclicDeformation [label="rtk::CyclicDeformationImageFilter (for DVFs)" URL="\ref rtk::CyclicDeformationImageFilter"];
   * SingleProjectionSource [ label="rtk::ConstantImageSource (single projection)" URL="\ref rtk::ConstantImageSource"];
   * WarpForwardProject [ label="rtk::rtkCudaWarpedForwardProjectionImageFilter" URL="\ref rtk::rtkCudaWarpedForwardProjectionImageFilter"];
   * ProjectionStackSource [ label="rtk::ConstantImageSource (projection stack)" URL="\ref rtk::ConstantImageSource"];
   * VolumeSource [ label="rtk::ConstantImageSource (volume)" URL="\ref rtk::ConstantImageSource"];
   * Paste [ label="itk::PasteImageFilter" URL="\ref itk::PasteImageFilter"];
   * BeforePaste [label="", fixedsize="false", width=0, height=0, shape=none];
   * AfterPaste [label="", fixedsize="false", width=0, height=0, shape=none];
   *
   * Input1 -> Interpolator;
   * VolumeSource -> Interpolator;
   * Input2 -> CyclicDeformation;
   * SingleProjectionSource -> WarpForwardProject;
   * Interpolator -> WarpForwardProject;
   * CyclicDeformation -> WarpForwardProject;
   * WarpForwardProject -> Paste;
   * ProjectionStackSource -> BeforePaste;
   * BeforePaste -> Paste;
   * Paste -> AfterPaste [arrowhead=none];
   * AfterPaste -> BeforePaste [style=dashed, constraint=false];
   * AfterPaste -> Output [style=dashed];
   * }
   * \enddot
   *
   * \test ??
   *
   * \author Cyril Mory
   *
   * \ingroup ReconstructionAlgorithm
   */

template< typename TImageSequence,
          typename TMVFImageSequence = itk::Image< itk::CovariantVector < typename TImageSequence::ValueType,
                                                                          TImageSequence::ImageDimension-1 >,
                                                   TImageSequence::ImageDimension >,
          typename TImage = itk::Image< typename TImageSequence::ValueType,
                                        TImageSequence::ImageDimension-1 >,
          typename TMVFImage = itk::Image<itk::CovariantVector < typename TImageSequence::ValueType,
                                                                 TImageSequence::ImageDimension - 1 >,
                                          TImageSequence::ImageDimension - 1> >
class WarpForwardProjectSequenceImageFilter : public itk::ImageToImageFilter<TImage, TImage>
{
public:
    /** Standard class typedefs. */
    typedef WarpForwardProjectSequenceImageFilter         Self;
    typedef itk::ImageToImageFilter<TImage, TImage>       Superclass;
    typedef itk::SmartPointer< Self >                     Pointer;
    typedef TImage                                        ProjectionStackType;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(WarpForwardProjectSequenceImageFilter, IterativeConeBeamReconstructionFilter)

    /** The stack of measured projections to be computed */
    void SetInputProjectionStack(const ProjectionStackType* Projection);
    typename ProjectionStackType::Pointer   GetInputProjectionStack();

    /** The ND + time sequence to forward project */
    void SetInputVolumeSeries(const TImageSequence* VolumeSeries);
    typename TImageSequence::Pointer GetInputVolumeSeries();

    /** The ND + time motion vector field */
    void SetDisplacementField(const TMVFImageSequence* MVFs);
    typename TMVFImageSequence::Pointer GetDisplacementField();

    /** The file containing the phase at which each projection has been acquired */
    itkGetMacro(SignalFilename, std::string);
    virtual void SetSignalFilename (const std::string _arg);

    /** Pass the interpolation weights to the interpolator */
    virtual void SetWeights(const itk::Array2D<float> _arg);

    /** Projection geometry */
    itkSetMacro(Geometry, typename ThreeDCircularProjectionGeometry::Pointer);
    itkGetMacro(Geometry, typename ThreeDCircularProjectionGeometry::Pointer);

    /** Typedefs of internal filters */
#ifdef RTK_USE_CUDA
    typedef rtk::CudaWarpedForwardProjectionImageFilter                                       CudaWarpForwardProjectFilterType;
    typedef rtk::CudaInterpolateImageFilter                                                   CudaInterpolateFilterType;
#endif
    typedef rtk::JosephForwardProjectionImageFilter<ProjectionStackType,ProjectionStackType>  JosephForwardProjectFilterType;
    typedef rtk::InterpolatorWithKnownWeightsImageFilter<TImage, TImageSequence>              InterpolateFilterType;
    typedef rtk::CyclicDeformationImageFilter<TMVFImage>                                      MVFInterpolatorType;
    typedef itk::PasteImageFilter<ProjectionStackType,ProjectionStackType>                    PasteFilterType;
    typedef rtk::ConstantImageSource<ProjectionStackType>                                     ConstantImageSourceType;

protected:
    WarpForwardProjectSequenceImageFilter();
    ~WarpForwardProjectSequenceImageFilter(){}

    /** Does the real work. */
    virtual void GenerateData();

    /** Member pointers to the filters used internally (for convenience) */
    typename InterpolateFilterType::Pointer               m_InterpolatorFilter;
    typename MVFInterpolatorType::Pointer                 m_MVFInterpolatorFilter;
    typename PasteFilterType::Pointer                     m_PasteFilter;
    typename ConstantImageSourceType::Pointer             m_SingleProjectionSource;
    typename ConstantImageSourceType::Pointer             m_ProjectionStackSource;
    typename ConstantImageSourceType::Pointer             m_VolumeSource;
#ifdef RTK_USE_CUDA
    typename CudaWarpForwardProjectFilterType::Pointer    m_ForwardProjectFilter;
#else
    typename JosephForwardProjectFilterType::Pointer      m_ForwardProjectFilter;
#endif

    /** Region for the paste filter and the single projection source */
    typename ProjectionStackType::RegionType                  m_SingleProjectionRegion;
    typename rtk::ThreeDCircularProjectionGeometry::Pointer   m_Geometry;
    std::string                                               m_SignalFilename;
    std::vector<double>                                       m_Signal;

    /** The inputs of this filter do not occupy the same physical space. Therefore this check
    * must be removed */
    void VerifyInputInformation(){}

    /** The volume and the projections must have different requested regions */
    void GenerateOutputInformation();
    void GenerateInputRequestedRegion();

private:
    WarpForwardProjectSequenceImageFilter(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented
};
} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkWarpForwardProjectSequenceImageFilter.hxx"
#endif

#endif
