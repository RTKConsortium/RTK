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
#ifndef __rtkWarpProjectionStackToFourDImageFilter_h
#define __rtkWarpProjectionStackToFourDImageFilter_h

#include "rtkCyclicDeformationImageFilter.h"
#include "rtkProjectionStackToFourDImageFilter.h"

#ifdef RTK_USE_CUDA
  #include "rtkCudaWarpBackProjectionImageFilter.h"
#endif

namespace rtk
{
  /** \class WarpProjectionStackToFourDImageFilter
   * \brief Back projection part for motion compensated iterative 4D reconstruction
   *
   * This filter is similar to ProjectionStackToFourDImageFilter, except
   * that it uses a motion-compensated backprojection. A 4D displacement
   * vector field is therefore required, and its back projection filter
   * cannot be changed.
   *
   * \dot
   * digraph WarpProjectionStackToFourDImageFilter {
   *
   * Input0 [label="Input 0 (4D sequence of volumes)"];
   * Input0 [shape=Mdiamond];
   * Input1 [label="Input 1 (Projections)"];
   * Input1 [shape=Mdiamond];
   * Input2 [label="Input 2 (4D Sequence of DVFs)"];
   * Input2 [shape=Mdiamond];
   * Output [label="Output (4D sequence of volumes)"];
   * Output [shape=Mdiamond];
   *
   * node [shape=box];
   * Extract [ label="itk::ExtractImageFilter" URL="\ref itk::ExtractImageFilter"];
   * DisplacedDetector [ label="rtk::DisplacedDetectorImageFilter" URL="\ref rtk::DisplacedDetectorImageFilter"];
   * VolumeSeriesSource [ label="rtk::ConstantImageSource (4D)" URL="\ref rtk::ConstantImageSource"];
   * AfterSource4D [label="", fixedsize="false", width=0, height=0, shape=none];
   * Source [ label="rtk::ConstantImageSource" URL="\ref rtk::ConstantImageSource"];
   * CyclicDeformation [label="rtk::CyclicDeformationImageFilter (for DVFs)" URL="\ref rtk::CyclicDeformationImageFilter"];
   * Backproj [ label="rtk::CudaWarpBackProjectionImageFilter" URL="\ref rtk::CudaWarpBackProjectionImageFilter"];
   * Splat [ label="rtk::SplatWithKnownWeightsImageFilter" URL="\ref rtk::SplatWithKnownWeightsImageFilter"];
   * AfterSplat [label="", fixedsize="false", width=0, height=0, shape=none];
   *
   * Input1 -> Extract;
   * Input0 -> VolumeSeriesSource [style=invis];
   * VolumeSeriesSource -> AfterSource4D[arrowhead=none];
   * AfterSource4D -> Splat;
   * Extract -> DisplacedDetector;
   * DisplacedDetector -> Backproj;
   * CyclicDeformation -> Backproj;
   * Source -> Backproj;
   * Backproj -> Splat;
   * Splat -> AfterSplat[arrowhead=none];
   * AfterSplat -> Output;
   * AfterSplat -> AfterSource4D[style=dashed, constraint=none];
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
          typename TFFTPrecision=double,
          typename TMVFImageSequence = itk::Image< itk::CovariantVector < typename VolumeSeriesType::ValueType, VolumeSeriesType::ImageDimension-1 >, VolumeSeriesType::ImageDimension >,
          typename TMVFImage = itk::Image< itk::CovariantVector < typename VolumeSeriesType::ValueType, VolumeSeriesType::ImageDimension-1 >, VolumeSeriesType::ImageDimension -1 >
          >
class WarpProjectionStackToFourDImageFilter : public rtk::ProjectionStackToFourDImageFilter< VolumeSeriesType, ProjectionStackType, TFFTPrecision>
{
public:
    /** Standard class typedefs. */
    typedef WarpProjectionStackToFourDImageFilter                         Self;
    typedef rtk::ProjectionStackToFourDImageFilter< VolumeSeriesType,
                                                    ProjectionStackType,
                                                    TFFTPrecision>        Superclass;
    typedef itk::SmartPointer< Self >                                     Pointer;

    /** Convenient typedefs */
    typedef ProjectionStackType VolumeType;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(WarpProjectionStackToFourDImageFilter, rtk::ProjectionStackToFourDImageFilter)

    typedef rtk::CyclicDeformationImageFilter<TMVFImage>                  MVFInterpolatorType;

    /** The back projection filter cannot be set by the user */
    void SetBackProjectionFilter (const typename BackProjectionFilterType::Pointer _arg) {}

    /** The ND + time motion vector field */
    void SetDisplacementField(const TMVFImageSequence* MVFs);
    typename TMVFImageSequence::Pointer GetDisplacementField();

    /** The file containing the phase at which each projection has been acquired */
    itkGetMacro(SignalFilename, std::string)
    virtual void SetSignalFilename (const std::string _arg);

protected:
    WarpProjectionStackToFourDImageFilter();
    ~WarpProjectionStackToFourDImageFilter(){}

    /** Does the real work. */
    virtual void GenerateData();

    virtual void GenerateOutputInformation();

    virtual void SetSignalFilename(const std::string _arg);

    /** Member pointers to the filters used internally (for convenience)*/
    typename MVFInterpolatorType::Pointer           m_MVFInterpolatorFilter;
    std::string                                     m_SignalFilename;
    std::vector<double>                             m_Signal;

private:
    WarpProjectionStackToFourDImageFilter(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented

};
} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkWarpProjectionStackToFourDImageFilter.hxx"
#endif

#endif
