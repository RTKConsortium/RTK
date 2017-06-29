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
#ifndef rtkWarpProjectionStackToFourDImageFilter_h
#define rtkWarpProjectionStackToFourDImageFilter_h

#include "rtkCyclicDeformationImageFilter.h"
#include "rtkProjectionStackToFourDImageFilter.h"

#ifdef RTK_USE_CUDA
  #include "rtkCudaWarpBackProjectionImageFilter.h"
  #include "rtkCudaCyclicDeformationImageFilter.h"
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
   * Input2 -> CyclicDeformation;
   * VolumeSeriesSource -> AfterSource4D[arrowhead=none];
   * AfterSource4D -> Splat;
   * Extract -> Backproj;
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

template< typename VolumeSeriesType, typename ProjectionStackType>
class WarpProjectionStackToFourDImageFilter : public rtk::ProjectionStackToFourDImageFilter< VolumeSeriesType, ProjectionStackType>
{
public:
    /** Standard class typedefs. */
    typedef WarpProjectionStackToFourDImageFilter                         Self;
    typedef rtk::ProjectionStackToFourDImageFilter< VolumeSeriesType,
                                                    ProjectionStackType>  Superclass;
    typedef itk::SmartPointer< Self >                                     Pointer;

    /** Convenient typedefs */
    typedef ProjectionStackType VolumeType;
    typedef itk::CovariantVector< typename VolumeSeriesType::ValueType, VolumeSeriesType::ImageDimension - 1>   VectorForDVF;

#ifdef RTK_USE_CUDA
    typedef itk::CudaImage<VectorForDVF, VolumeSeriesType::ImageDimension>          DVFSequenceImageType;
    typedef itk::CudaImage<VectorForDVF, VolumeSeriesType::ImageDimension - 1>      DVFImageType;
#else
    typedef itk::Image<VectorForDVF, VolumeSeriesType::ImageDimension>              DVFSequenceImageType;
    typedef itk::Image<VectorForDVF, VolumeSeriesType::ImageDimension - 1>          DVFImageType;
#endif

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(WarpProjectionStackToFourDImageFilter, rtk::ProjectionStackToFourDImageFilter)

    typedef rtk::CyclicDeformationImageFilter<DVFImageType>                  DVFInterpolatorType;
    typedef std::vector<double>                                              SignalVectorType;

    /** The back projection filter cannot be set by the user */
    void SetBackProjectionFilter (const typename Superclass::BackProjectionFilterType::Pointer _arg) {}

    /** The ND + time motion vector field */
    void SetDisplacementField(const DVFSequenceImageType* DVFs);
    typename DVFSequenceImageType::ConstPointer GetDisplacementField();

    void SetSignal(const std::vector<double> signal) ITK_OVERRIDE;

    /** Set and Get for the UseCudaCyclicDeformation variable */
    itkSetMacro(UseCudaCyclicDeformation, bool)
    itkGetMacro(UseCudaCyclicDeformation, bool)

protected:
    WarpProjectionStackToFourDImageFilter();
    ~WarpProjectionStackToFourDImageFilter() {}

    /** Does the real work. */
    void GenerateData() ITK_OVERRIDE;

    void GenerateOutputInformation() ITK_OVERRIDE;

    /** The first two inputs should not be in the same space so there is nothing
     * to verify. */
    void VerifyInputInformation() ITK_OVERRIDE {}

    /** Member pointers to the filters used internally (for convenience)*/
    typename DVFInterpolatorType::Pointer           m_DVFInterpolatorFilter;
    std::vector<double>                             m_Signal;
    bool                                            m_UseCudaCyclicDeformation;

private:
    WarpProjectionStackToFourDImageFilter(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented

};
} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkWarpProjectionStackToFourDImageFilter.hxx"
#endif

#endif
