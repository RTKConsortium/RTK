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
#ifndef rtkWarpFourDToProjectionStackImageFilter_h
#define rtkWarpFourDToProjectionStackImageFilter_h

#include "rtkFourDToProjectionStackImageFilter.h"
#include "rtkCyclicDeformationImageFilter.h"
#include "rtkJosephForwardProjectionImageFilter.h"
#include <vector>

#ifdef RTK_USE_CUDA
  #include "rtkCudaWarpForwardProjectionImageFilter.h"
  #include "rtkCudaCyclicDeformationImageFilter.h"
#endif

namespace rtk
{
  /** \class WarpFourDToProjectionStackImageFilter
   * \brief Forward projection part for motion compensated iterative 4D reconstruction
   *
   * This filter is similar to FourDToProjectionStackImageFilter, except
   * that it uses a motion-compensated forward projection. A 4D displacement
   * vector field is therefore required, and its forward projection filter
   * cannot be changed.
   *
   * \dot
   * digraph WarpFourDToProjectionStackImageFilter {
   *
   * Input0 [ label="Input 0 (Projections)"];
   * Input0 [shape=Mdiamond];
   * Input1 [label="Input 1 (Input: 4D sequence of volumes)"];
   * Input1 [shape=Mdiamond];
   * Input2 [label="Input 2 (4D Sequence of DVFs)"];
   * Input2 [shape=Mdiamond];
   * Output [label="Output (Output projections)"];
   * Output [shape=Mdiamond];
   *
   * node [shape=box];
   * FourDSource [ label="rtk::ConstantImageSource (4D volume)" URL="\ref rtk::ConstantImageSource"];
   * ProjectionSource [ label="rtk::ConstantImageSource (projections)" URL="\ref rtk::ConstantImageSource"];
   * ForwardProj [ label="rtk::ForwardProjectionImageFilter" URL="\ref rtk::ForwardProjectionImageFilter"];
   * Interpolation [ label="rtk::InterpolatorWithKnownWeightsImageFilter" URL="\ref rtk::InterpolatorWithKnownWeightsImageFilter"];
   * CyclicDeformation [label="rtk::CyclicDeformationImageFilter (for DVFs)" URL="\ref rtk::CyclicDeformationImageFilter"];
   * BeforePaste [label="", fixedsize="false", width=0, height=0, shape=none];
   * Paste [ label="itk::PasteImageFilter" URL="\ref itk::PasteImageFilter"];
   * AfterPaste [label="", fixedsize="false", width=0, height=0, shape=none];
   *
   * Input2 -> CyclicDeformation;
   * ProjectionSource -> ForwardProj;
   * BeforePaste -> Paste;
   * FourDSource -> Interpolation;
   * Input1 -> Interpolation;
   * Interpolation -> ForwardProj;
   * CyclicDeformation -> ForwardProj;
   * Input0 -> BeforePaste[arrowhead=none];
   * ForwardProj -> Paste;
   * Paste -> AfterPaste[arrowhead=none];
   * AfterPaste -> Output;
   * AfterPaste -> BeforePaste [style=dashed, constraint=false];
   * }
   * \enddot
   *
   * \test rtkwarpfourdtoprojectionstacktest.cxx
   *
   * \author Cyril Mory
   *
   * \ingroup ReconstructionAlgorithm
   */

  template< typename VolumeSeriesType, typename ProjectionStackType>
class WarpFourDToProjectionStackImageFilter : public rtk::FourDToProjectionStackImageFilter<ProjectionStackType, VolumeSeriesType>
{
public:
    /** Standard class typedefs. */
    typedef WarpFourDToProjectionStackImageFilter                         Self;
    typedef rtk::FourDToProjectionStackImageFilter< ProjectionStackType,
                                                    VolumeSeriesType>     Superclass;
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
    itkTypeMacro(WarpFourDToProjectionStackImageFilter, rtk::FourDToProjectionStackImageFilter)

    typedef rtk::CyclicDeformationImageFilter<DVFImageType>                  DVFInterpolatorType;
    typedef std::vector<double>                                              SignalVectorType;

    /** The forward projection filter cannot be set by the user */
    void SetForwardProjectionFilter (const typename Superclass::ForwardProjectionFilterType::Pointer _arg) {}

    /** The ND + time motion vector field */
    void SetDisplacementField(const DVFSequenceImageType* DVFs);
    typename DVFSequenceImageType::ConstPointer GetDisplacementField();

    void SetSignal(const std::vector<double> signal) ITK_OVERRIDE;

    /** Set and Get for the UseCudaCyclicDeformation variable */
    itkSetMacro(UseCudaCyclicDeformation, bool)
    itkGetMacro(UseCudaCyclicDeformation, bool)

protected:
    WarpFourDToProjectionStackImageFilter();
    ~WarpFourDToProjectionStackImageFilter() {}

    /** Does the real work. */
    void GenerateData() ITK_OVERRIDE;

    void GenerateOutputInformation() ITK_OVERRIDE;

    void GenerateInputRequestedRegion() ITK_OVERRIDE;

    /** The first two inputs should not be in the same space so there is nothing
     * to verify. */
    void VerifyInputInformation() ITK_OVERRIDE {}

    /** Member pointers to the filters used internally (for convenience)*/
    typename DVFInterpolatorType::Pointer               m_DVFInterpolatorFilter;
    std::vector<double>                                 m_Signal;
    bool                                                m_UseCudaCyclicDeformation;

private:
    WarpFourDToProjectionStackImageFilter(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented

};
} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkWarpFourDToProjectionStackImageFilter.hxx"
#endif

#endif
