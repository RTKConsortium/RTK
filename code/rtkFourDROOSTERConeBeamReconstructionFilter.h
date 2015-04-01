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
#ifndef __rtkFourDROOSTERConeBeamReconstructionFilter_h
#define __rtkFourDROOSTERConeBeamReconstructionFilter_h

#include <itkThresholdImageFilter.h>

#include "rtkFourDConjugateGradientConeBeamReconstructionFilter.h"
#include "rtkTotalVariationDenoiseSequenceImageFilter.h"
#ifdef RTK_USE_CUDA
  #include "rtkCudaLastDimensionTVDenoisingImageFilter.h"
  #include "rtkCudaAverageOutOfROIImageFilter.h"
#else
  #include "rtkTotalVariationDenoisingBPDQImageFilter.h"
  #include "rtkAverageOutOfROIImageFilter.h"
#endif
#include "rtkWarpSequenceImageFilter.h"
#include "rtkUnwarpSequenceImageFilter.h"
#include <itkSubtractImageFilter.h>
#include <itkAddImageFilter.h>

namespace rtk
{
  /** \class FourDROOSTERConeBeamReconstructionFilter
   * \brief Implements 4D RecOnstructiOn using Spatial and TEmporal
   * Regularization (short 4D ROOSTER)
   *
   * See the reference paper: "Cardiac C-arm computed tomography using
   * a 3D + time ROI reconstruction method with spatial and temporal regularization"
   * by Mory et al.
   *
   * 4D ROOSTER reconstruction consists in performing 4D Conjugate
   * Gradient reconstruction, then applying several regularization steps :
   * - Replacing all negative values by zero
   * - Averaging along time where no movement is expected
   * - Applying total variation denoising in space
   * - Applying total variation denoising in time
   * and starting over as many times as the number of main loop iterations desired.
   *
   * If the required motion vector fields are provided, 4D ROOSTER performs
   * total variation denoising in time the following way:
   * - each 3D volume of the sequence is warped to an average position
   * - TV denoising in time is applied on the warped sequence
   * - each 3D volume is warped back
   * Otherwise, TV denoising in time is applied on the non-warped sequence
   *
   * \dot
   * digraph FourDROOSTERConeBeamReconstructionFilter {
   *
   * subgraph clusterROOSTER
   *    {
   *    label="ROOSTER"
   *
   *    Input0 [ label="Input 0 (Input: 4D sequence of volumes)"];
   *    Input0 [shape=Mdiamond];
   *    Input1 [label="Input 1 (Projections)"];
   *    Input1 [shape=Mdiamond];
   *    Input2 [label="Input 2 (Motion mask)"];
   *    Input2 [shape=Mdiamond];
   *    Output [label="Output (Reconstruction: 4D sequence of volumes)"];
   *    Output [shape=Mdiamond];
   *
   *    node [shape=box];
   *    FourDCG [ label="rtk::FourDConjugateGradientConeBeamReconstructionFilter" URL="\ref rtk::FourDConjugateGradientConeBeamReconstructionFilter"];
   *    Positivity [ label="itk::ThresholdImageFilter (positivity)" URL="\ref itk::ThresholdImageFilter"];
   *    ROI [ label="rtk::AverageOutOfROIImageFilter" URL="\ref rtk::AverageOutOfROIImageFilter"];
   *    TVSpace [ label="rtk::TotalVariationDenoisingBPDQImageFilter (in space)" URL="\ref rtk::TotalVariationDenoisingBPDQImageFilter"];
   *    TVTime [ label="rtk::TotalVariationDenoisingBPDQImageFilter (along time)" URL="\ref rtk::TotalVariationDenoisingBPDQImageFilter"];
   *    AfterInput0 [label="", fixedsize="false", width=0, height=0, shape=none];
   *    AfterTVTime [label="", fixedsize="false", width=0, height=0, shape=none];
   *
   *    Input0 -> AfterInput0 [arrowhead=None];
   *    AfterInput0 -> FourDCG;
   *    Input1 -> FourDCG;
   *    FourDCG -> Positivity;
   *    Positivity -> ROI;
   *    Input2 -> ROI;
   *    ROI -> TVSpace;
   *    TVSpace -> TVTime;
   *    TVTime -> AfterTVTime [arrowhead=None];
   *    AfterTVTime -> Output;
   *    AfterTVTime -> AfterInput0 [style=dashed];
   *    }
   *
   * subgraph clusterMCROOSTER
   *    {
   *    label="Motion Compensated ROOSTER"
   *
   *    MC_Input0 [label="Input 0 (Input: 4D sequence of volumes)"];
   *    MC_Input0 [shape=Mdiamond];
   *    MC_Input1 [label="Input 1 (Projections)"];
   *    MC_Input1 [shape=Mdiamond];
   *    MC_Input2 [label="Input 2 (Motion mask)"];
   *    MC_Input2 [shape=Mdiamond];
   *    MC_Input3 [label="Input 3 (4D Displacement Vector Field)"];
   *    MC_Input3 [shape=Mdiamond];
   *    MC_Output [label="Output (Reconstruction: 4D sequence of volumes)"];
   *    MC_Output [shape=Mdiamond];
   *
   *    node [shape=box];
   *    MC_FourDCG [ label="rtk::FourDConjugateGradientConeBeamReconstructionFilter" URL="\ref rtk::FourDConjugateGradientConeBeamReconstructionFilter"];
   *    MC_Positivity [ label="itk::ThresholdImageFilter (positivity)" URL="\ref itk::ThresholdImageFilter"];
   *    MC_ROI [ label="rtk::AverageOutOfROIImageFilter" URL="\ref rtk::AverageOutOfROIImageFilter"];
   *    MC_TVSpace [ label="rtk::TotalVariationDenoisingBPDQImageFilter (in space)" URL="\ref rtk::TotalVariationDenoisingBPDQImageFilter"];
   *    MC_BackwardWarp [ label="rtk::WarpSequenceImageFilter (interpolate)" URL="\ref rtk::WarpSequenceImageFilter"];
   *    MC_TVTime [ label="rtk::TotalVariationDenoisingBPDQImageFilter (along time)" URL="\ref rtk::TotalVariationDenoisingBPDQImageFilter"];
   *    MC_ForwardWarp [ label="rtk::WarpSequenceImageFilter (splat)" URL="\ref rtk::WarpSequenceImageFilter"];
   *    MC_Subtract [ label="itk::SubtractImageFilter" URL="\ref itk::SubtractImageFilter"];
   *    MC_Add [ label="itk::AddImageFilter" URL="\ref itk::AddImageFilter"];
   *    MC_AfterInput0 [label="", fixedsize="false", width=0, height=0, shape=none];
   *    MC_AfterWarpBackward [label="", fixedsize="false", width=0, height=0, shape=none];
   *    MC_AfterTVSpace [label="", fixedsize="false", width=0, height=0, shape=none];
   *    MC_AfterAdd [label="", fixedsize="false", width=0, height=0, shape=none];
   *
   *    MC_Input0 -> MC_AfterInput0 [arrowhead=None];
   *    MC_AfterInput0 -> MC_FourDCG;
   *    MC_Input1 -> MC_FourDCG;
   *    MC_FourDCG -> MC_Positivity;
   *    MC_Positivity -> MC_ROI;
   *    MC_Input2 -> MC_ROI;
   *    MC_ROI -> MC_TVSpace;
   *    MC_TVSpace -> MC_AfterTVSpace [arrowhead=None];
   *    MC_AfterTVSpace -> MC_BackwardWarp;
   *    MC_AfterTVSpace -> MC_Add;
   *    MC_Input3 -> MC_BackwardWarp;
   *    MC_BackwardWarp -> MC_AfterWarpBackward;
   *    MC_AfterWarpBackward -> MC_TVTime;
   *    MC_AfterWarpBackward -> MC_Subtract;
   *    MC_TVTime -> MC_Subtract;
   *    MC_Subtract -> MC_ForwardWarp;
   *    MC_Input3 -> MC_ForwardWarp;
   *    MC_ForwardWarp -> MC_Add;
   *    MC_Add -> MC_AfterAdd [arrowhead=None];
   *    MC_AfterAdd -> MC_Output;
   *    MC_AfterAdd -> MC_AfterInput0 [style=dashed];
   *    }
   *
   * }
   * \enddot
   *
   * \test rtkfourdroostertest.cxx
   *
   * \author Cyril Mory
   *
   * \ingroup ReconstructionAlgorithm
   */

template< typename VolumeSeriesType, typename ProjectionStackType>
class FourDROOSTERConeBeamReconstructionFilter : public rtk::IterativeConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>
{
public:
    /** Standard class typedefs. */
    typedef FourDROOSTERConeBeamReconstructionFilter             Self;
    typedef rtk::IterativeConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType> Superclass;
    typedef itk::SmartPointer< Self >        Pointer;
    typedef ProjectionStackType              VolumeType;
    typedef itk::CovariantVector< typename VolumeSeriesType::ValueType, VolumeSeriesType::ImageDimension - 1> CovariantVectorForSpatialGradient;
    typedef itk::CovariantVector< typename VolumeSeriesType::ValueType, 1> CovariantVectorForTemporalGradient;
    typedef CovariantVectorForSpatialGradient MVFVectorType;

#ifdef RTK_USE_CUDA
    typedef itk::CudaImage<CovariantVectorForSpatialGradient, VolumeSeriesType::ImageDimension> SpatialGradientImageType;
    typedef itk::CudaImage<CovariantVectorForTemporalGradient, VolumeSeriesType::ImageDimension> TemporalGradientImageType;
    typedef itk::CudaImage<MVFVectorType, VolumeSeriesType::ImageDimension> MVFSequenceImageType;
    typedef itk::CudaImage<MVFVectorType, VolumeSeriesType::ImageDimension - 1> MVFImageType;
#else
    typedef itk::Image<CovariantVectorForSpatialGradient, VolumeSeriesType::ImageDimension> SpatialGradientImageType;
    typedef itk::Image<CovariantVectorForTemporalGradient, VolumeSeriesType::ImageDimension> TemporalGradientImageType;
    typedef itk::Image<MVFVectorType, VolumeSeriesType::ImageDimension> MVFSequenceImageType;
    typedef itk::Image<MVFVectorType, VolumeSeriesType::ImageDimension - 1> MVFImageType;
#endif

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(FourDROOSTERConeBeamReconstructionFilter, itk::ImageToImageFilter)

    /** The 4D image to be updated.*/
    void SetInputVolumeSeries(const VolumeSeriesType* VolumeSeries);
    typename VolumeSeriesType::ConstPointer GetInputVolumeSeries();

    /** The stack of measured projections */
    void SetInputProjectionStack(const ProjectionStackType* Projection);
    typename ProjectionStackType::Pointer   GetInputProjectionStack();

    /** The region of interest outside of which all movement is removed */
    void SetMotionMask(const VolumeType* mask);
    typename VolumeType::Pointer            GetInputROI();

    /** The motion vector field used to warp the sequence before and after TV denoising along time */
    void SetDisplacementField(const MVFSequenceImageType* MVFs);
    typename MVFSequenceImageType::Pointer            GetDisplacementField();

    typedef rtk::FourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>    FourDCGFilterType;
    typedef itk::ThresholdImageFilter<VolumeSeriesType>                                                       ThresholdFilterType;
    typedef rtk::AverageOutOfROIImageFilter <VolumeSeriesType, VolumeType>                                    AverageOutOfROIFilterType;
    typedef rtk::TotalVariationDenoiseSequenceImageFilter<VolumeSeriesType>                                   SpatialTVDenoisingFilterType;
    typedef rtk::WarpSequenceImageFilter<VolumeSeriesType, MVFSequenceImageType, VolumeType, MVFImageType>    WarpSequenceFilterType;
    typedef rtk::TotalVariationDenoisingBPDQImageFilter<VolumeSeriesType, TemporalGradientImageType>          TemporalTVDenoisingFilterType;
    typedef rtk::UnwarpSequenceImageFilter<VolumeSeriesType, MVFSequenceImageType, VolumeType, MVFImageType>  UnwarpSequenceFilterType;

    /** Pass the ForwardProjection filter to SingleProjectionToFourDFilter */
    void SetForwardProjectionFilter(int fwtype);

    /** Pass the backprojection filter to ProjectionStackToFourD*/
    void SetBackProjectionFilter(int bptype);

    /** Pass the interpolation weights to SingleProjectionToFourDFilter */
    void SetWeights(const itk::Array2D<float> _arg);

    void PrintTiming(std::ostream& os) const;

    // Main loop iterations
    itkSetMacro(MainLoop_iterations, int)
    itkGetMacro(MainLoop_iterations, int)

    // Conjugate gradient iterations
    itkSetMacro(CG_iterations, int)
    itkGetMacro(CG_iterations, int)

    // TV filter parameters
    itkSetMacro(GammaSpace, float)
    itkGetMacro(GammaSpace, float)

    itkSetMacro(GammaTime, float)
    itkGetMacro(GammaTime, float)

    itkSetMacro(PhaseShift, float)
    itkGetMacro(PhaseShift, float)

    itkSetMacro(TV_iterations, int)
    itkGetMacro(TV_iterations, int)

    itkSetMacro(Geometry, typename ThreeDCircularProjectionGeometry::Pointer)
    itkGetMacro(Geometry, typename ThreeDCircularProjectionGeometry::Pointer)

    // Boolean : should warping be performed ?
    itkSetMacro(PerformWarping, bool)
    itkGetMacro(PerformWarping, bool)

protected:
    FourDROOSTERConeBeamReconstructionFilter();
    ~FourDROOSTERConeBeamReconstructionFilter(){}

    /** Does the real work. */
    virtual void GenerateData();

    virtual void GenerateOutputInformation();

    virtual void GenerateInputRequestedRegion();

    // Inputs are not supposed to occupy the same physical space,
    // so there is nothing to verify
    virtual void VerifyInputInformation(){}

    /** Member pointers to the filters used internally (for convenience)*/
    typename FourDCGFilterType::Pointer                     m_FourDCGFilter;
    typename ThresholdFilterType::Pointer                   m_PositivityFilter;
    typename AverageOutOfROIFilterType::Pointer             m_AverageOutOfROIFilter;
    typename SpatialTVDenoisingFilterType::Pointer          m_TVDenoisingSpace;
    typename WarpSequenceFilterType::Pointer                m_Warp;
    typename TemporalTVDenoisingFilterType::Pointer         m_TVDenoisingTime;
    typename UnwarpSequenceFilterType::Pointer              m_Unwarp;

private:
    FourDROOSTERConeBeamReconstructionFilter(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented

    // Boolean : should warping be performed ?
    bool m_PerformWarping;

    // Regularization parameters
    float m_GammaSpace;
    float m_GammaTime;
    float m_PhaseShift;
    bool m_DimensionsProcessedForTVSpace[VolumeSeriesType::ImageDimension];
    bool m_DimensionsProcessedForTVTime[VolumeSeriesType::ImageDimension];

    // Iterations
    int   m_MainLoop_iterations;
    int   m_CG_iterations;
    int   m_TV_iterations;

    // Geometry
    typename rtk::ThreeDCircularProjectionGeometry::Pointer m_Geometry;

    /** Time probes */
    itk::TimeProbe m_CGProbe;
    itk::TimeProbe m_PositivityProbe;
    itk::TimeProbe m_ROIProbe;
    itk::TimeProbe m_TVSpaceProbe;
    itk::TimeProbe m_TVTimeProbe;
    itk::TimeProbe m_WarpProbe;
    itk::TimeProbe m_UnwarpProbe;
};
} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkFourDROOSTERConeBeamReconstructionFilter.txx"
#endif

#endif
