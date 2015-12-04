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
#include "rtkDaubechiesWaveletsDenoiseSequenceImageFilter.h"
#include "rtkWarpSequenceImageFilter.h"
#include "rtkUnwarpSequenceImageFilter.h"
#include <itkSubtractImageFilter.h>
#include <itkAddImageFilter.h>

#include <itkResampleImageFilter.h>
#include <itkNearestNeighborInterpolateImageFunction.h>
#include <itkIdentityTransform.h>


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
   * - Applying total variation or wavelets denoising in space
   * - Applying total variation denoising in time
   * and starting over as many times as the number of main loop iterations desired.
   *
   * If both the displacement vector fields to a reference phase and from a reference phase are provided,
   * 4D ROOSTER performs total variation denoising in time the following way:
   * - each 3D volume of the sequence is warped to the reference phase using the first DVF
   * - TV denoising in time is applied on the warped sequence
   * - the difference sequence between the warped-then-denoised sequence and the warped sequence is computed
   * - that difference sequence is warped from the reference phase using the second DVF, and added to the output of spatial denoising
   *
   * If only the displacement vector field to a reference phase is provided,
   * 4D ROOSTER performs total variation denoising in time the following way:
   * - each 3D volume of the sequence is warped to the reference phase using the first DVF
   * - TV denoising in time is applied on the warped sequence
   * - the warped-then-denoised sequence is warped from the reference phase by
   * an iterative procedure based on conjugate gradient. This significantly increases
   * computation time.
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
   *    Resample [ label="itk::ResampleImageFilter" URL="\ref itk::ResampleImageFilter"];
   *    ROI [ label="rtk::AverageOutOfROIImageFilter" URL="\ref rtk::AverageOutOfROIImageFilter"];
   *    TVSpace [ label="rtk::TotalVariationDenoisingBPDQImageFilter (in space)" URL="\ref rtk::TotalVariationDenoisingBPDQImageFilter"];
   *    TVTime [ label="rtk::TotalVariationDenoisingBPDQImageFilter (along time)" URL="\ref rtk::TotalVariationDenoisingBPDQImageFilter"];
   *    AfterInput0 [label="", fixedsize="false", width=0, height=0, shape=none];
   *    AfterTVTime [label="", fixedsize="false", width=0, height=0, shape=none];
   *
   *    Input0 -> AfterInput0 [arrowhead=none];
   *    AfterInput0 -> FourDCG;
   *    Input1 -> FourDCG;
   *    FourDCG -> Positivity;
   *    Positivity -> ROI;
   *    Input2 -> Resample;
   *    Resample -> ROI;
   *    ROI -> TVSpace;
   *    TVSpace -> TVTime;
   *    TVTime -> AfterTVTime [arrowhead=none];
   *    AfterTVTime -> Output;
   *    AfterTVTime -> AfterInput0 [style=dashed];
   *    }
   *
   * subgraph clusterMCROOSTER_one_way
   *    {
   *    label="Motion Compensated ROOSTER, one-way DVF"
   *
   *    MC_OW_Input0 [label="Input 0 (Input: 4D sequence of volumes)"];
   *    MC_OW_Input0 [shape=Mdiamond];
   *    MC_OW_Input1 [label="Input 1 (Projections)"];
   *    MC_OW_Input1 [shape=Mdiamond];
   *    MC_OW_Input2 [label="Input 2 (Motion mask)"];
   *    MC_OW_Input2 [shape=Mdiamond];
   *    MC_OW_Input3 [label="Input 3 (4D Displacement Vector Field)"];
   *    MC_OW_Input3 [shape=Mdiamond];
   *    MC_OW_Output [label="Output (Reconstruction: 4D sequence of volumes)"];
   *    MC_OW_Output [shape=Mdiamond];
   *
   *    node [shape=box];
<<<<<<< Updated upstream
   *    MC_FourDCG [ label="rtk::FourDConjugateGradientConeBeamReconstructionFilter" URL="\ref rtk::FourDConjugateGradientConeBeamReconstructionFilter"];
   *    MC_Positivity [ label="itk::ThresholdImageFilter (positivity)" URL="\ref itk::ThresholdImageFilter"];
   *    MC_Resample [ label="itk::ResampleImageFilter" URL="\ref itk::ResampleImageFilter"];
   *    MC_ROI [ label="rtk::AverageOutOfROIImageFilter" URL="\ref rtk::AverageOutOfROIImageFilter"];
   *    MC_TVSpace [ label="rtk::TotalVariationDenoisingBPDQImageFilter (in space)" URL="\ref rtk::TotalVariationDenoisingBPDQImageFilter"];
   *    MC_BackwardWarp [ label="rtk::WarpSequenceImageFilter (direct field)" URL="\ref rtk::WarpSequenceImageFilter"];
   *    MC_TVTime [ label="rtk::TotalVariationDenoisingBPDQImageFilter (along time)" URL="\ref rtk::TotalVariationDenoisingBPDQImageFilter"];
   *    MC_ForwardWarp [ label="rtk::WarpSequenceImageFilter (inverse field)" URL="\ref rtk::WarpSequenceImageFilter"];
   *    MC_Subtract [ label="itk::SubtractImageFilter" URL="\ref itk::SubtractImageFilter"];
   *    MC_Add [ label="itk::AddImageFilter" URL="\ref itk::AddImageFilter"];
   *    MC_AfterInput0 [label="", fixedsize="false", width=0, height=0, shape=none];
   *    MC_AfterWarpBackward [label="", fixedsize="false", width=0, height=0, shape=none];
   *    MC_AfterTVSpace [label="", fixedsize="false", width=0, height=0, shape=none];
   *    MC_AfterAdd [label="", fixedsize="false", width=0, height=0, shape=none];
=======
   *    MC_OW_FourDCG [ label="rtk::FourDConjugateGradientConeBeamReconstructionFilter" URL="\ref rtk::FourDConjugateGradientConeBeamReconstructionFilter"];
   *    MC_OW_Positivity [ label="itk::ThresholdImageFilter (positivity)" URL="\ref itk::ThresholdImageFilter"];
   *    MC_OW_Resample [ label="itk::ResampleImageFilter" URL="\ref itk::ResampleImageFilter"];
   *    MC_OW_ROI [ label="rtk::AverageOutOfROIImageFilter" URL="\ref rtk::AverageOutOfROIImageFilter"];
   *    MC_OW_TVSpace [ label="rtk::TotalVariationDenoisingBPDQImageFilter (in space)" URL="\ref rtk::TotalVariationDenoisingBPDQImageFilter"];
   *    MC_OW_Warp [ label="rtk::WarpSequenceImageFilter" URL="\ref rtk::WarpSequenceImageFilter"];
   *    MC_OW_TVTime [ label="rtk::TotalVariationDenoisingBPDQImageFilter (along time)" URL="\ref rtk::TotalVariationDenoisingBPDQImageFilter"];
   *    MC_OW_Unwarp [ label="rtk::UnwarpSequenceImageFilter" URL="\ref rtk::UnwarpSequenceImageFilter"];
   *    MC_OW_AfterInput0 [label="", fixedsize="false", width=0, height=0, shape=none];
   *    MC_OW_AfterUnwarp [label="", fixedsize="false", width=0, height=0, shape=none];
>>>>>>> Stashed changes
   *
   *    MC_OW_Input0 -> MC_OW_AfterInput0 [arrowhead=none];
   *    MC_OW_AfterInput0 -> MC_OW_FourDCG;
   *    MC_OW_Input1 -> MC_OW_FourDCG;
   *    MC_OW_FourDCG -> MC_OW_Positivity;
   *    MC_OW_Positivity -> MC_OW_ROI;
   *    MC_OW_Input2 -> MC_OW_Resample;
   *    MC_OW_Resample -> MC_OW_ROI;
   *    MC_OW_ROI -> MC_OW_TVSpace;
   *    MC_OW_TVSpace -> MC_OW_Warp;
   *    MC_OW_Input3 -> MC_OW_Warp;
   *    MC_OW_Input3 -> MC_OW_Unwarp;
   *    MC_OW_Warp -> MC_OW_TVTime;
   *    MC_OW_TVTime -> MC_OW_Unwarp;
   *    MC_OW_Unwarp -> MC_OW_AfterUnwarp [arrowhead=none];
   *    MC_OW_AfterUnwarp -> MC_OW_Output;
   *    MC_OW_AfterUnwarp -> MC_OW_AfterInput0 [style=dashed];
   *    }
   *
   * subgraph clusterMCROOSTER_both_ways
   *    {
   *    label="Motion Compensated ROOSTER, both-ways DVFs"
   *
   *    MC_BW_Input0 [label="Input 0 (Input: 4D sequence of volumes)"];
   *    MC_BW_Input0 [shape=Mdiamond];
   *    MC_BW_Input1 [label="Input 1 (Projections)"];
   *    MC_BW_Input1 [shape=Mdiamond];
   *    MC_BW_Input2 [label="Input 2 (Motion mask)"];
   *    MC_BW_Input2 [shape=Mdiamond];
   *    MC_BW_Input3 [label="Input 3 (4D DVF to reference phase)"];
   *    MC_BW_Input3 [shape=Mdiamond];
   *    MC_BW_Input4 [label="Input 4 (4D DVF from reference phase)"];
   *    MC_BW_Input4 [shape=Mdiamond];
   *    MC_BW_Output [label="Output (Reconstruction: 4D sequence of volumes)"];
   *    MC_BW_Output [shape=Mdiamond];
   *
   *    node [shape=box];
   *    MC_BW_FourDCG [ label="rtk::FourDConjugateGradientConeBeamReconstructionFilter" URL="\ref rtk::FourDConjugateGradientConeBeamReconstructionFilter"];
   *    MC_BW_Positivity [ label="itk::ThresholdImageFilter (positivity)" URL="\ref itk::ThresholdImageFilter"];
   *    MC_BW_Resample [ label="itk::ResampleImageFilter" URL="\ref itk::ResampleImageFilter"];
   *    MC_BW_ROI [ label="rtk::AverageOutOfROIImageFilter" URL="\ref rtk::AverageOutOfROIImageFilter"];
   *    MC_BW_TVSpace [ label="rtk::TotalVariationDenoisingBPDQImageFilter (in space)" URL="\ref rtk::TotalVariationDenoisingBPDQImageFilter"];
   *    MC_BW_Warp [ label="rtk::WarpSequenceImageFilter (to reference phase)" URL="\ref rtk::WarpSequenceImageFilter"];
   *    MC_BW_TVTime [ label="rtk::TotalVariationDenoisingBPDQImageFilter (along time)" URL="\ref rtk::TotalVariationDenoisingBPDQImageFilter"];
   *    MC_BW_InverseWarp [ label="rtk::WarpSequenceImageFilter (from reference phase)" URL="\ref rtk::WarpSequenceImageFilter"];
   *    MC_BW_Subtract [ label="itk::SubtractImageFilter" URL="\ref itk::SubtractImageFilter"];
   *    MC_BW_Add [ label="itk::AddImageFilter" URL="\ref itk::AddImageFilter"];
   *    MC_BW_AfterInput0 [label="", fixedsize="false", width=0, height=0, shape=none];
   *    MC_BW_AfterWarpBackward [label="", fixedsize="false", width=0, height=0, shape=none];
   *    MC_BW_AfterTVSpace [label="", fixedsize="false", width=0, height=0, shape=none];
   *    MC_BW_AfterAdd [label="", fixedsize="false", width=0, height=0, shape=none];
   *
   *    MC_BW_Input0 -> MC_BW_AfterInput0 [arrowhead=none];
   *    MC_BW_AfterInput0 -> MC_BW_FourDCG;
   *    MC_BW_Input1 -> MC_BW_FourDCG;
   *    MC_BW_FourDCG -> MC_BW_Positivity;
   *    MC_BW_Positivity -> MC_BW_ROI;
   *    MC_BW_Input2 -> MC_BW_Resample;
   *    MC_BW_Resample -> MC_BW_ROI;
   *    MC_BW_ROI -> MC_BW_TVSpace;
   *    MC_BW_TVSpace -> MC_BW_AfterTVSpace [arrowhead=none];
   *    MC_BW_AfterTVSpace -> MC_BW_Warp;
   *    MC_BW_AfterTVSpace -> MC_BW_Add;
   *    MC_BW_Input3 -> MC_BW_Warp;
   *    MC_BW_Warp -> MC_BW_AfterWarpBackward;
   *    MC_BW_AfterWarpBackward -> MC_BW_TVTime;
   *    MC_BW_AfterWarpBackward -> MC_BW_Subtract;
   *    MC_BW_TVTime -> MC_BW_Subtract;
   *    MC_BW_Subtract -> MC_BW_InverseWarp;
   *    MC_BW_Input4 -> MC_BW_InverseWarp;
   *    MC_BW_InverseWarp -> MC_BW_Add;
   *    MC_BW_Add -> MC_BW_AfterAdd [arrowhead=none];
   *    MC_BW_AfterAdd -> MC_BW_Output;
   *    MC_BW_AfterAdd -> MC_BW_AfterInput0 [style=dashed];
   *    }
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
  typedef FourDROOSTERConeBeamReconstructionFilter                                          Self;
  typedef rtk::IterativeConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType> Superclass;
  typedef itk::SmartPointer< Self >                                                         Pointer;
  typedef ProjectionStackType                                                               VolumeType;
  typedef itk::CovariantVector< typename VolumeSeriesType::ValueType, VolumeSeriesType::ImageDimension - 1> CovariantVectorForSpatialGradient;
  typedef itk::CovariantVector< typename VolumeSeriesType::ValueType, 1>                                    CovariantVectorForTemporalGradient;
  typedef CovariantVectorForSpatialGradient                                                                 MVFVectorType;

#ifdef RTK_USE_CUDA
  typedef itk::CudaImage<CovariantVectorForSpatialGradient, VolumeSeriesType::ImageDimension>   SpatialGradientImageType;
  typedef itk::CudaImage<CovariantVectorForTemporalGradient, VolumeSeriesType::ImageDimension>  TemporalGradientImageType;
  typedef itk::CudaImage<MVFVectorType, VolumeSeriesType::ImageDimension>                       MVFSequenceImageType;
  typedef itk::CudaImage<MVFVectorType, VolumeSeriesType::ImageDimension - 1>                   MVFImageType;
#else
  typedef itk::Image<CovariantVectorForSpatialGradient, VolumeSeriesType::ImageDimension>       SpatialGradientImageType;
  typedef itk::Image<CovariantVectorForTemporalGradient, VolumeSeriesType::ImageDimension>      TemporalGradientImageType;
  typedef itk::Image<MVFVectorType, VolumeSeriesType::ImageDimension>                           MVFSequenceImageType;
  typedef itk::Image<MVFVectorType, VolumeSeriesType::ImageDimension - 1>                       MVFImageType;
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

  /** The motion vector fields used to warp the sequence before and after TV denoising along time */
  void SetDisplacementField(const MVFSequenceImageType* MVFs);
  void SetInverseDisplacementField(const MVFSequenceImageType* MVFs);
  typename MVFSequenceImageType::Pointer            GetDisplacementField();
  typename MVFSequenceImageType::Pointer            GetInverseDisplacementField();

  typedef rtk::FourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>    FourDCGFilterType;
  typedef itk::ThresholdImageFilter<VolumeSeriesType>                                                       ThresholdFilterType;
  typedef itk::ResampleImageFilter<VolumeType, VolumeType>                                                  ResampleFilterType;
  typedef rtk::AverageOutOfROIImageFilter <VolumeSeriesType, VolumeType>                                    AverageOutOfROIFilterType;
  typedef rtk::TotalVariationDenoiseSequenceImageFilter<VolumeSeriesType>                                   SpatialTVDenoisingFilterType;
  typedef rtk::DaubechiesWaveletsDenoiseSequenceImageFilter<VolumeSeriesType>                               SpatialWaveletsDenoisingFilterType;
  typedef rtk::WarpSequenceImageFilter<VolumeSeriesType, MVFSequenceImageType, VolumeType, MVFImageType>    WarpSequenceFilterType;
  typedef rtk::TotalVariationDenoisingBPDQImageFilter<VolumeSeriesType, TemporalGradientImageType>          TemporalTVDenoisingFilterType;
  typedef rtk::UnwarpSequenceImageFilter<VolumeSeriesType, MVFSequenceImageType, VolumeType, MVFImageType>  UnwarpSequenceFilterType;
  typedef itk::SubtractImageFilter<VolumeSeriesType, VolumeSeriesType>                                      SubtractFilterType;
  typedef itk::AddImageFilter<VolumeSeriesType, VolumeSeriesType>                                           AddFilterType;

  /** Pass the ForwardProjection filter to SingleProjectionToFourDFilter */
  void SetForwardProjectionFilter(int fwtype);

  /** Pass the backprojection filter to ProjectionStackToFourD*/
  void SetBackProjectionFilter(int bptype);

  /** Pass the interpolation weights to SingleProjectionToFourDFilter */
  virtual void SetWeights(const itk::Array2D<float> _arg);

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

  // Booleans : should warping be performed ? how should inverse warping be performed ?
  itkSetMacro(PerformWarping, bool)
  itkGetMacro(PerformWarping, bool)

  itkSetMacro(ComputeInverseWarpingByConjugateGradient, bool)
  itkGetMacro(ComputeInverseWarpingByConjugateGradient, bool)

  itkSetMacro(UseNearestNeighborInterpolationInWarping, bool)
  itkGetMacro(UseNearestNeighborInterpolationInWarping, bool)

  /** Get / Set whether conjugate gradient should be performed on GPU */
  itkGetMacro(CudaConjugateGradient, bool)
  itkSetMacro(CudaConjugateGradient, bool)

  /** Get / Set whether Daubechies wavelets should replace TV in spatial denoising*/
  itkGetMacro(WaveletsSpatialDenoising, bool)
  itkSetMacro(WaveletsSpatialDenoising, bool)

  /** Set the number of levels of the wavelets decomposition */
  itkGetMacro(NumberOfLevels, unsigned int)
  itkSetMacro(NumberOfLevels, unsigned int)

  /** Sets the order of the Daubechies wavelet used to deconstruct/reconstruct the image pyramid */
  itkGetMacro(Order, unsigned int)
  itkSetMacro(Order, unsigned int)

protected:
  FourDROOSTERConeBeamReconstructionFilter();
  ~FourDROOSTERConeBeamReconstructionFilter(){}

  /** Does the real work. */
  virtual void GenerateData();

  virtual void PreparePipeline();

  virtual void GenerateOutputInformation();

  virtual void GenerateInputRequestedRegion();

  // Inputs are not supposed to occupy the same physical space,
  // so there is nothing to verify
  virtual void VerifyInputInformation(){}

  /** Member pointers to the filters used internally (for convenience)*/
  typename FourDCGFilterType::Pointer                     m_FourDCGFilter;
  typename ThresholdFilterType::Pointer                   m_PositivityFilter;
  typename ResampleFilterType::Pointer                    m_ResampleFilter;
  typename AverageOutOfROIFilterType::Pointer             m_AverageOutOfROIFilter;
  typename SpatialTVDenoisingFilterType::Pointer          m_TVDenoisingSpace;
  typename SpatialWaveletsDenoisingFilterType::Pointer    m_WaveletsDenoisingSpace;
  typename WarpSequenceFilterType::Pointer                m_Warp;
  typename TemporalTVDenoisingFilterType::Pointer         m_TVDenoisingTime;
  typename UnwarpSequenceFilterType::Pointer              m_Unwarp;
  typename WarpSequenceFilterType::Pointer                m_InverseWarp;
  typename SubtractFilterType::Pointer                    m_SubtractFilter;
  typename AddFilterType::Pointer                         m_AddFilter;

  // Booleans :
  // should warping be performed ?
  // should conjugate gradient be performed on GPU ?
  // should wavelets replace TV in spatial denoising ?
  bool  m_PerformWarping;
  bool  m_ComputeInverseWarpingByConjugateGradient;
  bool  m_UseNearestNeighborInterpolationInWarping; //Default is false, linear interpolation is used instead
  bool  m_CudaConjugateGradient;
  bool  m_WaveletsSpatialDenoising;

  // Regularization parameters
  float m_GammaSpace;
  float m_GammaTime;
  float m_PhaseShift;
  bool  m_DimensionsProcessedForTVSpace[VolumeSeriesType::ImageDimension];
  bool  m_DimensionsProcessedForTVTime[VolumeSeriesType::ImageDimension];

  /** Information for the wavelets denoising filter */
  unsigned int    m_Order;
  unsigned int    m_NumberOfLevels;

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
  itk::TimeProbe m_DenoisingSpaceProbe;
  itk::TimeProbe m_TVTimeProbe;
  itk::TimeProbe m_WarpProbe;
  itk::TimeProbe m_UnwarpProbe;

private:
  FourDROOSTERConeBeamReconstructionFilter(const Self &); //purposely not implemented
  void operator=(const Self &);  //purposely not implemented

};
} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkFourDROOSTERConeBeamReconstructionFilter.txx"
#endif

#endif
