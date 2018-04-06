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
#ifndef rtkFourDROOSTERConeBeamReconstructionFilter_h
#define rtkFourDROOSTERConeBeamReconstructionFilter_h

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
#include "rtkLastDimensionL0GradientDenoisingImageFilter.h"
#include "rtkTotalNuclearVariationDenoisingBPDQImageFilter.h"

#include <itkThresholdImageFilter.h>
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
   * - Applying total variation denoising in space
   * - Applying wavelets denoising in space
   * - Applying total variation denoising in time
   * - Applying gradient's L0 norm denoising in time
   * - Applying total nuclear variation denoising
   * and starting over as many times as the number of main loop iterations desired.
   *
   * If both the displacement vector fields to a reference phase and from a reference phase are provided,
   * 4D ROOSTER performs the denoising in time the following way:
   * - each 3D volume of the sequence is warped to the reference phase using the first DVF
   * - denoising in time is applied on the warped sequence
   * - the difference sequence between the warped-then-denoised sequence and the warped sequence is computed
   * - that difference sequence is warped from the reference phase using the second DVF, and added to the output of spatial denoising
   *
   * If only the displacement vector field to a reference phase is provided,
   * 4D ROOSTER performs total variation denoising in time the following way:
   * - each 3D volume of the sequence is warped to the reference phase using the first DVF
   * - denoising in time is applied on the warped sequence
   * - the warped-then-denoised sequence is warped from the reference phase by
   * an iterative procedure based on conjugate gradient. This significantly increases
   * computation time.
   *
   * \dot
   * digraph FourDROOSTERConeBeamReconstructionFilter {
   *
   * PrimaryInput [label="Primary input (4D sequence of volumes)"];
   * PrimaryInput [shape=Mdiamond];
   * InputProjectionStack [label="Input projection stack"];
   * InputProjectionStack [shape=Mdiamond];
   * InputMotionMask [label="Input motion mask"];
   * InputMotionMask [shape=Mdiamond];
   * InputDisplacementField [label="Input displacement field"];
   * InputDisplacementField [shape=Mdiamond];
   * InputInverseDisplacementField [group=invwarp, label="Input inverse displacement field"];
   * InputInverseDisplacementField [shape=Mdiamond];
   * Output [label="Output (Reconstruction: 4D sequence of volumes)"];
   * Output [shape=Mdiamond];
   *
   * node [shape=box];
   * FourDCG [ label="rtk::FourDConjugateGradientConeBeamReconstructionFilter" URL="\ref rtk::FourDConjugateGradientConeBeamReconstructionFilter"];
   * Positivity [group=regul, label="itk::ThresholdImageFilter (positivity)" URL="\ref itk::ThresholdImageFilter"];
   * Resample [group=regul, label="itk::ResampleImageFilter" URL="\ref itk::ResampleImageFilter"];
   * MotionMask [group=regul, label="rtk::AverageOutOfROIImageFilter" URL="\ref rtk::AverageOutOfROIImageFilter"];
   * TVSpace [group=regul, label="rtk::TotalVariationDenoisingBPDQImageFilter (in space)" URL="\ref rtk::TotalVariationDenoisingBPDQImageFilter"];
   * Wavelets [group=regul, label="rtk::DaubechiesWaveletsDenoiseSequenceImageFilter (in space)" URL="\ref rtk::DaubechiesWaveletsDenoiseSequenceImageFilter"];
   * Warp [group=regul, label="rtk::WarpSequenceImageFilter (direct field)" URL="\ref rtk::WarpSequenceImageFilter"];
   * TVTime [group=regul, label="rtk::TotalVariationDenoisingBPDQImageFilter (along time)" URL="\ref rtk::TotalVariationDenoisingBPDQImageFilter"];
   * L0Time [group=regul, label="rtk::LastDimensionL0GradientDenoisingImageFilter (along time)" URL="\ref rtk::LastDimensionL0GradientDenoisingImageFilter"];
   * TNV [group=regul, label="rtk::TotalNuclearVariationDenoisingBPDQImageFilter" URL="\ref rtk::TotalNuclearVariationDenoisingBPDQImageFilter"];
   * Unwarp [group=regul, label="rtk::UnwarpSequenceImageFilter" URL="\ref rtk::UnwarpSequenceImageFilter"];
   * Subtract [group=invwarp, label="itk::SubtractImageFilter" URL="\ref itk::SubtractImageFilter"];
   * InverseWarp [group=invwarp, label="rtk::WarpSequenceImageFilter (inverse field)" URL="\ref rtk::WarpSequenceImageFilter"];
   * Add [group=invwarp, label="itk::AddImageFilter" URL="\ref itk::AddImageFilter"];
   *
   * AfterPrimaryInput [group=invisible, label="", fixedsize="false", width=0, height=0, shape=none];
   * AfterFourDCG [group=invisible, label="m_PerformPositivity ?", fixedsize="false", width=0, height=0, shape=none];
   * AfterPositivity [group=invisible, label="m_PerformMotionMask ?", fixedsize="false", width=0, height=0, shape=none];
   * AfterMotionMask [group=invisible, label="m_PerformTVSpatialDenoising ?", fixedsize="false", width=0, height=0, shape=none];
   * AfterTVSpace [group=invisible, label="m_PerformWaveletsSpatialDenoising ?", fixedsize="false", width=0, height=0, shape=none];
   * AfterWavelets [group=invisible, label="m_PerformWarping ?", fixedsize="false", width=0, height=0, shape=none];
   * AfterWarp [group=invisible, label="m_PerformTVTemporalDenoising ?", fixedsize="false", width=0, height=0, shape=none];
   * AfterTVTime [group=invisible, label="m_PerformL0TemporalDenoising ?", fixedsize="false", width=0, height=0, shape=none];
   * AfterL0Time [group=invisible, label="m_ComputeInverseWarpingByConjugateGradient ?", fixedsize="false", width=0, height=0, shape=none];
   * AfterTNV [group=invisible, label="m_PerformTNVDenoising ?", fixedsize="false", width=0, height=0, shape=none];
   * AfterUnwarp [group=invisible, label="", fixedsize="false", width=0, height=0, shape=none];
   *
   * InputDisplacementField -> Warp;
   * InputDisplacementField -> Unwarp;
   * InputInverseDisplacementField -> InverseWarp;
   *
   * PrimaryInput -> AfterPrimaryInput [arrowhead=none];
   * AfterPrimaryInput -> FourDCG;
   * InputProjectionStack -> FourDCG;
   * FourDCG -> AfterFourDCG;
   * AfterFourDCG -> Positivity [label="true"];
   * Positivity -> AfterPositivity;
   * AfterPositivity -> MotionMask [label="true"];
   * MotionMask -> AfterMotionMask;
   * InputMotionMask -> Resample;
   * Resample -> MotionMask;
   * AfterMotionMask -> TVSpace [label="true"];
   * TVSpace -> AfterTVSpace;
   * AfterTVSpace -> Wavelets [label="true"];
   * Wavelets -> AfterWavelets;
   * AfterWavelets -> Warp [label="true"];
   * Warp -> AfterWarp;
   * AfterWarp -> TVTime [label="true"];
   * TVTime -> AfterTVTime [arrowhead=none];
   * AfterTVTime -> L0Time [label="true"];
   * L0Time -> AfterL0Time;
   * AfterL0Time -> TNV [label="true"];
   * TNV -> AfterTNV;
   * AfterTNV -> Unwarp [label="true"];
   * Unwarp -> AfterUnwarp
   * AfterUnwarp -> Output;
   * AfterUnwarp -> AfterPrimaryInput [style=dashed];
   *
   * AfterL0Time -> Subtract [label="false"];
   * AfterWarp -> Subtract;
   * Subtract -> InverseWarp;
   * InverseWarp -> Add;
   * AfterWavelets -> Add;
   * Add -> AfterUnwarp;
   *
   * AfterFourDCG -> AfterPositivity  [label="false"];
   * AfterPositivity -> AfterMotionMask [label="false"];
   * AfterMotionMask -> AfterTVSpace [label="false"];
   * AfterTVSpace -> AfterWavelets [label="false"];
   * AfterWavelets -> AfterWarp [label="false"];
   * AfterWarp -> AfterTVTime [label="false"];
   * AfterTVTime -> AfterL0Time [label="false"];
   * AfterL0Time -> AfterTNV [label="false"];
   * AfterTNV -> AfterUnwarp [label="m_PerformWarping = false"];
   *
   * // Invisible edges between the regularization filters
   * edge[style=invis];
   * Positivity -> MotionMask;
   * MotionMask -> TVSpace;
   * TVSpace -> Wavelets;
   * Wavelets -> Warp;
   * Warp -> TVTime;
   * TVTime -> L0Time;
   * L0Time -> TNV;
   * TNV -> Unwarp;
   *
   * InputInverseDisplacementField -> Subtract;
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
  typedef CovariantVectorForSpatialGradient                                                                 DVFVectorType;

#ifdef RTK_USE_CUDA
  typedef itk::CudaImage<CovariantVectorForSpatialGradient, VolumeSeriesType::ImageDimension>   SpatialGradientImageType;
  typedef itk::CudaImage<CovariantVectorForTemporalGradient, VolumeSeriesType::ImageDimension>  TemporalGradientImageType;
  typedef itk::CudaImage<DVFVectorType, VolumeSeriesType::ImageDimension>                       DVFSequenceImageType;
  typedef itk::CudaImage<DVFVectorType, VolumeSeriesType::ImageDimension - 1>                   DVFImageType;
#else
  typedef itk::Image<CovariantVectorForSpatialGradient, VolumeSeriesType::ImageDimension>       SpatialGradientImageType;
  typedef itk::Image<CovariantVectorForTemporalGradient, VolumeSeriesType::ImageDimension>      TemporalGradientImageType;
  typedef itk::Image<DVFVectorType, VolumeSeriesType::ImageDimension>                           DVFSequenceImageType;
  typedef itk::Image<DVFVectorType, VolumeSeriesType::ImageDimension - 1>                       DVFImageType;
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
  typename VolumeType::Pointer            GetMotionMask();

  /** The motion vector fields used to warp the sequence before and after TV denoising along time */
  void SetDisplacementField(const DVFSequenceImageType* DVFs);
  void SetInverseDisplacementField(const DVFSequenceImageType* DVFs);
  typename DVFSequenceImageType::Pointer            GetDisplacementField();
  typename DVFSequenceImageType::Pointer            GetInverseDisplacementField();

  typedef rtk::FourDConjugateGradientConeBeamReconstructionFilter<VolumeSeriesType, ProjectionStackType>    FourDCGFilterType;
  typedef itk::ThresholdImageFilter<VolumeSeriesType>                                                       ThresholdFilterType;
  typedef itk::ResampleImageFilter<VolumeType, VolumeType>                                                  ResampleFilterType;
  typedef rtk::AverageOutOfROIImageFilter <VolumeSeriesType, VolumeType>                                    AverageOutOfROIFilterType;
  typedef rtk::TotalVariationDenoiseSequenceImageFilter<VolumeSeriesType>                                   SpatialTVDenoisingFilterType;
  typedef rtk::DaubechiesWaveletsDenoiseSequenceImageFilter<VolumeSeriesType>                               SpatialWaveletsDenoisingFilterType;
  typedef rtk::WarpSequenceImageFilter<VolumeSeriesType, DVFSequenceImageType, VolumeType, DVFImageType>    WarpSequenceFilterType;
  typedef rtk::TotalVariationDenoisingBPDQImageFilter<VolumeSeriesType, TemporalGradientImageType>          TemporalTVDenoisingFilterType;
  typedef rtk::UnwarpSequenceImageFilter<VolumeSeriesType, DVFSequenceImageType, VolumeType, DVFImageType>  UnwarpSequenceFilterType;
  typedef itk::SubtractImageFilter<VolumeSeriesType, VolumeSeriesType>                                      SubtractFilterType;
  typedef itk::AddImageFilter<VolumeSeriesType, VolumeSeriesType>                                           AddFilterType;
  typedef rtk::LastDimensionL0GradientDenoisingImageFilter<VolumeSeriesType>                                TemporalL0DenoisingFilterType;
  typedef rtk::TotalNuclearVariationDenoisingBPDQImageFilter<VolumeSeriesType, SpatialGradientImageType>    TNVDenoisingFilterType;

  /** Pass the ForwardProjection filter to SingleProjectionToFourDFilter */
  void SetForwardProjectionFilter(int fwtype) ITK_OVERRIDE;

  /** Pass the backprojection filter to ProjectionStackToFourD*/
  void SetBackProjectionFilter(int bptype) ITK_OVERRIDE;

  /** Pass the interpolation weights to SingleProjectionToFourDFilter */
  virtual void SetWeights(const itk::Array2D<float> _arg);

//  void PrintTiming(std::ostream& os) const;

  /** Set / Get whether the displaced detector filter should be disabled */
  itkSetMacro(DisableDisplacedDetectorFilter, bool)
  itkGetMacro(DisableDisplacedDetectorFilter, bool)

  // Regularization steps to perform
  itkSetMacro(PerformPositivity, bool)
  itkGetMacro(PerformPositivity, bool)
  itkSetMacro(PerformMotionMask, bool)
  itkGetMacro(PerformMotionMask, bool)
  itkSetMacro(PerformTVSpatialDenoising, bool)
  itkGetMacro(PerformTVSpatialDenoising, bool)
  itkSetMacro(PerformWaveletsSpatialDenoising, bool)
  itkGetMacro(PerformWaveletsSpatialDenoising, bool)
  itkSetMacro(PerformWarping, bool)
  itkGetMacro(PerformWarping, bool)
  itkSetMacro(PerformTVTemporalDenoising, bool)
  itkGetMacro(PerformTVTemporalDenoising, bool)
  itkSetMacro(PerformL0TemporalDenoising, bool)
  itkGetMacro(PerformL0TemporalDenoising, bool)
  itkSetMacro(PerformTNVDenoising, bool)
  itkGetMacro(PerformTNVDenoising, bool)
  itkSetMacro(ComputeInverseWarpingByConjugateGradient, bool)
  itkGetMacro(ComputeInverseWarpingByConjugateGradient, bool)
  itkSetMacro(UseNearestNeighborInterpolationInWarping, bool)
  itkGetMacro(UseNearestNeighborInterpolationInWarping, bool)
  itkGetMacro(CudaConjugateGradient, bool)
  itkSetMacro(CudaConjugateGradient, bool)

  /** Set and Get for the UseCudaCyclicDeformation variable */
  itkSetMacro(UseCudaCyclicDeformation, bool)
  itkGetMacro(UseCudaCyclicDeformation, bool)

  // Regularization parameters
  itkSetMacro(GammaTVSpace, float)
  itkGetMacro(GammaTVSpace, float)
  itkSetMacro(GammaTVTime, float)
  itkGetMacro(GammaTVTime, float)
  itkSetMacro(GammaTNV, float)
  itkGetMacro(GammaTNV, float)
  itkSetMacro(LambdaL0Time, float)
  itkGetMacro(LambdaL0Time, float)
  itkSetMacro(SoftThresholdWavelets, float)
  itkGetMacro(SoftThresholdWavelets, float)
  itkSetMacro(PhaseShift, float)
  itkGetMacro(PhaseShift, float)

  /** Set the number of levels of the wavelets decomposition */
  itkGetMacro(NumberOfLevels, unsigned int)
  itkSetMacro(NumberOfLevels, unsigned int)

  /** Sets the order of the Daubechies wavelet used to deconstruct/reconstruct the image pyramid */
  itkGetMacro(Order, unsigned int)
  itkSetMacro(Order, unsigned int)

  // Iterations
  itkSetMacro(MainLoop_iterations, int)
  itkGetMacro(MainLoop_iterations, int)
  itkSetMacro(CG_iterations, int)
  itkGetMacro(CG_iterations, int)
  itkSetMacro(TV_iterations, int)
  itkGetMacro(TV_iterations, int)
  itkSetMacro(L0_iterations, int)
  itkGetMacro(L0_iterations, int)  

  // Geometry
  itkSetMacro(Geometry, typename ThreeDCircularProjectionGeometry::Pointer)
  itkGetMacro(Geometry, typename ThreeDCircularProjectionGeometry::Pointer)

  /** Store the phase signal in a member variable */
  virtual void SetSignal(const std::vector<double> signal);

protected:
  FourDROOSTERConeBeamReconstructionFilter();
  ~FourDROOSTERConeBeamReconstructionFilter() {}

  /** Does the real work. */
  void GenerateData() ITK_OVERRIDE;

  void GenerateOutputInformation() ITK_OVERRIDE;

  void GenerateInputRequestedRegion() ITK_OVERRIDE;

  // Inputs are not supposed to occupy the same physical space,
  // so there is nothing to verify
  void VerifyInputInformation() ITK_OVERRIDE {}

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
  typename TemporalL0DenoisingFilterType::Pointer         m_L0DenoisingTime;
  typename TNVDenoisingFilterType::Pointer                m_TNVDenoising;

  // Booleans :
  // should warping be performed ?
  // should conjugate gradient be performed on GPU ?
  // should wavelets replace TV in spatial denoising ?
  bool  m_PerformPositivity;
  bool  m_PerformMotionMask;
  bool  m_PerformTVSpatialDenoising;
  bool  m_PerformWaveletsSpatialDenoising;
  bool  m_PerformWarping;
  bool  m_PerformTVTemporalDenoising;
  bool  m_PerformL0TemporalDenoising;
  bool  m_PerformTNVDenoising;
  bool  m_ComputeInverseWarpingByConjugateGradient;
  bool  m_UseNearestNeighborInterpolationInWarping; //Default is false, linear interpolation is used instead
  bool  m_CudaConjugateGradient;
  bool  m_UseCudaCyclicDeformation;
  bool  m_DisableDisplacedDetectorFilter;

  // Regularization parameters
  float m_GammaTVSpace;
  float m_GammaTVTime;
  float m_GammaTNV;
  float m_LambdaL0Time;
  float m_SoftThresholdWavelets;
  float m_PhaseShift;
  bool  m_DimensionsProcessedForTVSpace[VolumeSeriesType::ImageDimension];
  bool  m_DimensionsProcessedForTVTime[VolumeSeriesType::ImageDimension];

  typename itk::ImageToImageFilter<VolumeSeriesType, VolumeSeriesType>::Pointer m_DownstreamFilter;

  /** Information for the wavelets denoising filter */
  unsigned int    m_Order;
  unsigned int    m_NumberOfLevels;

  // Iterations
  int   m_MainLoop_iterations;
  int   m_CG_iterations;
  int   m_TV_iterations;
  int   m_L0_iterations;

  // Geometry
  typename rtk::ThreeDCircularProjectionGeometry::Pointer m_Geometry;

  // Signal
  std::vector<double>                            m_Signal;

//  /** Time probes */
//  itk::TimeProbe m_CGProbe;
//  itk::TimeProbe m_PositivityProbe;
//  itk::TimeProbe m_MotionMaskProbe;
//  itk::TimeProbe m_TVSpatialDenoisingProbe;
//  itk::TimeProbe m_WaveletsSpatialDenoisingProbe;
//  itk::TimeProbe m_TVTemporalDenoisingProbe;
//  itk::TimeProbe m_TNVDenoisingProbe;
//  itk::TimeProbe m_L0TemporalDenoisingProbe;
//  itk::TimeProbe m_WarpingProbe;
//  itk::TimeProbe m_UnwarpingProbe;

private:
  FourDROOSTERConeBeamReconstructionFilter(const Self &); //purposely not implemented
  void operator=(const Self &);  //purposely not implemented

};
} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkFourDROOSTERConeBeamReconstructionFilter.hxx"
#endif

#endif
