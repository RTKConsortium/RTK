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
#ifndef rtkRegularizedConjugateGradientConeBeamReconstructionFilter_h
#define rtkRegularizedConjugateGradientConeBeamReconstructionFilter_h

#include "rtkConjugateGradientConeBeamReconstructionFilter.h"
#ifdef RTK_USE_CUDA
  #include "rtkCudaTotalVariationDenoisingBPDQImageFilter.h"
#else
  #include "rtkTotalVariationDenoisingBPDQImageFilter.h"
#endif
#include "rtkDeconstructSoftThresholdReconstructImageFilter.h"

#include <itkThresholdImageFilter.h>

namespace rtk
{
  /** \class RegularizedConjugateGradientConeBeamReconstructionFilter
   * \brief Performs 3D regularized reconstruction
   *
   * Performs 3D Conjugate Gradient reconstruction, then
   * - Replaces all negative values by zero
   * - Applies total variation denoising in space
   * - Applies wavelets denoising in space
   * and starting over as many times as the number of main loop iterations desired.
   *
   * \dot
   * digraph RegularizedConjugateGradientConeBeamReconstructionFilter {
   *
   * PrimaryInput [label="Primary input (4D sequence of volumes)"];
   * PrimaryInput [shape=Mdiamond];
   * InputProjectionStack [label="Input projection stack"];
   * InputProjectionStack [shape=Mdiamond];
   * Output [label="Output (Reconstruction: 4D sequence of volumes)"];
   * Output [shape=Mdiamond];
   *
   * node [shape=box];
   * CG [ label="rtk::ConjugateGradientConeBeamReconstructionFilter" URL="\ref rtk::ConjugateGradientConeBeamReconstructionFilter"];
   * Positivity [group=regul, label="itk::ThresholdImageFilter (positivity)" URL="\ref itk::ThresholdImageFilter"];
   * TV [group=regul, label="rtk::TotalVariationDenoisingBPDQImageFilter" URL="\ref rtk::TotalVariationDenoisingBPDQImageFilter"];
   * Wavelets [group=regul, label="rtk::DeconstructSoftThresholdReconstructImageFilter" URL="\ref rtk::DeconstructSoftThresholdReconstructImageFilter"];
   * SoftThreshold [group=regul, label="rtk::SoftThresholdImageFilter" URL="\ref rtk::SoftThresholdImageFilter"];
   *
   * AfterPrimaryInput [group=invisible, label="", fixedsize="false", width=0, height=0, shape=none];
   * AfterCG [group=invisible, label="m_PerformPositivity ?", fixedsize="false", width=0, height=0, shape=none];
   * AfterPositivity [group=invisible, label="m_PerformTVSpatialDenoising ?", fixedsize="false", width=0, height=0, shape=none];
   * AfterTV [group=invisible, label="m_PerformWaveletsSpatialDenoising ?", fixedsize="false", width=0, height=0, shape=none];
   * AfterWavelets [group=invisible, label="", fixedsize="false", width=0, height=0, shape=none];
   * AfterSoftThreshold [group=invisible, label="", fixedsize="false", width=0, height=0, shape=none];
   *
   * PrimaryInput -> AfterPrimaryInput [arrowhead=none];
   * AfterPrimaryInput -> CG;
   * InputProjectionStack -> CG;
   * CG -> AfterCG;
   * AfterCG -> Positivity [label="true"];
   * Positivity -> AfterPositivity;
   * AfterPositivity -> TV [label="true"];
   * TV -> AfterTV;
   * AfterTV -> Wavelets [label="true"];
   * Wavelets -> AfterWavelets;
   * AfterWavelets -> SoftThreshold [label="true"];
   * SoftThreshold -> AfterSoftThreshold;
   * AfterSoftThreshold -> Output;
   * AfterSoftThreshold -> AfterPrimaryInput [style=dashed];
   *
   * AfterCG -> AfterPositivity  [label="false"];
   * AfterPositivity -> AfterTV [label="false"];
   * AfterTV -> AfterWavelets [label="false"];
   * AfterWavelets -> AfterSoftThreshold [label="false"];
   *
   * // Invisible edges between the regularization filters
   * edge[style=invis];
   * Positivity -> TV;
   * TV -> Wavelets;
   * Wavelets -> SoftThreshold;
   * }
   * \enddot
   *
   * \test rtkregularizedconjugategradienttest.cxx
   *
   * \author Cyril Mory
   *
   * \ingroup ReconstructionAlgorithm
   */

template< typename TImage>
class RegularizedConjugateGradientConeBeamReconstructionFilter : public rtk::IterativeConeBeamReconstructionFilter<TImage, TImage>
{
public:
  /** Standard class typedefs. */
  typedef RegularizedConjugateGradientConeBeamReconstructionFilter                      Self;
  typedef rtk::IterativeConeBeamReconstructionFilter<TImage, TImage>                    Superclass;
  typedef itk::SmartPointer< Self >                                                     Pointer;
  typedef itk::CovariantVector< typename TImage::ValueType, TImage::ImageDimension>     CovariantVectorForSpatialGradient;

#ifdef RTK_USE_CUDA
  typedef itk::CudaImage<CovariantVectorForSpatialGradient, TImage::ImageDimension>   GradientImageType;
#else
  typedef itk::Image<CovariantVectorForSpatialGradient, TImage::ImageDimension>       GradientImageType;
#endif

  /** Method for creation through the object factory. */
  itkNewMacro(Self)

  /** Run-time type information (and related methods). */
  itkTypeMacro(RegularizedConjugateGradientConeBeamReconstructionFilter, itk::ImageToImageFilter)

  /** The image to be updated.*/
  void SetInputVolume(const TImage* Volume);
  typename TImage::ConstPointer GetInputVolume();

  /** The stack of measured projections */
  void SetInputProjectionStack(const TImage* Projection);
  typename TImage::Pointer   GetInputProjectionStack();

  /** The weights map (Weighted least squares optimization in the conjugate gradient filter)*/
  void SetInputWeights(const TImage* Projection);
  typename TImage::Pointer   GetInputWeights();

  /** Set the support mask, if any, for support constraint in reconstruction */
  void SetSupportMask(const TImage *SupportMask);
  typename TImage::ConstPointer GetSupportMask();

  typedef rtk::ConjugateGradientConeBeamReconstructionFilter<TImage>                    CGFilterType;
  typedef itk::ThresholdImageFilter<TImage>                                             ThresholdFilterType;
  typedef rtk::TotalVariationDenoisingBPDQImageFilter<TImage, GradientImageType>        TVDenoisingFilterType;
  typedef rtk::DeconstructSoftThresholdReconstructImageFilter<TImage>                   WaveletsDenoisingFilterType;
  typedef rtk::SoftThresholdImageFilter<TImage, TImage>                                 SoftThresholdFilterType;

  /** Pass the ForwardProjection filter to SingleProjectionToFourDFilter */
  void SetForwardProjectionFilter(int fwtype) ITK_OVERRIDE;

  /** Pass the backprojection filter to ProjectionStackToFourD*/
  void SetBackProjectionFilter(int bptype) ITK_OVERRIDE;

  void PrintTiming(std::ostream& os) const;

  // Regularization steps to perform
  itkSetMacro(PerformPositivity, bool)
  itkGetMacro(PerformPositivity, bool)
  itkSetMacro(PerformTVSpatialDenoising, bool)
  itkGetMacro(PerformTVSpatialDenoising, bool)
  itkSetMacro(PerformWaveletsSpatialDenoising, bool)
  itkGetMacro(PerformWaveletsSpatialDenoising, bool)
  itkSetMacro(PerformSoftThresholdOnImage, bool)
  itkGetMacro(PerformSoftThresholdOnImage, bool)

  // Regularization parameters
  itkSetMacro(GammaTV, float)
  itkGetMacro(GammaTV, float)
  itkSetMacro(SoftThresholdWavelets, float)
  itkGetMacro(SoftThresholdWavelets, float)
  itkSetMacro(SoftThresholdOnImage, float)
  itkGetMacro(SoftThresholdOnImage, float)

  /** Set the number of levels of the wavelets decomposition */
  itkGetMacro(NumberOfLevels, unsigned int)
  itkSetMacro(NumberOfLevels, unsigned int)

  /** Sets the order of the Daubechies wavelet used to deconstruct/reconstruct the image pyramid */
  itkGetMacro(Order, unsigned int)
  itkSetMacro(Order, unsigned int)

  /** Displays the conjugate gradient cost function at each iteration. */
  itkSetMacro(IterationCosts, bool)
  itkGetMacro(IterationCosts, bool)

  // Iterations
  itkSetMacro(MainLoop_iterations, int)
  itkGetMacro(MainLoop_iterations, int)
  itkSetMacro(CG_iterations, int)
  itkGetMacro(CG_iterations, int)
  itkSetMacro(TV_iterations, int)
  itkGetMacro(TV_iterations, int)

  // Geometry
  itkSetMacro(Geometry, typename ThreeDCircularProjectionGeometry::Pointer)
  itkGetMacro(Geometry, typename ThreeDCircularProjectionGeometry::Pointer)

  /** Preconditioning flag for the conjugate gradient filter */
  itkSetMacro(Preconditioned, bool)
  itkGetMacro(Preconditioned, bool)

  /** Quadratic regularization for the conjugate gradient filter */
  itkSetMacro(RegularizedCG, bool)
  itkGetMacro(RegularizedCG, bool)
  itkSetMacro(Gamma, float)
  itkGetMacro(Gamma, float)

  /** Perform CG operations on GPU ? */
  itkSetMacro(CudaConjugateGradient, bool)
  itkGetMacro(CudaConjugateGradient, bool)

  /** Set / Get whether the displaced detector filter should be disabled */
  itkSetMacro(DisableDisplacedDetectorFilter, bool)
  itkGetMacro(DisableDisplacedDetectorFilter, bool)

protected:
  RegularizedConjugateGradientConeBeamReconstructionFilter();
  ~RegularizedConjugateGradientConeBeamReconstructionFilter() {}

  /** Does the real work. */
  void GenerateData() ITK_OVERRIDE;

  void GenerateOutputInformation() ITK_OVERRIDE;

  void GenerateInputRequestedRegion() ITK_OVERRIDE;

  // Inputs are not supposed to occupy the same physical space,
  // so there is nothing to verify
  void VerifyInputInformation() ITK_OVERRIDE {}

  /** Member pointers to the filters used internally (for convenience)*/
  typename CGFilterType::Pointer                   m_CGFilter;
  typename ThresholdFilterType::Pointer            m_PositivityFilter;
  typename TVDenoisingFilterType::Pointer          m_TVDenoising;
  typename WaveletsDenoisingFilterType::Pointer    m_WaveletsDenoising;
  typename SoftThresholdFilterType::Pointer        m_SoftThresholdFilter;

  // Booleans for each regularization (should it be performed or not)
  // as well as to choose whether CG should be on GPU or not
  bool  m_PerformPositivity;
  bool  m_PerformTVSpatialDenoising;
  bool  m_PerformWaveletsSpatialDenoising;
  bool  m_CudaConjugateGradient;
  bool  m_PerformSoftThresholdOnImage;

  // Regularization parameters
  float m_GammaTV;
  float m_Gamma;
  float m_SoftThresholdWavelets;
  float m_SoftThresholdOnImage;
  bool  m_DimensionsProcessedForTV[TImage::ImageDimension];
  bool  m_Preconditioned;
  bool  m_RegularizedCG;

  /** Information for the wavelets denoising filter */
  unsigned int    m_Order;
  unsigned int    m_NumberOfLevels;

  /** Conjugate gradient parameters */
  bool            m_IterationCosts;
  bool            m_DisableDisplacedDetectorFilter;

  // Iterations
  int   m_MainLoop_iterations;
  int   m_CG_iterations;
  int   m_TV_iterations;

  // Geometry
  typename rtk::ThreeDCircularProjectionGeometry::Pointer m_Geometry;

  /** Time probes */
  itk::TimeProbe m_CGProbe;
  itk::TimeProbe m_PositivityProbe;
  itk::TimeProbe m_TVSpatialDenoisingProbe;
  itk::TimeProbe m_WaveletsSpatialDenoisingProbe;
  itk::TimeProbe m_SoftThresholdImageProbe;

private:
  RegularizedConjugateGradientConeBeamReconstructionFilter(const Self &); //purposely not implemented
  void operator=(const Self &);  //purposely not implemented
};
} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkRegularizedConjugateGradientConeBeamReconstructionFilter.hxx"
#endif

#endif
