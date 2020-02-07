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

#ifndef rtkADMMWaveletsConeBeamReconstructionFilter_h
#define rtkADMMWaveletsConeBeamReconstructionFilter_h

#include <itkImageToImageFilter.h>
#include <itkAddImageFilter.h>
#include <itkSubtractImageFilter.h>
#include <itkMultiplyImageFilter.h>

#include "rtkConjugateGradientImageFilter.h"
#include "rtkDeconstructSoftThresholdReconstructImageFilter.h"
#include "rtkADMMWaveletsConjugateGradientOperator.h"
#include "rtkIterativeConeBeamReconstructionFilter.h"
#include "rtkThreeDCircularProjectionGeometry.h"
#include "rtkDisplacedDetectorImageFilter.h"

namespace rtk
{
/** \class ADMMWaveletsConeBeamReconstructionFilter
 * \brief Implements the ADMM reconstruction with wavelets regularization
 *
 * This filter implements the operator A used in the conjugate gradient step
 * of a reconstruction method based on compressed sensing. The method attempts
 * to find the f that minimizes || Rf -p ||_2^2 + alpha * || W(f) ||_1, with R the
 * forward projection operator, p the measured projections, and W the
 * Daubechies wavelets transform. Note that since Daubechies wavelets are orthogonal,
 * \f$ W^{T} = W^{-1} \f$
 * Details on the method and the calculations can be found on page 53 of
 *
 * Mory, C. "Cardiac C-Arm Computed Tomography", PhD thesis, 2014.
 * https://hal.inria.fr/tel-00985728/document
 *
 * \f$ f_{k+1} \f$ is obtained by linear conjugate solving the following:
 * \f[ ( R^T R + \beta I ) f = R^T p + \beta W^{-1} ( g_k + d_k ) \f]
 *
 * \f$ g_{k+1} \f$ is obtained by soft thresholding:
 * \f[ g_{k+1} = ST_{ \frac{\alpha}{2 \beta} } ( W f_{k+1} - d_k ) \f]
 *
 * \f$ d_{k+1} \f$ is a simple subtraction:
 * \f[ d_{k+1} = g_{k+1} - ( W f_{k+1} - d_k) \f]
 *
 * In ITK, it is much easier to store a volume than its wavelets decomposition.
 * Therefore, we store \f$ g'_k = W^{-1} g_k \f$ and \f$ d'_k = W^{-1} d_k \f$
 * instead of \f$ g_k \f$ and \f$ d_k \f$. The above algorithm can therefore be re-written as follows:
 *
 * \f$ f_{k+1} \f$ is obtained by linear conjugate solving the following:
 * \f[ ( R^T R + \beta I ) f = R^T p + \beta ( g'_k + d'_k ) \f]
 *
 * \f$ g'_{k+1} \f$ is obtained by soft thresholding:
 * \f[ g'_{k+1} = W^{-1} ( ST_{ \frac{\alpha}{2 \beta} } W( f_{k+1} - d'_k )) \f]
 *
 * \f$ d'_{k+1} \f$ is a simple subtraction:
 * \f[ d'_{k+1} = g'_{k+1} - ( f_{k+1} - d'_k) \f]
 *
 * The wavelet decomposition and reconstruction steps are therefore performed only for
 * soft thresholding.
 *
 * \dot
 * digraph ADMMWaveletsConeBeamReconstructionFilter {
 *
 * Input0 [ label="Input 0 (Volume)"];
 * Input0 [shape=Mdiamond];
 * Input1 [label="Input 1 (Projections)"];
 * Input1 [shape=Mdiamond];
 * Output [label="Output (Volume)"];
 * Output [shape=Mdiamond];
 *
 * node [shape=box];
 * ZeroMultiply [label="itk::MultiplyImageFilter (by zero)" URL="\ref itk::MultiplyImageFilter"];
 * BeforeZeroMultiply [label="", fixedsize="false", width=0, height=0, shape=none];
 * AfterZeroMultiply [label="", fixedsize="false", width=0, height=0, shape=none];
 * D [label="", fixedsize="false", width=0, height=0, shape=none];
 * G [label="", fixedsize="false", width=0, height=0, shape=none];
 * BackProjection [ label="rtk::BackProjectionImageFilter" URL="\ref rtk::BackProjectionImageFilter"];
 * Add [ label="itk::AddImageFilter" URL="\ref itk::AddImageFilter"];
 * Multiply [ label="itk::MultiplyImageFilter (by beta)" URL="\ref itk::MultiplyImageFilter"];
 * AddTwo [ label="itk::AddImageFilter" URL="\ref itk::AddImageFilter"];
 * ConjugateGradient[ label="rtk::ConjugateGradientImageFilter" URL="\ref rtk::ConjugateGradientImageFilter"];
 * AfterConjugateGradient [label="", fixedsize="false", width=0, height=0, shape=none];
 * Subtract [ label="itk::SubtractImageFilter" URL="\ref itk::SubtractImageFilter"];
 * SoftThreshold [ label="rtk::DeconstructSoftThresholdReconstructImageFilter"
 *                 URL="\ref rtk::DeconstructSoftThresholdReconstructImageFilter"];
 * BeforeSoftThreshold [label="", fixedsize="false", width=0, height=0, shape=none];
 * AfterSoftThreshold [label="", fixedsize="false", width=0, height=0, shape=none];
 * SubtractTwo [ label="itk::SubtractImageFilter" URL="\ref itk::SubtractImageFilter"];
 * Displaced [ label="rtk::DisplacedDetectorImageFilter" URL="\ref rtk::DisplacedDetectorImageFilter"];
 *
 * Input0 -> BeforeZeroMultiply [arrowhead=none];
 * BeforeZeroMultiply -> ZeroMultiply;
 * BeforeZeroMultiply -> G [label="g'_0"];
 * ZeroMultiply -> AfterZeroMultiply;
 * BeforeZeroMultiply -> ConjugateGradient [label="f_0"];
 * Input1 -> Displaced;
 * Displaced -> BackProjection;
 * AfterZeroMultiply -> D [label="d'_0"];
 * AfterZeroMultiply -> BackProjection;
 * D -> Add;
 * G -> Add;
 * D -> Subtract;
 * Add -> Multiply [label="d'_k + g'_k"];
 * Multiply -> AddTwo [label="beta (d'_k + g'_k)"];
 * BackProjection -> AddTwo;
 * AddTwo -> ConjugateGradient [label="b"];
 * ConjugateGradient -> AfterConjugateGradient [label="f_k+1"];
 * AfterConjugateGradient -> Subtract;
 * Subtract -> BeforeSoftThreshold [arrowhead=none, label="f_k+1 - d'k"];
 * BeforeSoftThreshold -> SoftThreshold;
 * BeforeSoftThreshold -> SubtractTwo;
 * SoftThreshold -> AfterSoftThreshold [arrowhead=none];
 * AfterSoftThreshold -> SubtractTwo [label="g'_k+1"];
 * AfterSoftThreshold -> G [style=dashed, constraint=false];
 * SubtractTwo -> D [style=dashed, label="d'_k+1"];
 * AfterConjugateGradient -> BeforeZeroMultiply [style=dashed];
 * AfterConjugateGradient -> Output [style=dashed];
 * }
 * \enddot
 *
 * \test rtkadmmwaveletstest.cxx
 *
 * \author Cyril Mory
 *
 * \ingroup RTK ReconstructionAlgorithm
 */

template <typename TOutputImage>
class ADMMWaveletsConeBeamReconstructionFilter
  : public rtk::IterativeConeBeamReconstructionFilter<TOutputImage, TOutputImage>
{
public:
  ITK_DISALLOW_COPY_AND_ASSIGN(ADMMWaveletsConeBeamReconstructionFilter);

  /** Standard class type alias. */
  using Self = ADMMWaveletsConeBeamReconstructionFilter;
  using Superclass = IterativeConeBeamReconstructionFilter<TOutputImage, TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ADMMWaveletsConeBeamReconstructionFilter, itk::ImageToImageFilter);

  //    /** The 3D image to be updated */
  //    void SetInputVolume(const TOutputImage* Volume);

  //    /** The gated measured projections */
  //    void SetInputProjectionStack(const TOutputImage* Projection);

  using ForwardProjectionFilterType = rtk::ForwardProjectionImageFilter<TOutputImage, TOutputImage>;
  using BackProjectionFilterType = rtk::BackProjectionImageFilter<TOutputImage, TOutputImage>;
  using ConjugateGradientFilterType = rtk::ConjugateGradientImageFilter<TOutputImage>;
  using SubtractFilterType = itk::SubtractImageFilter<TOutputImage>;
  using AddFilterType = itk::AddImageFilter<TOutputImage>;
  using MultiplyFilterType = itk::MultiplyImageFilter<TOutputImage>;
  using CGOperatorFilterType = rtk::ADMMWaveletsConjugateGradientOperator<TOutputImage>;
  using SoftThresholdFilterType = rtk::DeconstructSoftThresholdReconstructImageFilter<TOutputImage>;
  using DisplacedDetectorFilterType = rtk::DisplacedDetectorImageFilter<TOutputImage>;

  using ForwardProjectionType = typename Superclass::ForwardProjectionType;
  using BackProjectionType = typename Superclass::BackProjectionType;

  /** Pass the ForwardProjection filter to the conjugate gradient operator */
  void
  SetForwardProjectionFilter(ForwardProjectionType _arg) override;

  /** Pass the backprojection filter to the conjugate gradient operator and to the back projection filter generating the
   * B of AX=B */
  void
  SetBackProjectionFilter(BackProjectionType _arg) override;

  /** Pass the geometry to all filters needing it */
  itkSetObjectMacro(Geometry, ThreeDCircularProjectionGeometry);

  itkSetMacro(Alpha, float);
  itkGetMacro(Alpha, float);

  itkSetMacro(Beta, float);
  itkGetMacro(Beta, float);

  itkSetMacro(AL_iterations, float);
  itkGetMacro(AL_iterations, float);

  itkSetMacro(CG_iterations, float);
  itkGetMacro(CG_iterations, float);

  itkSetMacro(Order, unsigned int);
  itkGetMacro(Order, unsigned int);

  itkSetMacro(NumberOfLevels, unsigned int);
  itkGetMacro(NumberOfLevels, unsigned int);

  /** Set / Get whether the displaced detector filter should be disabled */
  itkSetMacro(DisableDisplacedDetectorFilter, bool);
  itkGetMacro(DisableDisplacedDetectorFilter, bool);

protected:
  ADMMWaveletsConeBeamReconstructionFilter();
  ~ADMMWaveletsConeBeamReconstructionFilter() override = default;

  /** Does the real work. */
  void
  GenerateData() override;

  /** Member pointers to the filters used internally (for convenience)*/
  typename SubtractFilterType::Pointer          m_SubtractFilter1;
  typename SubtractFilterType::Pointer          m_SubtractFilter2;
  typename MultiplyFilterType::Pointer          m_MultiplyFilter;
  typename MultiplyFilterType::Pointer          m_ZeroMultiplyFilter;
  typename AddFilterType::Pointer               m_AddFilter1;
  typename AddFilterType::Pointer               m_AddFilter2;
  typename ConjugateGradientFilterType::Pointer m_ConjugateGradientFilter;
  typename SoftThresholdFilterType::Pointer     m_SoftThresholdFilter;
  typename CGOperatorFilterType::Pointer        m_CGOperator;
  typename ForwardProjectionImageFilter<TOutputImage, TOutputImage>::Pointer
                                                                          m_ForwardProjectionFilterForConjugateGradient;
  typename BackProjectionImageFilter<TOutputImage, TOutputImage>::Pointer m_BackProjectionFilterForConjugateGradient;
  typename BackProjectionImageFilter<TOutputImage, TOutputImage>::Pointer m_BackProjectionFilter;
  typename DisplacedDetectorFilterType::Pointer                           m_DisplacedDetectorFilter;

  /** The inputs of this filter have the same type (float, 3) but not the same meaning
   * It is normal that they do not occupy the same physical space. Therefore this check
   * must be removed */
  void
  VerifyInputInformation() const override
  {}

  /** The volume and the projections must have different requested regions
   */
  void
  GenerateInputRequestedRegion() override;
  void
  GenerateOutputInformation() override;

private:
  float        m_Alpha{ 1 };
  float        m_Beta{ 1 };
  unsigned int m_AL_iterations{ 10 };
  unsigned int m_CG_iterations{ 3 };
  unsigned int m_Order{ 3 };
  unsigned int m_NumberOfLevels{ 5 };
  bool         m_DisableDisplacedDetectorFilter;

  ThreeDCircularProjectionGeometry::Pointer m_Geometry;
};
} // namespace rtk


#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkADMMWaveletsConeBeamReconstructionFilter.hxx"
#endif

#endif
