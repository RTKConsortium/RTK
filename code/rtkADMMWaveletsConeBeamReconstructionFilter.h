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

#ifndef __rtkADMMWaveletsConeBeamReconstructionFilter_h
#define __rtkADMMWaveletsConeBeamReconstructionFilter_h

#include <itkImageToImageFilter.h>
#include <itkAddImageFilter.h>
#include <itkSubtractImageFilter.h>
#include <itkMultiplyImageFilter.h>
#include <itkTimeProbe.h>

#include "rtkConjugateGradientImageFilter.h"
#include "giftDeconstructSoftThresholdReconstructImageFilter.h"
#include "rtkADMMWaveletsConjugateGradientOperator.h"
#include "rtkIterativeConeBeamReconstructionFilter.h"
#include "rtkThreeDCircularProjectionGeometry.h"

namespace rtk
{
  /** \class ADMMWaveletsConeBeamReconstructionFilter
   * \brief Implements the ADMM reconstruction with wavelets regularization
   *
   * This filter implements the operator A used in the conjugate gradient step
   * of a reconstruction method based on compressed sensing. The method attempts
   * to find the f that minimizes || Rf -p ||_2^2 + alpha * || W(f) ||_1, with R the
   * forward projection operator, p the measured projections, and W the
   * Daubechies wavelets transform.
   * Details on the method and the calculations can be found in
   *
   * Mory, C., B. Zhang, V. Auvray, M. Grass, D. Schafer, F. Peyrin, S. Rit, P. Douek,
   * and L. Boussel. “ECG-Gated C-Arm Computed Tomography Using L1 Regularization.”
   * In Proceedings of the 20th European Signal Processing Conference (EUSIPCO), 2728–32, 2012.
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
   * SoftThreshold [ label="gift::DeconstructSoftThresholdReconstructImageFilter" URL="\ref gift::DeconstructSoftThresholdReconstructImageFilter"];
   * BeforeSoftThreshold [label="", fixedsize="false", width=0, height=0, shape=none];
   * AfterSoftThreshold [label="", fixedsize="false", width=0, height=0, shape=none];
   * SubtractTwo [ label="itk::SubtractImageFilter" URL="\ref itk::SubtractImageFilter"];
   *
   * Input0 -> BeforeZeroMultiply [arrowhead=None];
   * BeforeZeroMultiply -> ZeroMultiply;
   * ZeroMultiply -> AfterZeroMultiply;
   * BeforeZeroMultiply -> ConjugateGradient;
   * Input1 -> BackProjection;
   * AfterZeroMultiply -> D;
   * AfterZeroMultiply -> G;
   * AfterZeroMultiply -> BackProjection;
   * D -> Add;
   * G -> Add;
   * D -> Subtract;
   * Add -> Multiply;
   * Multiply -> AddTwo;
   * BackProjection -> AddTwo;
   * AddTwo -> ConjugateGradient;
   * ConjugateGradient -> AfterConjugateGradient;
   * AfterConjugateGradient -> Subtract;
   * Subtract -> BeforeSoftThreshold [arrowhead=None];
   * BeforeSoftThreshold -> SoftThreshold;
   * BeforeSoftThreshold -> SubtractTwo;
   * SoftThreshold -> AfterSoftThreshold [arrowhead=None];
   * AfterSoftThreshold -> SubtractTwo;
   *
   * AfterSoftThreshold -> G [style=dashed];
   * SubtractTwo -> D [style=dashed];
   * AfterConjugateGradient -> BeforeZeroMultiply [style=dashed];
   * AfterConjugateGradient -> Output [style=dashed];
   * }
   * \enddot
   *
   * \test rtkadmmwaveletstest.cxx
   *
   * \author Cyril Mory
   *
   * \ingroup ReconstructionAlgorithm
   */

template< typename TOutputImage>
class ADMMWaveletsConeBeamReconstructionFilter : public rtk::IterativeConeBeamReconstructionFilter<TOutputImage, TOutputImage>
{
public:
    /** Standard class typedefs. */
    typedef ADMMWaveletsConeBeamReconstructionFilter       Self;
    typedef itk::ImageToImageFilter<TOutputImage, TOutputImage>  Superclass;
    typedef itk::SmartPointer< Self >                            Pointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(ADMMWaveletsConeBeamReconstructionFilter, itk::ImageToImageFilter)

    /** The 3D image to be updated */
    void SetInputVolume(const TOutputImage* Volume);

    /** The gated measured projections */
    void SetInputProjectionStack(const TOutputImage* Projection);

    typedef rtk::ForwardProjectionImageFilter< TOutputImage, TOutputImage >               ForwardProjectionFilterType;
    typedef rtk::BackProjectionImageFilter< TOutputImage, TOutputImage >                  BackProjectionFilterType;
    typedef rtk::ConjugateGradientImageFilter<TOutputImage>                               ConjugateGradientFilterType;
    typedef itk::SubtractImageFilter<TOutputImage>                                        SubtractFilterType;
    typedef itk::AddImageFilter<TOutputImage>                                             AddFilterType;
    typedef itk::MultiplyImageFilter<TOutputImage>                                        MultiplyFilterType;
    typedef rtk::ADMMWaveletsConjugateGradientOperator<TOutputImage>                      CGOperatorFilterType;
    typedef gift::DeconstructSoftThresholdReconstructImageFilter<TOutputImage>            SoftThresholdFilterType;

    /** Pass the ForwardProjection filter to the conjugate gradient operator */
    void SetForwardProjectionFilter (int _arg);

    /** Pass the backprojection filter to the conjugate gradient operator and to the back projection filter generating the B of AX=B */
    void SetBackProjectionFilter (int _arg);

    /** Pass the geometry to all filters needing it */
    itkSetMacro(Geometry, ThreeDCircularProjectionGeometry::Pointer)

    /** Increase the value of Beta at each iteration */
    void SetBetaForCurrentIteration(int iter);

    itkSetMacro(Alpha, float)
    itkGetMacro(Alpha, float)

    itkSetMacro(Beta, float)
    itkGetMacro(Beta, float)

    itkSetMacro(AL_iterations, float)
    itkGetMacro(AL_iterations, float)

    itkSetMacro(CG_iterations, float)
    itkGetMacro(CG_iterations, float)

    itkSetMacro(WaveletsOrder, unsigned int)
    itkGetMacro(WaveletsOrder, unsigned int)

    itkSetMacro(NumberOfLevels, unsigned int)
    itkGetMacro(NumberOfLevels, unsigned int)

    void PrintTiming(std::ostream& os) const;

protected:
    ADMMWaveletsConeBeamReconstructionFilter();
    ~ADMMWaveletsConeBeamReconstructionFilter(){}

    /** Does the real work. */
    virtual void GenerateData();

    /** Member pointers to the filters used internally (for convenience)*/
    typename SubtractFilterType::Pointer                                        m_SubtractFilter1;
    typename SubtractFilterType::Pointer                                        m_SubtractFilter2;
    typename MultiplyFilterType::Pointer                                        m_MultiplyFilter;
    typename MultiplyFilterType::Pointer                                        m_ZeroMultiplyFilter;
    typename AddFilterType::Pointer                                             m_AddFilter1;
    typename AddFilterType::Pointer                                             m_AddFilter2;
    typename ConjugateGradientFilterType::Pointer                               m_ConjugateGradientFilter;
    typename SoftThresholdFilterType::Pointer                                   m_SoftThresholdFilter;
    typename CGOperatorFilterType::Pointer                                      m_CGOperator;
    typename ForwardProjectionImageFilter<TOutputImage, TOutputImage>::Pointer  m_ForwardProjectionFilterForConjugateGradient;
    typename BackProjectionImageFilter<TOutputImage, TOutputImage>::Pointer     m_BackProjectionFilterForConjugateGradient;
    typename BackProjectionImageFilter<TOutputImage, TOutputImage>::Pointer     m_BackProjectionFilter;

    /** The inputs of this filter have the same type (float, 3) but not the same meaning
    * It is normal that they do not occupy the same physical space. Therefore this check
    * must be removed */
    void VerifyInputInformation(){}

    /** The volume and the projections must have different requested regions
    */
    void GenerateInputRequestedRegion();
    void GenerateOutputInformation();

private:
    ADMMWaveletsConeBeamReconstructionFilter(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented

    float           m_Alpha;
    float           m_Beta;
    unsigned int    m_AL_iterations;
    unsigned int    m_CG_iterations;
    unsigned int    m_WaveletsOrder;
    unsigned int    m_NumberOfLevels;

    ThreeDCircularProjectionGeometry::Pointer m_Geometry;

    /** Time probes */
    itk::TimeProbe m_BeforeConjugateGradientProbe;
    itk::TimeProbe m_ConjugateGradientProbe;
    itk::TimeProbe m_WaveletsSoftTresholdingProbe;

};
} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkADMMWaveletsConeBeamReconstructionFilter.txx"
#endif

#endif
