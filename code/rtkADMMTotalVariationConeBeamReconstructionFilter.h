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

#ifndef __rtkADMMTotalVariationConeBeamReconstructionFilter_h
#define __rtkADMMTotalVariationConeBeamReconstructionFilter_h

#include <itkImageToImageFilter.h>
#include <itkAddImageFilter.h>
#include <itkSubtractImageFilter.h>
#include <itkMultiplyImageFilter.h>
#include <itkTimeProbe.h>

#include "rtkConjugateGradientImageFilter.h"
#include "rtkSoftThresholdTVImageFilter.h"
#include "rtkADMMTotalVariationConjugateGradientOperator.h"
#include "rtkIterativeConeBeamReconstructionFilter.h"
#include "rtkThreeDCircularProjectionGeometry.h"

namespace rtk
{
  /** \class ADMMTotalVariationConeBeamReconstructionFilter
   * \brief Implements the ADMM reconstruction with total variation regularization
   *
   * This filter implements a reconstruction method based on compressed sensing.
   * The method attempts to find the f that minimizes
   * || Rf -p ||_2^2 + alpha * TV(f),
   * with R the forward projection operator, p the measured projections,
   * and TV the total variation. Details on the method and the calculations can be found in
   *
   * Mory, C., B. Zhang, V. Auvray, M. Grass, D. Schafer, F. Peyrin, S. Rit, P. Douek,
   * and L. Boussel. “ECG-Gated C-Arm Computed Tomography Using L1 Regularization.”
   * In Proceedings of the 20th European Signal Processing Conference (EUSIPCO), 2728–32, 2012.
   *
   * \dot
   * digraph ADMMTotalVariationConeBeamReconstructionFilter {
   *
   * Input0 [ label="Input 0 (Volume)"];
   * Input0 [shape=Mdiamond];
   * Input1 [label="Input 1 (Projections)"];
   * Input1 [shape=Mdiamond];
   * Output [label="Output (Volume)"];
   * Output [shape=Mdiamond];
   *
   * node [shape=box];
   * ZeroMultiplyVolume [label="itk::MultiplyImageFilter (by zero)" URL="\ref itk::MultiplyImageFilter"];
   * ZeroMultiplyGradient [label="itk::MultiplyImageFilter (by zero)" URL="\ref itk::MultiplyImageFilter"];
   * BeforeZeroMultiplyVolume [label="", fixedsize="false", width=0, height=0, shape=none];
   * AfterGradient [label="", fixedsize="false", width=0, height=0, shape=none];
   * AfterZeroMultiplyGradient [label="", fixedsize="false", width=0, height=0, shape=none];
   * Gradient [ label="rtk::ForwardDifferenceGradientImageFilter" URL="\ref rtk::ForwardDifferenceGradientImageFilter"];
   * BackProjection [ label="rtk::BackProjectionImageFilter" URL="\ref rtk::BackProjectionImageFilter"];
   * AddGradient [ label="itk::AddImageFilter" URL="\ref itk::AddImageFilter"];
   * Divergence [ label="rtk::BackwardDifferenceDivergenceImageFilter" URL="\ref rtk::BackwardDifferenceDivergenceImageFilter"];
   * Multiply [ label="itk::MultiplyImageFilter (by beta)" URL="\ref itk::MultiplyImageFilter"];
   * SubtractVolume [ label="itk::SubtractImageFilter" URL="\ref itk::SubtractImageFilter"];
   * ConjugateGradient[ label="rtk::ConjugateGradientImageFilter" URL="\ref rtk::ConjugateGradientImageFilter"];
   * AfterConjugateGradient [label="", fixedsize="false", width=0, height=0, shape=none];
   * GradientTwo [ label="rtk::ForwardDifferenceGradientImageFilter" URL="\ref rtk::ForwardDifferenceGradientImageFilter"];
   * Subtract [ label="itk::SubtractImageFilter" URL="\ref itk::SubtractImageFilter"];
   * TVSoftThreshold [ label="rtk::SoftThresholdTVImageFilter" URL="\ref rtk::SoftThresholdTVImageFilter"];
   * BeforeTVSoftThreshold [label="", fixedsize="false", width=0, height=0, shape=none];
   * AfterTVSoftThreshold [label="", fixedsize="false", width=0, height=0, shape=none];
   * SubtractTwo [ label="itk::SubtractImageFilter" URL="\ref itk::SubtractImageFilter"];
   *
   * Input0 -> BeforeZeroMultiplyVolume [arrowhead=None];
   * BeforeZeroMultiplyVolume -> ZeroMultiplyVolume;
   * BeforeZeroMultiplyVolume -> Gradient;
   * BeforeZeroMultiplyVolume -> ConjugateGradient;
   * Input1 -> BackProjection;
   * Gradient -> AfterGradient [arrowhead=None];
   * AfterGradient -> AddGradient;
   * AfterGradient -> ZeroMultiplyGradient;
   * ZeroMultiplyGradient -> AfterZeroMultiplyGradient [arrowhead=None];
   * AfterZeroMultiplyGradient -> AddGradient;
   * AfterZeroMultiplyGradient -> Subtract;
   * AddGradient -> Divergence;
   * Divergence -> Multiply;
   * Multiply -> SubtractVolume;
   * BackProjection -> SubtractVolume;
   * SubtractVolume -> ConjugateGradient;
   * ConjugateGradient -> AfterConjugateGradient;
   * AfterConjugateGradient -> GradientTwo;
   * GradientTwo -> Subtract;
   * Subtract -> BeforeTVSoftThreshold [arrowhead=None];
   * BeforeTVSoftThreshold -> TVSoftThreshold;
   * BeforeTVSoftThreshold -> SubtractTwo;
   * TVSoftThreshold -> AfterTVSoftThreshold [arrowhead=None];
   * AfterTVSoftThreshold -> SubtractTwo;
   *
   * AfterTVSoftThreshold -> AfterGradient [style=dashed];
   * SubtractTwo -> AfterZeroMultiplyGradient [style=dashed];
   * AfterConjugateGradient -> BeforeZeroMultiplyVolume [style=dashed];
   * AfterConjugateGradient -> Output [style=dashed];
   * }
   * \enddot
   *
   * \test rtkadmmtotalvariationtest.cxx
   *
   * \author Cyril Mory
   *
   * \ingroup ReconstructionAlgorithm
   */

template< typename TOutputImage, typename TGradientOutputImage = 
    itk::Image< itk::CovariantVector < typename TOutputImage::ValueType, TOutputImage::ImageDimension >, 
    TOutputImage::ImageDimension > >
class ADMMTotalVariationConeBeamReconstructionFilter : public rtk::IterativeConeBeamReconstructionFilter<TOutputImage, TOutputImage>
{
public:
    /** Standard class typedefs. */
    typedef ADMMTotalVariationConeBeamReconstructionFilter       Self;
    typedef itk::ImageToImageFilter<TOutputImage, TOutputImage>  Superclass;
    typedef itk::SmartPointer< Self >                            Pointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(ADMMTotalVariationConeBeamReconstructionFilter, itk::ImageToImageFilter)

    /** The 3D image to be updated */
    void SetInputVolume(const TOutputImage* Volume);

    /** The gated measured projections */
    void SetInputProjectionStack(const TOutputImage* Projection);

    typedef rtk::ForwardProjectionImageFilter< TOutputImage, TOutputImage >               ForwardProjectionFilterType;
    typedef typename ForwardProjectionFilterType::Pointer                                 ForwardProjectionFilterPointer;
    typedef rtk::BackProjectionImageFilter< TOutputImage, TOutputImage >                  BackProjectionFilterType;
    typedef typename BackProjectionFilterType::Pointer                                    BackProjectionFilterPointer;
    typedef rtk::ConjugateGradientImageFilter<TOutputImage>                               ConjugateGradientFilterType;
    typedef ForwardDifferenceGradientImageFilter<TOutputImage, 
            typename TOutputImage::ValueType, 
            typename TOutputImage::ValueType, 
            TGradientOutputImage>                            ImageGradientFilterType;
    typedef rtk::BackwardDifferenceDivergenceImageFilter
        <TGradientOutputImage, TOutputImage>                 ImageDivergenceFilterType;
    typedef rtk::SoftThresholdTVImageFilter
        <TGradientOutputImage>                               SoftThresholdTVFilterType;
    typedef itk::SubtractImageFilter<TOutputImage>                                        SubtractVolumeFilterType;
    typedef itk::AddImageFilter<TGradientOutputImage>        AddGradientsFilterType;
    typedef itk::MultiplyImageFilter<TOutputImage>                                        MultiplyVolumeFilterType;
    typedef itk::MultiplyImageFilter<TGradientOutputImage>   MultiplyGradientFilterType;
    typedef itk::SubtractImageFilter<TGradientOutputImage>   SubtractGradientsFilterType;
    typedef rtk::ADMMTotalVariationConjugateGradientOperator<TOutputImage>                CGOperatorFilterType;

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

    itkSetMacro(MeasureExecutionTimes, bool)
    itkGetMacro(MeasureExecutionTimes, bool)

protected:
    ADMMTotalVariationConeBeamReconstructionFilter();
    ~ADMMTotalVariationConeBeamReconstructionFilter(){}

    /** Does the real work. */
    virtual void GenerateData();

    /** Member pointers to the filters used internally (for convenience)*/
    typename SubtractGradientsFilterType::Pointer                               m_SubtractFilter2; 
    typename SubtractGradientsFilterType::Pointer                               m_SubtractFilter3;
    typename MultiplyVolumeFilterType::Pointer                                  m_MultiplyFilter;
    typename MultiplyVolumeFilterType::Pointer                                  m_ZeroMultiplyVolumeFilter;
    typename MultiplyGradientFilterType::Pointer                                m_ZeroMultiplyGradientFilter;
    typename ImageGradientFilterType::Pointer                                   m_GradientFilter1; 
    typename ImageGradientFilterType::Pointer                                   m_GradientFilter2;
    typename SubtractVolumeFilterType::Pointer                                  m_SubtractVolumeFilter;
    typename AddGradientsFilterType::Pointer                                    m_AddGradientsFilter;
    typename ImageDivergenceFilterType::Pointer                                 m_DivergenceFilter;
    typename ConjugateGradientFilterType::Pointer                               m_ConjugateGradientFilter;
    typename SoftThresholdTVFilterType::Pointer                                 m_SoftThresholdFilter;
    typename CGOperatorFilterType::Pointer                                      m_CGOperator;
    typename ForwardProjectionImageFilter<TOutputImage, TOutputImage>::Pointer  m_ForwardProjectionFilter;
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
    ADMMTotalVariationConeBeamReconstructionFilter(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented

    float           m_Alpha;
    float           m_Beta;
    unsigned int    m_AL_iterations;
    unsigned int    m_CG_iterations;
    bool            m_MeasureExecutionTimes;

    ThreeDCircularProjectionGeometry::Pointer m_Geometry;


};
} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkADMMTotalVariationConeBeamReconstructionFilter.txx"
#endif

#endif
